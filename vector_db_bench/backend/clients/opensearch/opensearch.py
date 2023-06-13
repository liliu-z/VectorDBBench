"""Wrapper around the AliCloud Opensearch"""
import logging
from contextlib import contextmanager
from typing import Type, Iterable
from ..milvus.milvus import Milvus
from ..api import VectorDB, DBConfig, DBCaseConfig, IndexType
from .config import OpensearchConfig, OpensearchIndexConfig
import numpy as np
from datetime import datetime
import json

log = logging.getLogger(__name__)


class OpenSearch(VectorDB):
    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: OpensearchIndexConfig,
        table: str = "anns_table",
        partitionKey: str = "pt",
        vectorDelimiter: str = ",",
        indexName: str = "anns_index",
        id_col_name: str = "id",
        vector_col_name: str = "vector",
        drop_old: bool = False,
    ):
        self.dim = dim
        self.db_config = db_config
        self.db_case_config = db_case_config
        self.table = table
        self.doneTable = f"{table}_done"
        self.partitionKey = partitionKey
        self.vectorDelimiter = vectorDelimiter
        self.indexName = indexName
        self.id_col_name = id_col_name
        self.vector_col_name = vector_col_name
        self.partition = np.random.randint(999999)

        if drop_old:
            from odps import ODPS

            mcClient = ODPS(
                self.db_config.get("access_id"),
                self.db_config.get("access_key"),
                self.db_config.get("project_name"),
                endpoint=self.db_config.get("mc_endpoint"),
            )
            # mcClient.delete_table(self.table, if_exists=True)
            # mcClient.delete_table(self.doneTable, if_exists=True)
            self._createMCTable(mcClient)

    def _createMCTable(self, mcClient):
        from odps.models import TableSchema, Column, Partition

        columns = [
            Column(name=self.id_col_name, type="int", comment="id column"),
            Column(
                name=self.vector_col_name,
                type="string",
                comment="the vector column, float32[] => string",
            ),
        ]
        partitions = [
            Partition(name=self.partitionKey, type="string", comment="the partition")
        ]
        schema = TableSchema(columns=columns, partitions=partitions)
        mcClient.create_table(self.table, schema, if_not_exists=True)

        doneColumns = [
            Column(
                name="attribute",
                type="string",
                comment='example: {"swift_start_timestamp":1642003200}',
            )
        ]
        doneSchema = TableSchema(columns=doneColumns, partitions=partitions)
        mcClient.create_table(self.doneTable, doneSchema, if_not_exists=True)

    @classmethod
    def config_cls(cls) -> Type[DBConfig]:
        return OpensearchConfig

    @classmethod
    def case_config_cls(cls, index_type: IndexType | None = None) -> Type[DBCaseConfig]:
        return OpensearchIndexConfig

    @contextmanager
    def init(self):
        from odps import ODPS

        mcClient = ODPS(
            self.db_config.get("access_id"),
            self.db_config.get("access_key"),
            self.db_config.get("project_name"),
            endpoint=self.db_config.get("mc_endpoint"),
        )
        self.mcClient = mcClient
        
        from alibabacloud_ha3engine import models, client
        Config = models.Config(
            endpoint=self.db_config.get("os_endpoint"),
            instance_id=self.db_config.get("instance_id"),
            protocol="http",
            access_user_name=self.db_config.get("access_user_name"),
            access_pass_word=self.db_config.get("access_password"),
        )
        self.ha3Client = client.Client(Config)

        yield
        self.mcClient = None
        self.ha3Client = None
        del self.mcClient
        del self.ha3Client

    def insert_embeddings(
        self,
        embeddings: Iterable[list[float]],
        metadata: list[int],
    ) -> int:
        assert self.mcClient is not None, "should self.init() first"
        log.warning('hello insert')

        insert_data = [
            [metadata[i], self.vectorDelimiter.join([str(v) for v in embeddings[i]])]
            for i in range(len(embeddings))
        ]
        self.mcClient.write_table(
            self.table,
            insert_data,
            partition=f"{self.partitionKey}={self.partition}",
            create_partition=True,
        )
        return len(metadata)

    def _insert_done_signal(self):
        doneRecords = [
            ['{"swift_start_timestamp":' + str(int(datetime.now().timestamp())) + "}"]
        ]
        self.mcClient.write_table(
            self.doneTable,
            doneRecords,
            partition=f"{self.partitionKey}={self.partition}",
            create_partition=True,
        )

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        filters: dict | None = None,
    ) -> list[int]:
        assert self.ha3Client is not None, "should self.init() first"
        from alibabacloud_ha3engine import models
        queryVectorString = self.vectorDelimiter.join([str(v) for v in query])
        sql_str=f"""cluster=general&&config=start:0,hit:{k},format:json&&query=index_vector:'{queryVectorString}&n={k}'"""
        sqlsearchQuery = models.SearchQuery(query=sql_str)
        optionsHeaders = {}
        sqlSearchRequestModel = models.SearchRequestModel(headers=optionsHeaders, query=sqlsearchQuery)
        sqlstrSearchResponseModel = self.ha3Client.search(sqlSearchRequestModel)
        items = json.loads(sqlstrSearchResponseModel.to_map().get("body", "{}")).get("result", {}).get("items", [])
        res = [int(item.get('fields', {}).get('id', 0)) for item in items]
        return res

    def ready_to_search(self):
        # After loading data into Odps, should insert a signal to doneTable to build index automatically.
        # If it is the first time to test, we must mannually build index.
        self._insert_done_signal()
        partition = f"{self.partitionKey}={self.partition}"
        table = self.table
        log.warning(
            f"""Go to Opensearch Web Console. 
            dim: {self.dim}, partition: {partition}, table: {table}.
            Make sure the index has been built then [Press Enter] to continue."""
        )
        input()
        """ready_to_search will be called between insertion and search in performance cases."""
        pass

    def ready_to_load(self):
        """ready_to_load will be called before load in load cases."""
        pass

from pydantic import BaseModel, SecretStr
from ..api import DBCaseConfig, DBConfig, MetricType
import logging

log = logging.getLogger(__name__)


class OpensearchConfig(DBConfig, BaseModel):
    project_name: str | None = None
    mc_endpoint: str | None = None
    access_id: SecretStr | None = None
    access_key: SecretStr | None = None

    os_endpoint: str | None = None
    instance_id: str | None = None
    access_user_name: str | None = None
    access_password: SecretStr | None = None

    def to_dict(self) -> dict:
        return {
            "project_name": self.project_name,
            "mc_endpoint": self.mc_endpoint,
            "access_id": self.access_id.get_secret_value(),
            "access_key": self.access_key.get_secret_value(),
            "os_endpoint": self.os_endpoint,
            "instance_id": self.instance_id,
            "access_user_name": self.access_user_name,
            "access_password": self.access_password.get_secret_value(),
        }


class OpensearchIndexConfig(BaseModel, DBCaseConfig):
    ef: int = 500
    metric_type: MetricType | None = None

    def parse_metric(self) -> str:
        if self.metric_type == MetricType.L2:
            return "l2_norm"
        elif self.metric_type == MetricType.IP:
            return "dot_product"

        log.warn("only support L2 and IP")
        return "l2_norm"

    def index_param(self) -> dict:
        raise {}

    def search_param(self) -> dict:
        raise {"ef": self.ef}

from enum import Enum, IntEnum
from vector_db_bench.models import DB, CaseType, IndexType, CaseConfigParamType
from pydantic import BaseModel
import typing

# style const
CHECKBOX_MAX_COLUMNS = 4
DB_CONFIG_INPUT_MAX_COLUMNS = 2
CASE_CONFIG_INPUT_MAX_COLUMNS = 4
DB_CONFIG_INPUT_WIDTH_RADIO = 2
CASE_CONFIG_INPUT_WIDTH_RADIO = 1.2
CASE_INTRO_RATIO = 3
MAX_STREAMLIT_INT = (1 << 53) - 1

COLOR_SCHEME = ["#8dd3c7", "#ffffb3", "#bebada", "#fb8072", "#80b1d3"]
LEGEND_RECT_WIDTH = 32
LEGEND_RECT_HEIGHT = 20
LEGEND_TEXT_FONT_SIZE = 14

MAX_AUTO_REFRESH_COUNT = 999999
MAX_AUTO_REFRESH_INTERVAL = 2000  # 2s


DB_LIST = [DB.Milvus, DB.ZillizCloud, DB.Weaviate]

CASE_LIST = [
    {
        "name": CaseType.LoadLDim,
        "intro": "Capacity benchmark for high dimension vector",
    },
    {
        "name": CaseType.LoadSDim,
        "intro": "Capacity benchmark for low dimension vector",
    },
    {
        "name": CaseType.PerformanceLZero,
        "intro": "Performance benchmark for large dataset",
    },
    {
        "name": CaseType.PerformanceMZero,
        "intro": "Performance benchmark for medium dataset",
    },
    {
        "name": CaseType.PerformanceSZero,
        "intro": "Performance benchmark for small dataset",
    },
    {
        "name": CaseType.PerformanceLLow,
        "intro": "Performance benchmark for large dataset with low filtering rate",
    },
    {
        "name": CaseType.PerformanceMLow,
        "intro": "Performance benchmark for medium dataset with low filtering rate",
    },
    {
        "name": CaseType.PerformanceSLow,
        "intro": "Performance benchmark for low dataset with low filtering rate",
    },
    {
        "name": CaseType.PerformanceLHigh,
        "intro": "Performance benchmark for medium dataset high low filtering rate",
    },
    {
        "name": CaseType.PerformanceMHigh,
        "intro": "Performance benchmark for low dataset with high filtering rate",
    },
    {
        "name": CaseType.PerformanceSHigh,
        "intro": "Performance benchmark for low dataset with high filtering rate",
    },
]


class InputType(IntEnum):
    Text = 20001
    Number = 20002
    Option = 20003


class CaseConfigInput(BaseModel):
    label: CaseConfigParamType
    inputType: InputType = InputType.Text
    inputConfig: dict = {}
    # todo type should be a function
    isDisplayed: typing.Any = lambda x: True


CaseConfigParamInput_IndexType = CaseConfigInput(
    label=CaseConfigParamType.IndexType,
    inputType=InputType.Option,
    inputConfig={
        "options": [
            IndexType.HNSW.value,
            IndexType.IVFFlat.value,
            IndexType.DISKANN.value,
            IndexType.Flat.value,
        ],
    },
)

CaseConfigParamInput_M = CaseConfigInput(
    label=CaseConfigParamType.M,
    inputType=InputType.Number,
    inputConfig={
        "min": 4,
        "max": 64,
    },
    isDisplayed=lambda config: config.get(CaseConfigParamType.IndexType, None)
    == IndexType.HNSW.value,
)

CaseConfigParamInput_EFConstruction = CaseConfigInput(
    label=CaseConfigParamType.EFConstruction,
    inputType=InputType.Number,
    inputConfig={
        "min": 8,
        "max": 512,
    },
    isDisplayed=lambda config: config[CaseConfigParamType.IndexType]
    == IndexType.HNSW.value
    if CaseConfigParamType.IndexType in config
    else True,
)

CaseConfigParamInput_EF = CaseConfigInput(
    label=CaseConfigParamType.EF,
    inputType=InputType.Number,
    inputConfig={
        "min": 100,
        "max": MAX_STREAMLIT_INT,
    },
    isDisplayed=lambda config: config[CaseConfigParamType.IndexType]
    == IndexType.HNSW.value
    if CaseConfigParamType.IndexType in config
    else True,
)

CaseConfigParamInput_MaxConnections = CaseConfigInput(
    label=CaseConfigParamType.MaxConnections,
    inputType=InputType.Number,
    inputConfig={
        "min": 1,
        "max": MAX_STREAMLIT_INT,
    },
)

CaseConfigParamInput_SearchList = CaseConfigInput(
    label=CaseConfigParamType.SearchList,
    inputType=InputType.Number,
    inputConfig={
        "min": 100,
        "max": MAX_STREAMLIT_INT,
    },
    isDisplayed=lambda config: config.get(CaseConfigParamType.IndexType, None)
    == IndexType.DISKANN.value,
)

CaseConfigParamInput_Nlist = CaseConfigInput(
    label=CaseConfigParamType.Nlist,
    inputType=InputType.Number,
    inputConfig={
        "min": 1,
        "max": 65536,
    },
    isDisplayed=lambda config: config.get(CaseConfigParamType.IndexType, None)
    == IndexType.IVFFlat.value,
)

CaseConfigParamInput_Nprobe = CaseConfigInput(
    label=CaseConfigParamType.Nprobe,
    inputType=InputType.Number,
    inputConfig={
        "min": 1,
        "max": 65536,
    },
    isDisplayed=lambda config: config.get(CaseConfigParamType.IndexType, None)
    == IndexType.IVFFlat.value,
)


MilvusLoadConfig = [
    CaseConfigParamInput_IndexType,
    CaseConfigParamInput_M,
    CaseConfigParamInput_EFConstruction,
    CaseConfigParamInput_Nlist,
]


MilvusPerformanceConfig = [
    CaseConfigParamInput_IndexType,
    CaseConfigParamInput_M,
    CaseConfigParamInput_EFConstruction,
    CaseConfigParamInput_EF,
    CaseConfigParamInput_SearchList,
    CaseConfigParamInput_Nlist,
    CaseConfigParamInput_Nprobe,
]

WeaviateLoadConfig = [
    CaseConfigParamInput_MaxConnections,
    CaseConfigParamInput_EFConstruction,
]

WeaviatePerformanceConfig = [
    CaseConfigParamInput_MaxConnections,
    CaseConfigParamInput_EFConstruction,
    CaseConfigParamInput_EF,
]

CASE_CONFIG_MAP = {
    DB.Milvus: {
        CaseType.LoadLDim: MilvusLoadConfig,
        CaseType.LoadSDim: MilvusLoadConfig,
        CaseType.PerformanceLZero: MilvusPerformanceConfig,
        CaseType.PerformanceMZero: MilvusPerformanceConfig,
        CaseType.PerformanceSZero: MilvusPerformanceConfig,
        CaseType.PerformanceLLow: MilvusPerformanceConfig,
        CaseType.PerformanceMLow: MilvusPerformanceConfig,
        CaseType.PerformanceSLow: MilvusPerformanceConfig,
        CaseType.PerformanceLHigh: MilvusPerformanceConfig,
        CaseType.PerformanceMHigh: MilvusPerformanceConfig,
        CaseType.PerformanceSHigh: MilvusPerformanceConfig,
    },
    DB.Weaviate: {
        CaseType.LoadLDim: WeaviateLoadConfig,
        CaseType.LoadSDim: WeaviateLoadConfig,
        CaseType.PerformanceLZero: WeaviatePerformanceConfig,
        CaseType.PerformanceMZero: WeaviatePerformanceConfig,
        CaseType.PerformanceSZero: WeaviatePerformanceConfig,
        CaseType.PerformanceLLow: WeaviatePerformanceConfig,
        CaseType.PerformanceMLow: WeaviatePerformanceConfig,
        CaseType.PerformanceSLow: WeaviatePerformanceConfig,
        CaseType.PerformanceLHigh: WeaviatePerformanceConfig,
        CaseType.PerformanceMHigh: WeaviatePerformanceConfig,
        CaseType.PerformanceSHigh: WeaviatePerformanceConfig,
    },
}

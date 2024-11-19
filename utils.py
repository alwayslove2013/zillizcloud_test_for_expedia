import time
from pymilvus import (
    Collection,
    utility,
    connections,
    CollectionSchema,
    DataType,
    FieldSchema,
)
from loguru import logger
from config import get_config


config = get_config()
uri = config["uri"]
token = config["token"]
collection_name = config["collection_name"]
vector_field = config["vector_field"]
dim = int(config["dim"])
topk = int(config["topk"])
pk_field = "pk"
varchar_field = "varchar"
vector_index_name = "vector_idx"


def connect() -> None:
    connections.connect(uri=uri, token=token, timeout=30)


def drop_collection_if_existed():
    if utility.has_collection(collection_name):
        logger.info(f"drop_old collection: {collection_name}")
        utility.drop_collection(collection_name)


def create_collection():
    logger.info(f"create_collection - {collection_name}")
    fields = [
        FieldSchema(pk_field, DataType.INT64, is_primary=True),
        FieldSchema(varchar_field, DataType.VARCHAR, max_length=65000),
        FieldSchema(
            vector_field,
            DataType.FLOAT_VECTOR,
            dim=dim,
        ),
    ]

    Collection(
        name=collection_name,
        schema=CollectionSchema(fields),
        # consistency_level="Session",
    )


def release_collection():
    logger.info("release collection")
    col = Collection(collection_name)
    col.release()


def get_collection():
    return Collection(collection_name)


def drop_index():
    logger.info("drop index")
    col = Collection(collection_name)
    col.drop_index(index_name=vector_index_name)


def create_index(metric_type: str = "COSINE"):
    logger.info("create index")
    col = Collection(collection_name)
    index_params = {"metric_type": metric_type}
    col.create_index(vector_field, index_params, index_name=vector_index_name)


def flush():
    logger.info("flush")
    col = Collection(collection_name)
    col.flush()


def optimize():
    logger.info("optimize index. it may take several minutes, please wait...")
    col = Collection(collection_name)
    utility.wait_for_index_building_complete(collection_name)

    def wait_index_building():
        while True:
            progress = utility.index_building_progress(collection_name)
            if progress.get("pending_index_rows", -1) == 0:
                break
            time.sleep(5)

    wait_index_building()
    col.compact()
    col.wait_for_compaction_completed()
    wait_index_building()
    col.compact()
    col.wait_for_compaction_completed()
    wait_index_building()
    logger.info("optimize [done]")


def load_index():
    logger.info("load index")
    col = Collection(collection_name)
    col.load()


def insert_data(
    train_vectors: list,  # list of vectors
    vachars: list,  # list of string
    global_idx: int,
):
    col = Collection(collection_name)
    ll = len(train_vectors)
    pk_column = list(range(global_idx, global_idx + ll))
    varchar_column = vachars
    vector_column = train_vectors
    col.insert([pk_column, varchar_column, vector_column])


def search_with_pymilvus(
    query_vector: list,  # vector (list of float)
    col: Collection,
):
    res = col.search(
        data=[query_vector],
        anns_field=vector_field,
        param={},
        limit=topk,
    )

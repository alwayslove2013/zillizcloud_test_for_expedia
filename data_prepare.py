import json
from loguru import logger
from config import get_config
from utils import (
    connect,
    create_collection,
    create_index,
    drop_collection_if_existed,
    flush,
    insert_data,
    load_index,
    optimize,
)

config = get_config()
dim = int(config["dim"])


def insert():
    file = "./data/glove.840B.300d.txt"
    num_insert_batch = 1_000
    cnt = 0
    global_cnt = 0
    vectors = []
    texts = []

    with open(file, "r") as f:
        while True:
            linetext = f.readline()
            if not linetext:
                insert_data(vectors, texts, global_idx=global_cnt)
                global_cnt += cnt
                logger.info(f"insert - {global_cnt} rows [done]")
                break

            line = linetext.rstrip().split(" ")
            text = line[0]
            texts.append(text)
            vector = [float(d) for d in line[1:]]
            assert (
                len(vector) == dim
            ), f"dim={len(vector)} not match, should be {dim}, idx: {global_cnt + cnt}, linetext: {linetext}"
            vectors.append(vector)
            cnt += 1

            if cnt == num_insert_batch:
                insert_data(vectors, texts, global_idx=global_cnt)
                global_cnt += cnt
                vectors = []
                texts = []
                cnt = 0
                if global_cnt % 100_000 == 0:
                    logger.info(f"insert - {global_cnt} rows")


def main():
    connect()
    drop_collection_if_existed()
    create_collection()
    create_index()
    load_index()
    insert()
    flush()
    create_index()
    optimize()


if __name__ == "__main__":
    main()

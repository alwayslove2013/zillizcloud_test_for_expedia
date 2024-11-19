import json
import concurrent
import multiprocessing as mp
import time
import traceback
from loguru import logger
import numpy as np
from config import get_config
from utils import connect, get_collection, search_with_pymilvus


config = get_config()
dim = int(config["dim"])
conc_list = [int(c) for c in config["conc_list"].split(",")]
conc_duration = int(config["conc_duration"])
conc_intermission = int(config["conc_intermission"])
num_random_queries = 1000


def search_test(
    queries: list,  # list of query_vectos
    duration: int,
    q: mp.Queue,
    cond: mp.Condition,
):
    query_len = len(queries)
    count = 0
    idx = 0
    latencies = []

    connect()
    col = get_collection()

    # sync all process
    q.put(1)
    with cond:
        cond.wait()

    start_time = time.perf_counter()
    while time.perf_counter() < start_time + duration:
        s = time.perf_counter()
        search_with_pymilvus(queries[idx], col)
        latencies.append(time.perf_counter() - s)
        count += 1
        if idx >= query_len - 1:
            idx = 0
        else:
            idx += 1
    return count, latencies


def conc_test(conc: int):
    queries = np.random.rand(num_random_queries, dim).tolist()
    logger.info(f"conc_test [start] - conc: {conc}")
    try:
        with mp.Manager() as m:
            q, cond = m.Queue(), m.Condition()
            with concurrent.futures.ProcessPoolExecutor(
                mp_context=mp.get_context("spawn"), max_workers=conc
            ) as executor:
                future_iter = [
                    executor.submit(search_test, queries, conc_duration, q, cond)
                    for _ in range(conc)
                ]

                # sync all processes
                while q.qsize() < conc:
                    time.sleep(5)

                with cond:
                    cond.notify_all()
                    logger.info("  all processes are ready.")

                results = [r.result() for r in future_iter]
                all_count = sum([r[0] for r in results])
                qps = all_count / conc_duration
                all_latencies = sum([r[1] for r in results], [])
                latency_p99 = np.percentile(all_latencies, 99)
                latency_avg = np.mean(all_latencies)
                logger.info(
                    f"conc_test [done]. conc: {conc}, all_count: {all_count}, qps: {qps:.2f}, latency_avg: {latency_avg * 1000:.2f}ms, latency_p99: {latency_p99  * 1000:.2f}ms"
                )
                return qps, latency_p99, latency_avg
    except Exception as e:
        logger.warning(f"Fail to search all concurrencies: {conc}, reason={e}")
        traceback.print_exc()


def main():
    results = []
    for i, conc in enumerate(conc_list):
        qps, latency_p99, latency_avg = conc_test(conc)
        results.append(
            dict(conc=conc, qps=qps, latency_p99=latency_p99, latency_avg=latency_avg)
        )
        if i < len(conc_list) - 1:
            time.sleep(conc_intermission)
    with open("./pymilvus_conc_test_results.json", "w") as f:
        json.dump(results, f)

    # display
    print(
        f"{'conc':<10} | {'qps':<10} | {'latency_avg (ms)':<20} | {'latency_p99 (ms)':<20}"
    )
    print("─" * 75)
    for item in results:
        print(
            f"{item['conc']:<10} | {item['qps']:<10.2f} | {item['latency_avg'] * 1000:<20.2f} | {item['latency_p99'] * 1000:<20.2f}"
        )
    print("─" * 75)


if __name__ == "__main__":
    main()

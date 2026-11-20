"""Single-host, GPU-0 StarRocks query smoke check against a DuckDB oracle.

Run from the Sirius root with `pixi run python experimental/starrocks/integration/local_smoke.py
--data /path/to/tpch_parquet_sf10 --output /path/to/fresh/log-directory` after building FE/CN.
This checks query results and fragment plans; it is not a performance benchmark.
"""

import argparse
import decimal
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

import duckdb

QUERIES = {
    "ordered_limit": "SELECT n_nationkey, n_name FROM nation ORDER BY n_nationkey LIMIT 3 OFFSET 2",
    "group_sum": "SELECT n_regionkey, SUM(CAST(n_nationkey AS BIGINT)) FROM nation GROUP BY n_regionkey ORDER BY n_regionkey",
    "join": "SELECT r_name, COUNT(*) FROM region JOIN nation ON r_regionkey = n_regionkey GROUP BY r_name ORDER BY r_name",
    "orders_aggregate": "SELECT o_orderstatus, COUNT(*), SUM(CAST(o_orderkey AS BIGINT)) FROM orders GROUP BY o_orderstatus ORDER BY o_orderstatus",
    "revenue": "SELECT SUM(l_extendedprice * l_discount) FROM lineitem WHERE l_shipdate >= DATE '1994-01-01' AND l_shipdate < DATE '1995-01-01' AND l_discount BETWEEN 0.05 AND 0.07 AND l_quantity < 24",
    "decimal_average": "SELECT AVG(l_quantity) FROM lineitem WHERE l_shipdate >= DATE '1994-01-01' AND l_shipdate < DATE '1995-01-01'",
    "join_aggregate": "SELECT l_shipmode, SUM(CASE WHEN o_orderpriority = '1-URGENT' OR o_orderpriority = '2-HIGH' THEN 1 ELSE 0 END) FROM orders JOIN lineitem ON o_orderkey = l_orderkey WHERE l_shipmode IN ('MAIL', 'SHIP') AND l_commitdate < l_receiptdate AND l_shipdate < l_commitdate AND l_receiptdate >= DATE '1994-01-01' AND l_receiptdate < DATE '1995-01-01' GROUP BY l_shipmode ORDER BY l_shipmode",
}
SETTINGS = "SET single_node_exec_plan=false; SET pipeline_dop=1; SET parallel_fragment_exec_instance_num=1; SET new_planner_agg_stage=2; SET query_timeout=120;"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = args.data.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    workspace = Path(__file__).resolve().parents[1]
    for port in (9030, 8060, 9050, 9060):
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                raise RuntimeError(f"Port {port} is occupied; stop the existing local cluster first")
    config = output / "sirius.yaml"
    config.write_text("""sirius:
  topology:
    num_gpus: 1
  memory:
    gpu:
      usage_limit_fraction: 0.7
    host:
      capacity_bytes: 8GiB
      initial_number_pools: 2
      pool_size: 512
      block_size: 1048576
""")
    fragments = output / "fragments"
    fragments.mkdir()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", SIRIUS_CN_DUMP_FRAGMENTS=str(fragments), NO_COLOR="1")
    env.pop("SIRIUS_CN_TRANSLATE_ONLY", None)
    manifest = workspace / "pixi.toml"
    client = ["pixi", "run", "--manifest-path", str(manifest), "-e", "client", "mysql", "--host=127.0.0.1", "--port=9030", "--user=root", "--connect-timeout=2", "--batch", "--raw", "--skip-column-names"]

    def mysql(sql, timeout=150, headers=False):
        argv = client[:-1] if headers else client
        result = subprocess.run(argv, input=sql, text=True, capture_output=True, env=env, cwd=workspace, timeout=timeout)
        if result.returncode:
            raise RuntimeError(result.stderr)
        return result.stdout

    processes = []
    handles = []
    results = []
    catalog_created = False
    volume_created = False
    database = f"sirius_port_smoke_{int(time.time())}"
    volume = database + "_volume"
    try:
        for name, command in (
            ("fe", ["pixi", "run", "--manifest-path", str(manifest), "fe-run"]),
            ("cn", ["pixi", "run", "--manifest-path", str(manifest), "bash", "-c", 'source scripts/cn-env.sh && exec target/release/sirius-starrocks-cn --sirius-config "$1"', "cn", str(config)]),
        ):
            handle = (output / f"{name}.log").open("w")
            handles.append(handle)
            processes.append(subprocess.Popen(command, cwd=workspace, env=env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True))
        deadline = time.monotonic() + 120
        while True:
            if any(p.poll() is not None for p in processes):
                raise RuntimeError("A cluster process exited; inspect fe.log and cn.log")
            try:
                nodes = mysql("SHOW COMPUTE NODES;", timeout=8, headers=True)
                lines = [line.split("\t") for line in nodes.splitlines()]
                alive_column = [name.lower() for name in lines[0]].index("alive") if lines else None
                alive = [row for row in lines[1:] if row[alive_column].lower() == "true"]
                if len(alive) > 1:
                    raise AssertionError("The local FE has multiple live compute nodes; refusing a single-CN check")
                if len(alive) == 1:
                    (output / "compute-nodes.tsv").write_text(nodes)
                    break
            except (RuntimeError, subprocess.TimeoutExpired):
                pass
            if time.monotonic() >= deadline:
                raise TimeoutError("FE/CN registration did not complete")
            time.sleep(1)
        print("Local FE and one GPU-0 compute node are ready", flush=True)
        oracle = duckdb.connect()
        oracle.execute("SET threads=2; SET memory_limit='4GB'")
        # Shared-data FE requires a volume even for a database containing only views.
        # This run creates metadata only; FILES() reads the local Parquet files directly.
        mysql('ADMIN SET FRONTEND CONFIG ("enable_storage_volume_access_check" = "false");')
        mysql(f"CREATE STORAGE VOLUME {volume} TYPE = HDFS LOCATIONS = ('hdfs://127.0.0.1:1/');")
        volume_created = True
        mysql(f'CREATE DATABASE {database} PROPERTIES ("storage_volume" = "{volume}");')
        catalog_created = True
        ddl = [f"USE {database};"]
        for table in ("nation", "region", "orders", "lineitem"):
            pattern = str(data / table / "*.parquet")
            escaped = pattern.replace("'", "''")
            oracle.execute(f"CREATE VIEW {table} AS SELECT * FROM read_parquet('{escaped}')")
            uri = (data / table).as_uri() + "/*.parquet"
            ddl.append(f"CREATE VIEW {table} AS SELECT * FROM FILES(\"path\"=\"{uri}\", \"format\"=\"parquet\");")
        mysql("\n".join(ddl))
        for name, query in QUERIES.items():
            prefix = f"USE {database}; {SETTINGS} "
            plan = mysql(prefix + "EXPLAIN " + query)
            (output / f"{name}.plan.txt").write_text(plan)
            (output / f"{name}.sql").write_text(query + ";\n")
            actual_text = mysql(prefix + query)
            (output / f"{name}.actual.tsv").write_text(actual_text)
            actual = [line.split("\t") for line in actual_text.splitlines()]
            expected = oracle.execute(query).fetchall()
            (output / f"{name}.expected.json").write_text(json.dumps(expected, default=str))
            if len(actual) != len(expected):
                raise AssertionError(f"{name}: row count {len(actual)} != {len(expected)}")
            for actual_row, expected_row in zip(actual, expected):
                if len(actual_row) != len(expected_row):
                    raise AssertionError(f"{name}: column count mismatch")
                for value, reference in zip(actual_row, expected_row):
                    if reference is None:
                        same = value == "NULL"
                    elif isinstance(reference, (float, decimal.Decimal)):
                        same = math.isclose(float(value), float(reference), rel_tol=1e-7, abs_tol=1e-6)
                    else:
                        same = value == str(reference)
                    if not same:
                        raise AssertionError(f"{name}: {value!r} != {reference!r}")
            count = plan.count("PLAN FRAGMENT")
            if name == "orders_aggregate" and count < 2:
                raise AssertionError("The aggregate did not exercise multiple plan fragments")
            result = dict(query=name, rows=len(actual), fragments=count, status="PASS")
            results.append(result)
            print(json.dumps(result), flush=True)
        (output / "results.json").write_text(json.dumps(results, indent=2))
    finally:
        try:
            if catalog_created:
                mysql(f"DROP DATABASE {database} FORCE;", timeout=10)
            if volume_created:
                mysql(f"DROP STORAGE VOLUME {volume};", timeout=10)
        except (RuntimeError, subprocess.TimeoutExpired):
            pass
        for process in reversed(processes):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
        for handle in handles:
            handle.close()


if __name__ == "__main__":
    main()

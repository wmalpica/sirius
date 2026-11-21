#!/usr/bin/env python3
"""Run the shared TPC-H SQL on local Sirius StarRocks compute nodes.

From the repository root:
pixi run --manifest-path experimental/starrocks/pixi.toml -e bench python \
  test/tpch_performance/run_tpch_starrocks.py --config /path/sirius.yaml \
  --parquet-dir /data/tpch_parquet_sf300 --output-dir /path/fresh --iterations 2 300

Uses the regular runner's q<N>/{query.sql,result.txt,timings.csv} convention.
Startup, setup, EXPLAIN, and result serialization are outside query timing. Each
query's repetitions are consecutive, in one client session and one fresh cluster.
"""

import argparse
import csv
import datetime
import json
from pathlib import Path
import re
import signal
import sys
import time
import traceback

import pymysql

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experimental/starrocks/integration"))
from local_cluster import LocalCluster  # noqa: E402

TABLES = (
    "customer",
    "lineitem",
    "nation",
    "orders",
    "part",
    "partsupp",
    "region",
    "supplier",
)


def starrocks_sql(sql):
    # FE requires comma-form SUBSTRING; q22 uses equivalent SQL FROM/FOR syntax.
    # Retain both the shared query and the executed spelling in each query folder.
    return re.sub(
        r"\bsubstring\((\w+)\s+from\s+(\d+)\s+for\s+(\d+)\)",
        r"substring(\1, \2, \3)",
        sql,
        flags=re.IGNORECASE,
    )


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def connect(timeout):
    return pymysql.connect(
        host="127.0.0.1",
        port=9030,
        user="root",
        connect_timeout=3,
        read_timeout=timeout,
        write_timeout=timeout,
        autocommit=True,
    )


def execute(connection, sql):
    with connection.cursor() as cursor:
        cursor.execute(sql)
        rows = cursor.fetchall()
        columns = (
            [column[0] for column in cursor.description] if cursor.description else []
        )
        return columns, rows


def query_settings(connection, number):
    # Q16's default two-stage multi_distinct_count has an opaque partial state
    # the prototype cannot exchange. FE's three-stage plan uses explicit dedupe
    # keys instead. This changes only FE planning, not the SQL or engine config.
    setting = f"SET new_planner_agg_stage={3 if number == 16 else 0}"
    execute(connection, setting)
    return setting


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--parquet-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--devices",
        default="0",
        help="comma-separated CUDA ordinals or MIG UUIDs; one CN per device",
    )
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--query-timeout", type=int, default=1200)
    parser.add_argument("--run-timeout", type=int, default=7200)
    parser.add_argument("--explain-only", action="store_true")
    parser.add_argument(
        "--continue-on-query-error",
        action="store_true",
        help="record server query errors and continue; connection loss, timeout, or service exit still aborts",
    )
    parser.add_argument("scale_factor", type=int)
    parser.add_argument("queries", type=int, nargs="*", default=list(range(1, 23)))
    args = parser.parse_args()
    if args.iterations < 1 or args.query_timeout < 1 or args.run_timeout < 1:
        parser.error("iterations and timeouts must be positive")
    if any(query not in range(1, 23) for query in args.queries):
        parser.error("query numbers must be between 1 and 22")
    devices = [device.strip() for device in args.devices.split(",") if device.strip()]
    if not devices or len(set(devices)) != len(devices):
        parser.error("--devices must select distinct CUDA/MIG devices")
    data = args.parquet_dir.resolve(strict=True)
    config = args.config.resolve(strict=True)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Config/provenance may already be staged here, but an attempt is never reused.
    with (output / "attempt-start.json").open("x") as handle:
        json.dump(
            dict(
                argv=sys.argv,
                started=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            ),
            handle,
            indent=2,
        )

    def expired(*_):
        raise TimeoutError("Benchmark run deadline exceeded")

    signal.signal(signal.SIGALRM, expired)
    signal.signal(signal.SIGTERM, expired)
    signal.alarm(args.run_timeout)
    status = dict(status="running", completed=0, failed=0)
    try:
        with LocalCluster(output, config, devices) as cluster:
            deadline = time.monotonic() + 180
            while True:
                cluster.check()
                try:
                    with connect(5) as probe:
                        columns, rows = execute(probe, "SHOW COMPUTE NODES")
                    alive = columns.index("Alive")
                    live = [
                        row for row in rows if str(row[alive]).lower() in ("true", "1")
                    ]
                    if len(live) > len(devices):
                        raise RuntimeError(
                            "More compute nodes are alive than requested"
                        )
                    if len(live) == len(devices):
                        dump(
                            output / "compute-nodes.json",
                            dict(columns=columns, rows=rows),
                        )
                        break
                except pymysql.MySQLError:
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError("Cluster registration timed out")
                time.sleep(1)
            connection = connect(args.query_timeout + 30)
            try:
                database = "tpch_bench"
                ddl = [
                    'ADMIN SET FRONTEND CONFIG ("enable_storage_volume_access_check" = "false")',
                    "CREATE STORAGE VOLUME bench_volume TYPE = HDFS LOCATIONS = ('hdfs://127.0.0.1:1/')",
                    f'CREATE DATABASE {database} PROPERTIES ("storage_volume" = "bench_volume")',
                    f"USE {database}",
                ]
                for table in TABLES:
                    if not list((data / table).glob("*.parquet")):
                        raise ValueError(f"No Parquet files for {table}")
                    uri = (data / table).as_uri() + "/*.parquet"
                    ddl.append(
                        f'CREATE VIEW {table} AS SELECT * FROM FILES("path"="{uri}", "format"="parquet")'
                    )
                settings = [
                    "SET single_node_exec_plan=false",
                    "SET pipeline_dop=1",
                    "SET parallel_fragment_exec_instance_num=1",
                    f"SET query_timeout={args.query_timeout}",
                ]
                (output / "setup.sql").write_text(";\n".join(ddl + settings) + ";\n")
                for sql in ddl + settings:
                    execute(connection, sql)
                columns, rows = execute(connection, "SHOW VARIABLES")
                dump(
                    output / "session-variables.json", dict(columns=columns, rows=rows)
                )
                # Prepare every plan before timing any query; this does not execute the data scans.
                for number in args.queries:
                    folder = output / f"q{number}"
                    folder.mkdir()
                    sql = (
                        ROOT / f"test/tpch_performance/tpch_queries/orig/q{number}.sql"
                    ).read_text()
                    (folder / "query.sql").write_text(sql)
                    sql = starrocks_sql(sql)
                    (folder / "executed.sql").write_text(sql)
                    (folder / "session.sql").write_text(
                        query_settings(connection, number) + ";\n"
                    )
                    _, plan = execute(connection, "EXPLAIN " + sql)
                    (folder / "plan.txt").write_text(
                        "\n".join(str(row[0]) for row in plan) + "\n"
                    )
                print(
                    f"All requested plans prepared; {len(devices)} local compute node(s) ready",
                    flush=True,
                )
                if not args.explain_only:
                    with (output / "fragment-index.jsonl").open(
                        "w", buffering=1
                    ) as index:
                        for number in args.queries:
                            folder = output / f"q{number}"
                            sql = (folder / "executed.sql").read_text()
                            query_settings(connection, number)
                            with (folder / "timings.csv").open(
                                "w", buffering=1
                            ) as timing:
                                writer = csv.writer(timing)
                                writer.writerow(["step", "runtime_s"])
                                for iteration in range(1, args.iterations + 1):
                                    cluster.check()
                                    before = cluster.fragment_files()
                                    entry = dict(
                                        query=number,
                                        iteration=iteration,
                                        started_ns=time.time_ns(),
                                    )
                                    index.write(
                                        json.dumps(dict(entry, event="start")) + "\n"
                                    )
                                    started = time.perf_counter()
                                    try:
                                        columns, rows = execute(connection, sql)
                                    except pymysql.ProgrammingError as error:
                                        entry.update(
                                            finished_ns=time.time_ns(),
                                            failed_elapsed_s=time.perf_counter()
                                            - started,
                                            error=str(error),
                                        )
                                        entry["files"] = sorted(
                                            cluster.fragment_files() - before
                                        )
                                        index.write(
                                            json.dumps(dict(entry, event="error"))
                                            + "\n"
                                        )
                                        dump(
                                            folder / f"error_iter{iteration}.json",
                                            entry,
                                        )
                                        writer.writerow([f"iter_{iteration}", "N/A"])
                                        status["failed"] += 1
                                        print(
                                            f"Q{number} iter {iteration}: ERROR {error}",
                                            flush=True,
                                        )
                                        if not args.continue_on_query_error or any(
                                            word in str(error).lower()
                                            for word in (
                                                "timeout",
                                                "timed out",
                                                "deadline",
                                            )
                                        ):
                                            raise
                                        cluster.check()
                                        # Connection/FE liveness is required before proceeding.
                                        execute(connection, "SELECT 1")
                                        continue
                                    elapsed = time.perf_counter() - started
                                    entry.update(
                                        finished_ns=time.time_ns(), runtime_s=elapsed
                                    )
                                    writer.writerow(
                                        [f"iter_{iteration}", f"{elapsed:.9f}"]
                                    )
                                    dump(
                                        folder / f"result_iter{iteration}.json",
                                        dict(columns=columns, rows=rows),
                                    )
                                    (folder / "result.txt").write_text(
                                        "\n".join(
                                            "\t".join(
                                                "NULL" if value is None else str(value)
                                                for value in row
                                            )
                                            for row in rows
                                        )
                                        + "\n"
                                    )
                                    entry["files"] = sorted(
                                        cluster.fragment_files() - before
                                    )
                                    index.write(
                                        json.dumps(dict(entry, event="finish")) + "\n"
                                    )
                                    status["completed"] += 1
                                    print(
                                        f"Q{number} iter {iteration}: {elapsed:.6f}s ({len(rows)} rows)",
                                        flush=True,
                                    )
            finally:
                connection.close()
        status["status"] = "completed_with_errors" if status["failed"] else "completed"
        return 1 if status["failed"] else 0
    except BaseException as error:
        status.update(
            status="failed", error=str(error), traceback=traceback.format_exc()
        )
        raise
    finally:
        signal.alarm(0)
        dump(output / "status.json", status)


if __name__ == "__main__":
    raise SystemExit(main())

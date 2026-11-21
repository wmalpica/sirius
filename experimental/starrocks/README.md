# Sirius as a StarRocks compute node

A Rust process (`sirius-starrocks-cn`) that joins a StarRocks FE as a compute node and runs the
plan fragments it receives on the embedded Sirius engine. The `starrocks/` submodule pins the
StarRocks release the CN is built against; its thrift and protobuf definitions are compiled into
the CN.

## StarRocks versions

The CN works with one StarRocks release: the one the submodule pins (4.1.3 today). The heartbeat
reports it, so `SHOW COMPUTE NODES` shows e.g.
`sirius-starrocks-cn/0.1.0 (starrocks 4.1.3, 8a8e186)`.

- Build the CN for each StarRocks release.
- Run it with an FE of the same release.
- Upgrade the FE and the CN together.

Nothing in the protocol checks versions, and the thrift plan structures change between minor
releases. StarRocks upgrades backends before the FE and expects them to accept the previous
release's plans; the CN doesn't, so it may reject or misread plans from an FE of another release.

## Main-based integration port

The integration port keeps main's public C ABI crates and existing C++ embedding path.
The advanced compute node uses the opt-in `sirius-ffi/starrocks-experimental` Rust feature and
`sirius::starrocks_ffi` C++ namespace. Its fragment session and streaming sink have separate
implementations; main's SQL planner, distributed exchange implementation, scheduler, and I/O
backends remain the foundation. These experimental bindings are not a stable public ABI.

From this directory:

```bash
pixi run fe-build
pixi run cn-build
CUDA_VISIBLE_DEVICES=0 pixi run cluster
# In another terminal:
pixi run client
```

`cn-build` builds the main engine, then a private shared NIXL SDK from the same source revision
main pins, then the full Rust compute node. Main continues to use its existing static SDK.
The private SDK lives under `build/starrocks/nixl-install`; `scripts/cn-env.sh` selects it when
present. Operator-supplied `NIXL_PREFIX`, `UCX_PREFIX`, and plugin settings still take precedence.

For a real engine build that only exchanges fragments on the same compute node, without NIXL:

```bash
pixi run cn-build-local
```

The FE compiler is built from a checksum-verified Apache source release matching the Java
runtime version declared in `starrocks/fe/pom.xml`, and cached under `.pixi`. Its generator is
separate from the Rust generator pinned in Pixi. A StarRocks upgrade to an unknown Thrift
version requires adding a verified release checksum in `scripts/build_fe_thrift.py`.

### Local correctness smoke check

After building the full CN and FE, run from the Sirius repository root:

```bash
pixi run python experimental/starrocks/integration/local_smoke.py \
  --data /home/wmalpica/datasets/tpch_parquet_sf10 \
  --output /home/wmalpica/repos/logs/starrocks-local-smoke-UNIQUE
```

The output directory must not exist. The check starts its own local FE and one CN on GPU 0,
checks ordered limits, grouped aggregation, joins, revenue, and decimal averages against DuckDB,
and requires multiple fragments for the orders aggregation. It preserves plans, results, received
fragments, config, and service logs, then stops the processes it started. Existing services on the
required ports cause it to refuse startup. It does not test multiple GPUs or hosts and does not
measure comparative performance. The older integration scripts below this directory describe
the source prototype's historical multi-node runs, not validation of this port.

### Local TPC-H benchmark

The self-contained benchmark adapter uses the same SQL files and per-query timing CSV format as
`test/tpch_performance/run_tpch_parquet.sh`. Build FE/CN first, then run from the repository root:

```bash
pixi run --manifest-path experimental/starrocks/pixi.toml -e bench python \
  test/tpch_performance/run_tpch_starrocks.py \
  --config /path/to/sirius.yaml \
  --parquet-dir /home/wmalpica/datasets/tpch_parquet_sf300 \
  --output-dir /path/to/fresh-run --iterations 2 300
```

For two compute nodes, add `--devices 0,1` (or two comma-separated MIG UUIDs supported by the
installed driver and topology discovery). Each CN sees exactly one selected device as CUDA
device 0. Listener ports, engine logs, fragment dumps, and Quent directories are separate per
CN; `cluster.json` records the device selectors, configurations, and ports. Host capacity is
**not divided automatically**: for an aggregate 80-GiB engine-pool budget, supply a configuration
with `capacity_bytes: 40GiB` when launching two CNs. This limits each engine's host pool, not the
entire process's resident memory. Regular Sirius can use one shared 80-GiB pool for both devices.

MIG partitions share a physical GPU's compute and memory capacity. Treat two MIG instances as a
device-partitioning and frontend comparison rather than a doubling of physical resources. First
verify that CUDA and Sirius enumerate both devices, and smoke-test a cross-CN exchange. CUDA peer
access within a process does not establish that cross-process CUDA IPC works between MIG GPU
instances. Preserve the actual UCX transport diagnostics and the UUID-to-selector mapping.

Append query numbers to select a subset. Each query runs twice consecutively in one persistent
client session and one private FE/CN cluster on GPU 0. Preparation and startup are untimed;
latency covers submission through complete result retrieval. `--query-timeout` defaults to 1200
seconds and `--run-timeout` to 7200 seconds. `--explain-only` checks all requested FE plans without
executing them. A failure ends the attempt and preserves `status.json`; use a new output directory
for a retry. The adapter never changes an existing cluster or reuses an attempt.
Use `--continue-on-query-error` for coverage measurements: it saves an error record and `N/A`
timing for a failed server query, then proceeds if the FE connection and services are alive.
Timeouts, connection loss, and service exit still stop the attempt. Any query error makes the
final status `completed_with_errors` and the command exits nonzero; failed-query elapsed time
is saved separately and never counted as a successful latency.

The output contains private FE state/logs, CN and engine logs, SQL, plans, both iterations' results,
and `q<N>/timings.csv` with columns `step,runtime_s`. `fragment-index.jsonl` links each query and
iteration to wall-clock timestamps and newly dumped fragment files; Quent fragment records retain
their original labels. Enable Quent in the supplied YAML and point its output directory into the
run directory. The shared q22 uses `SUBSTRING(x FROM n FOR m)`; StarRocks requires the equivalent
comma spelling. Both `query.sql` and `executed.sql` are saved.
Q16 uses FE aggregation stage 3 to exchange explicit deduplication keys; the default FE plan's
opaque partial DISTINCT state is unsupported by this prototype. Other queries use AUTO, and
`session.sql` records the per-query setting. Pipeline degree and fragment-instance count are 1,
with `single_node_exec_plan=false` to exercise fragment boundaries on the single CN.

For the regular runner, `TPCH_QUERY_LABEL_PREFIX=baseline` adds untimed per-iteration Quent labels.
Single-session runs now retain `q<N>/result_iter<I>.txt` as well as the last `result.txt`.
The existing result comparator supports ordered cross-frontend values with
`--values-only --actual-format starrocks-json`: integer and decimal values remain exact; only
floating columns use `--float-tolerance` and `--relative-tolerance`. Run independent DuckDB
validation after timed measurements so it does not warm the dataset first.
Boxed CLI output cannot distinguish text whitespace from display padding. For an exact text
check against the JSON results, also capture the CPU query with DuckDB `-json` and pass that file
as `--expected-rows-json`; the boxed reference still supplies column types.

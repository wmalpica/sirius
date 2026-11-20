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

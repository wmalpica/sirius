# Completing concurrent query support

Investigation date: 2026-09-29. Source: `concurrency3` at
`98280218ba061329b4fa511adcf7684f8b2f91af`, including the current cuCascade submodule.

Status: implementation proposal for discussion. No runtime source changes, builds, or tests
were performed for this investigation. Findings are from source inspection by three agents
(admission/lifecycle, scheduling/errors/lifetime, scans/cache/multiple GPUs), followed by
cross-checking and synthesis. Concrete source interleavings are distinguished below from
risks that need a deterministic reproduction. Historical documents remain unchanged.

## 1. Conclusion and intended behavior

The existing per-query infrastructure is a useful foundation, but concurrency is not enabled
by changing the existing YAML value. `max_concurrent_queries` currently sizes scan workers;
the shared lifecycle mutex still permits only one execution window. Removing that mutex
without other changes would expose cross-query error propagation and unsafe lifetime gaps.

The target is multiple DuckDB connections executing against one shared Sirius runtime:

- A configured maximum N bounds admitted Sirius queries, including their initialization and
  retirement. Requests beyond N wait in a cancellable FIFO admission queue.
- Admitted queries coexist. Tasks favor the oldest runnable query on each compatible GPU;
  already-running tasks are not preempted. Equal GPU time is not required.
- An ordinary query failure or cancellation stops and retires only that query. Other queries
  continue and later requests can still enter.
- Plans, repositories, pinned data, scan providers, buffers, and CUDA events remain valid
  through the last asynchronous use.
- Queries can share configured GPUs and pinned/cache data safely. Prefetch and spilling use
  bounded shared resources and preserve progress under memory pressure.
- CPU-only paths retain their existing bypass behavior. Concurrent use of one DuckDB
  connection is not the SQL concurrency model.

Recommend startup configuration `sirius.max_concurrent_queries`, default 1 initially. This is
a proposed key, not an existing setting. Derive scan capacity from that same authoritative
value. Explicitly migrate the existing `sirius.executor.scan_manager.max_concurrent_queries`
setting or accept it as a deprecated alias; reject contradictory values.

The limit applies per shared Sirius runtime / DatabaseInstance. Multiple DatabaseInstances
in a process need a separate GPU allocator/resource-ownership contract before they can be
advertised as independently concurrent Sirius engines.

## 2. What is already present, and what the historical notes overstate

Retain these pieces:

- Per-query completion handlers, creator state, scan state, repository-manager registration,
  query-indexed task queues, and per-query bounded-pool counters.
- A lifecycle enqueue gate that **already rejects unknown query IDs**.
- Connection-local planning captures and connection state.
- Immutable operator-parameter snapshots in pipeline build contexts, although snapshots are
  currently taken separately at different stages.
- Pinned-entry shared ownership, query-local MVCC masks, task CUDA completion fences, and
  improved subsystem shutdown ordering.

Corrections to the older issue register:

| Historical topic | Current assessment |
|---|---|
| Unknown IDs accepted by the gate | Fixed; retain fail-closed behavior. |
| Query failure always stops the creator/scheduler | Several paths are fixed; GPU error drain and downgrade cleanup remain global. |
| Creator in-flight tracking completely fixed | Incomplete: queue removal precedes attribution, leaving an untracked interval. |
| No operator configuration snapshots | Snapshots exist, but need one consistent source per execution and synchronized settings. |
| Cache map iterator used after unlock | Fixed in the rebased branch. |
| New query epoch demotes another query's cache data | Old eviction scheme is gone; do not implement the obsolete live-epoch proposal. |
| One blocking scan sequencer per query | Current coalescers may have workers per scan slot; validate actual scheduling dependencies. |
| No concurrency tests | There are threaded lifecycle tests, mostly proving serialization, plus component tests. There is no sufficient end-to-end overlap/failure proof. |
| Prepared plan concurrently reset across SQL connections | Not established: supported SQL uses separate connections/plans, and DuckDB serializes one ClientContext. Treat shared-plan reentrancy as API hardening unless a supported overlapping caller is demonstrated. |
| Shared repositories plus `close()` must be the next patch | This is one possible design. Prefer a complete query-owner/borrow contract first; the historical partial migration crashed and was reverted. |

Some architecture documentation still describes a current query stored in SiriusContext or
connection-owned engines. Update those descriptions with the implementation; source is the
authority for this plan.

## 3. Remaining findings

### A. Admission and maintenance need different contracts

Evidence: `src/sirius_context.cpp:1680` takes the global mutex. The parsed limit at
`src/sirius_config.cpp:290` feeds scan-pool sizing at
`src/scan_manager/sirius_scan_manager.cpp:1314`; excess registrations only warn at `:1873`.

The existing SlotGuard also protects planning, operator-setting writes, pinned-table UPDATE
checks, pin/unpin, cache reset, and ANN index operations. An N-permit replacement would make
those operations overlap resource users that currently depend on exclusivity.

Introduce an admission monitor with bounded query permits, shared planning access, exclusive
maintenance access, and a closing state. Pending maintenance must stop new admissions so it
can eventually run. Never wait for exclusive maintenance while holding an execution permit
needed by that maintenance to drain. Internal pin queries must execute under their existing
maintenance ownership rather than recursively acquire a permit.

Initially keep pin/unpin, cache reset, and resource-changing index operations exclusive.
Pure planning can overlap execution once it holds stable configuration and pin snapshots.
ANN search needs explicit classification: conservatively retain its current exclusivity
until its resource borrowing is represented by a query owner.

### B. Lifetime tracking has gaps between queues and workers

Evidence: creator pops at `src/creator/task_creator.cpp:587`, captures query state at `:602`,
attaches the query to a pool slot at `:622`, then dereferences the operator at `:626`.

A teardown can occur between capture and attach: the queue is empty, the per-query worker
count is zero, and cleanup can destroy the plan before the operator is read. Keeping creator
state or a pipeline alive does not itself own the operators referenced by that pipeline.
There are analogous gaps in GPU dispatch, scheduler routing, lookahead, and spill extraction.

Also, `accepts_work()` followed by queue `push()` is a check followed by a separate action.
A producer may pass the gate, pause, then publish after quiescence and a queue drain. The gate
must coordinate outstanding publishers with retirement; one more lookup does not establish
the required lifetime guarantee.

Recommendation: a per-query execution owner and a continuously held work lease. A lease is
an RAII claim that the holder can use query-owned state. Acquire it atomically with permission
to publish work; transfer it through queues, manager-local variables, workers, retries,
lookahead, spilling, CUDA completion, and final callbacks. Queue removal must not create a
period with no owner/count. Idle worker-capacity slots remain separate from query lifetime.

Define how pre-close publishers settle: they either publish into a queue that is included in
retirement or cancel/release their work. Cleanup waits for publishers before declaring queue
drain complete. Existing holders may finish safely after closure but cannot create untracked
new work. Do not wait for the current worker's own lease inside its error callback.

### C. Error handling and cleanup still affect other queries

- `src/parallel/task_executor.cpp:231` interrupts and restarts the shared GPU manager on one
  query's error. Another query's in-transit enqueue can be refused. Two concurrent failing
  queries can also race manipulation of the same management thread.
- `src/sirius_context.cpp:458` drains every downgrade executor on every query end. Drain
  cancels shared requests, so finishing A can fail B's pending reservation/spill request.
- Beginning a query opens its gate, allocates a repository manager, and registers creator
  state (`src/sirius_context.cpp:414`). Any exception currently marks the shared runtime
  unavailable (`:661`). Ordinary allocation/setup failures need query-local rollback.
- Creator hint traversal runs before its worker catch boundary
  (`src/creator/task_creator.cpp:626`). Service-thread exceptions and exceptions thrown in
  retry/error callbacks must reach a query failure handler rather than terminate the process
  or end at a generic pool's log-only catch.
- Unexpected enqueue refusal is sometimes only logged. It must finish the affected query
  with an error, preventing an indefinitely unresolved future.

Initialization should be transactional, recording successful registrations and unwinding only
those registrations. Cleanup must be idempotent and query-specific. If quiescence cannot be
proved after an internal failure, preserve potentially borrowed resources until they can be
retired safely; freeing them best-effort is unsafe.

Ordinary SQL/operator errors, cancellation, and exhausted recoverable OOM retries should be
isolated. Illegal device access, device loss, or a corrupted shared invariant can require a
device/runtime health transition and failure of other affected queries. Clearing a CUDA error
does not prove the device context is healthy. Specify this distinction in the public contract.

### D. Repositories and spilling need explicit borrowing

The manager registry uses shared pointers, but erasure calls `clear_all_repositories()` and
destroys uniquely owned repositories. A spill worker holding a shared manager plus a raw
repository pointer is therefore not protected. See
`src/data/data_repository_manager_registry.hpp:123` and
`cucascade/include/cucascade/data/data_repository_manager.hpp:232`.

Prefer scoped query-owner borrows initially:

1. Quiesce Q and unregister it from new spill-candidate acquisition.
2. Existing spill operations retain a lease for the actual query whose repository/task they
   are accessing. A snapshot of all managers must not pin all queries until a global sweep ends.
3. Tasks extracted for Tier-2 spilling keep their query lease through conversion and disposal
   or resubmission. Their destructor must not access a dead plan or revive a retired query.
4. Wait for Q's actual borrows, tasks, callbacks, and device work, then clear its repositories.

The requesting query and spill victim are different identities: A may need memory that can
be freed by spilling B. Canceling B must not indiscriminately cancel A's memory request;
monitor requests may have no requesting query at all.

An alternative is shared repository handles with a carefully specified close operation.
That requires auditing every raw port reference and the semantics of clearing contents while
borrowers still exist. Do not restore the old failed migration mechanically. The owner/lease
approach should be prototyped against repository, pin, and materialized-result lifetimes before
deciding whether a wider cuCascade ownership API change is necessary.

The engine-owned plan currently dies before outer scope cleanup
(`src/sirius_interface.cpp:127`, `src/sirius_context.cpp:445`). Successful execute already
performs drains, so this is not evidence of a UAF on every success. Make the destruction order
structural for errors and partial initialization as well.

### E. Memory pressure must not block all dispatch or cancellation

One GPU's manager can block in a reservation or downgrade future
(`src/pipeline/gpu_pipeline_executor.cpp:227`, `:289`, `:308`). It then cannot dispatch other
tasks for that GPU, even if those tasks would release the memory being waited for. Current
reservation waits have no per-query cancellation.

Use a nonblocking reservation attempt and a tracked waiting state awakened by memory progress
or cancellation. Release scarce worker capacity while waiting; moving all blocking waits into
workers merely creates a different pool starvation risk. Waiting tasks remain visible to
spilling and lifetime accounting. Detect terminal no-progress/resource exhaustion and report
a bounded query failure instead of retrying forever.

Maintain FIFO among compatible runnable tasks. Under pressure, prioritize spilling and allow
memory-releasing work to make progress when an older task cannot run. This does not require
round-robin query fairness or equal GPU shares. Reserve sufficient capacity for spill and
completion paths, and size sort/result buffering from reservations rather than assuming all
currently free device/host memory belongs to one query.

### F. FIFO matching and readiness accounting need finishing

The scheduler currently takes an exact-device match before checking any-device work
(`src/pipeline/task_scheduler.cpp:368`). Newer device-specific work can beat older compatible
work. Compare the best compatible priorities atomically, including retries.

Readiness is consumed after dispatch even when the destination rejects a canceled task
(`src/pipeline/task_scheduler.cpp:416`). Define a one-use readiness token and an explicit
dispatch acceptance result, so rejected work neither loses capacity nor creates duplicate
credits when executors resume.

Use an admission-arrival sequence for query FIFO. Current query IDs are 32-bit and priority
packing truncates to 31 bits (`src/query_id.hpp:43`, `:69`). Prefer 64-bit identity plus a
structured priority comparator, updating all code that extracts identity from priority. An
explicit exhaustion guard is a smaller alternative, but silent priority/identity wrap is not
a suitable long-running service contract.

### G. Configuration and observability need query ownership

Planning, GPU selection, engine initialization, and scans currently read settings at separate
times. Snapshot once before physical planning and pass the same immutable query options
through every stage. SQL settings that appear session-local should have ClientContext-local
overrides of YAML defaults. Genuine runtime settings need synchronized publication or maintenance.

Existing unguarded writes include hardware decompression
(`src/sirius_extension.cpp:3159`), expression strategy (`:2665`), and logging configuration
strings (`:2878`). Decompression also controls process initialization behavior; recommend
startup-only semantics until a deliberate runtime update contract exists. Audit live Super
Sirius readers rather than refactoring dead legacy settings indiscriminately.

`batch_telemetry_registry::on_query_end()` clears all placements and port mappings
(`src/telemetry/batch_telemetry.cpp:475`). Add query ownership to placements/ports and retire
only Q's records. Shared pinned batches can have placements in multiple queries, so a batch
ID alone is insufficient. Global telemetry installation/shutdown remains a separate lifecycle.

Expose counts/timestamps for queued, admitted, waiting-for-memory, retiring, and completed
queries, with per-query error reasons. Tests need these to prove overlap and bounded admission.

### H. Pinning, cache and prefetch need several distinct fixes

**Pinning.** Current shared entry snapshots and per-query MVCC masks are useful. Publish a
complete immutable pin generation atomically, including identity, metadata, chunk placement,
compression information, and visibility information. Keep pin replacement/unpin exclusive
initially; later online replacement can retire generations after readers release leases.
Replace `use_count()` heuristics with actual reader ownership if online mutation is introduced.
Test overlapping scans with different transaction snapshots, compressed HOST/GPU pins,
mutation checks, and dynamic filters. Do not introduce cross-query in-place mutation of
shared pinned chunks during mask application, prefetch, or spill.

**Identity.** Compression plan registration is process-global and keyed by bare table name
(`src/compression/plan_register.cpp:33`). Use runtime/catalog/schema/table identity and atomic
resolve/publication, or execution-local compression-plan selection. Separate DatabaseInstances
with same-named tables must not share a plan unintentionally.

**Cache lifetime.** Cache reset relies on SlotGuard exclusivity while datasources borrow raw
cached-chunk handles (`src/scan_manager/sirius_scan_manager.hpp:545`). Retain an exclusive
maintenance drain before replacing caches. Generation-owned caches are a possible later
optimization, not necessary for initial correctness.

**Cache bookkeeping.** `_last_reported` is plain mutable state written in prepare and read
in summary (`src/io/cache/prefetching_cache.cpp:1085`, `:1099`). Synchronize it or make query
summary baselines query-owned. The Iceberg delete-data memo is globally cleared on QueryEnd
(`src/sirius_context.cpp:366`); scope its identity/retirement to the relevant transaction/runtime
so one query does not invalidate another's useful memo. Its mutex/shared ownership already
prevents a simple cache-clear UAF; do not overstate this as a proven wrong-result bug.

**Scan capacity.** Audit producers, coalescers, and waiters against actual scan-slot counts.
The current producers-first ordering prevents simply reproducing the old sequencer deadlock
argument. Prove progress with more scans than pool threads and multiple admitted queries;
use separate sequencing capacity or nonblocking continuations if bounded progress cannot be
established. Do not assume N extra threads prove the invariant.

**Prefetch budgets.** Each query's readahead manager can grant itself the full backend scan
budget and reacts to events without filtering query identity. Add a shared backend/device
budget with per-query queues, relevant event routing, cancellation, and FIFO demand priority.
Demand reads and spilling must make progress ahead of speculative prefetch. Deduplicated reads
shared by A and B need subscriber ownership: cancel A's interest without invalidating B's bytes.

### I. Multiple GPUs need explicit stream and placement work

Existing device affinity, per-task CUDA device guards, per-query GPU lists, and event fencing
provide a foundation. Query GPU selection currently chooses a prefix of the configured fleet;
that is a utilization limitation rather than proof of incorrect concurrent execution. Sharing
GPUs safely does not require assigning an exclusive GPU to each query.

Two concrete concerns remain:

1. `get_target_ctas()` uses an unsynchronized mutable cached-device/count pair
   (`src/cuda/scan/strings/common.cuh:255`). One caller can observe the device as initialized
   before the count is written, obtain zero, and pass it to the chunking division. The helper
   has live string decoder callers. Replace with a thread-safe immutable per-device cache.
2. GPU memory prefetch workers retain streams from a fixed round-robin pool
   (`src/scan_manager/memory_prefetcher.cpp:45`; cuCascade memory_space.cpp:119). Enough users
   reuse a stream. With per-stream reservation tracking, another user's reset can erase the
   tracker while an allocation uses its returned raw pointer
   (`cucascade/src/memory/reservation_aware_resource_adaptor.cpp:55`, `:78`, `:410`). Shared CUDA
   streams alone are not necessarily unsafe; this reservation ownership contract is the issue.
   Extend exclusive stream leases or otherwise guarantee safe tracker ownership for all memory
   consumers, including prefetch, dynamic filters, and conversions. Bound acquisition and
   integrate cancellation; do not hold all leases indefinitely in parked prefetch workers.

Separate storage I/O prefetch from GPU memory prefetch. The latter is currently disabled when
more than one GPU memory space exists (`src/scan_manager/sirius_scan_manager.cpp:1910`). To
meet the full prefetch-plus-multiple-GPUs goal, add target-device selection consistent with
task affinity, per-device budgets/reservations, and completion events before consumption or
reuse. Test concurrent queries sharing and spanning devices, including spill and peer-copy
fallback. Multi-GPU late materialization is independently disabled and can remain an explicit
unsupported optimization unless included in scope.

Dynamic-filter publication still has a device-wide synchronization fallback. Replace it with
writer-event/stream dependencies where possible; until then it is a serialization cost, not
evidence that all GPU work must run simultaneously to satisfy query concurrency.

## 4. Proposed ownership and retirement protocol

Suggested execution owner contents:

- Stable query identity and FIFO admission ticket.
- Admission permit, immutable options, cancellation state, and first-error/completion state.
- Plan/engine ownership and registered creator, scheduler, scan, and repository handles.
- Work/borrow accounting and required CUDA completion fences.
- Per-query telemetry and cache/pin generation references.

Avoid shared-pointer cycles between the owner, plan, pipelines, and tasks. A lifetime control
block may be separate from the object owning the plan; disposal occurs from an external scope
after work counts reach zero. Materialized results must own their returned buffers independently
or carry whatever lifetime handle their use still requires.

Retirement order:

1. Mark Q quiescing atomically with closing publication/borrow acquisition.
2. Signal cancellation when needed and settle outstanding publishers.
3. Detach Q's queued work, unregister future spill candidates, wake/cancel Q's waits, and stop
   Q's scan/prefetch producers. Other queries retain their queues and workers.
4. Wait for Q's in-hand/running work, repository borrows, callbacks, and CUDA operations.
   Destroy detached tasks outside queue locks, with the plan still alive.
5. Retire Q's telemetry and release its scan state, repositories, and plan in dependency order.
6. Close Q's registry entry and release its admission permit.

Logical result completion and resource retirement are separate milestones. SQL should retain
the existing property that an unconsumed materialized result does not indefinitely occupy an
execution permit. Runtime shutdown closes admission, wakes queued requests, cancels/drains
active owners, then stops shared executors and memory resources.

## 5. Implementation sequence and gates

| Phase | Concrete deliverable | Completion evidence |
|---|---|---|
| 1. Harness and lifecycle specification | Shared-runtime multi-connection fixture; test barriers/fault injection; active-query observability; cancellation/maintenance contract | N=1 baseline; deterministic handoff tests demonstrate the current gaps without relying on sleeps |
| 2. Query owner and work leases | Transactional registration; continuous work/borrow tracking; plan survives retirement; queue publication protocol | Pause before push, after pop, during routing/retry/spill; Q cleanup waits while B continues; partial-init rollback leaks nothing |
| 3. Query-local teardown and errors | Remove per-query shared-manager interruption and global downgrade drain; correct readiness; complete exception boundaries; cancellation | A fails/cancels, B returns correct results, C enters afterward; simultaneous failures do not race thread joins; unknown IDs remain rejected |
| 4. Settings, metadata and maintenance | One query options snapshot; synchronized runtime settings; query telemetry; immutable pin publication/cache reset protocol; shared-state race fixes | Concurrent settings/planning, different MVCC snapshots, maintenance waiting/cancellation, same-name catalogs, telemetry preservation |
| 5. Memory progress and FIFO | Cancellable pending reservations/downgrade; tracked spill victims; compatible-task priority comparison; bounded prefetch budgets | Low GPU/HOST memory, full/no disk, younger retained data, waiting-query cancellation; bounded success/error with no manager starvation |
| 6. Bounded admission | Single authoritative YAML limit; FIFO arrival tickets; shared planning/exclusive maintenance integration; closing protocol | Genuine overlapping windows for N=2 and representative N; N+1 queues; queued cancel; maintenance eventually runs; N=1 regressions pass |
| 7. Multi-GPU and prefetch qualification | Safe stream/reservation ownership; device-aware memory prefetch; placement and failure validation | Two or more real GPUs, sharing/spanning devices, peer-copy fallback, compressed pins, dynamic filters and spill with one query failing |
| 8. Release validation and documentation | Stress matrix, CPU result comparison, sanitizer coverage, updated docs/config examples | Repeated start/finish/failure cycles with no retained query registrations or monotonic resource leaks; full supported feature matrix documented |

Phases 2–5 are prerequisites to making N>1 a supported production setting. Isolated tests can
exercise overlap earlier through a test-only admission path. Configuration and scan/multi-GPU
work can be developed alongside the ownership work once the shared interfaces are settled.
Phase 7 is part of the requested full outcome, not an optional post-release promise.

Likely implementation boundaries are separate PRs for harness/lifecycle primitives; ownership
and retirement; errors/cancellation; settings and maintenance; memory progress/FIFO; admission;
and multi-GPU prefetch plus qualification. Exact splits depend on avoiding unsafe intermediate
states; keep the production default single-flight until the prerequisite gates pass.

## 6. Validation matrix

- Separate Connections on one DatabaseInstance, with barriers proving at least two execution
  windows are simultaneously admitted. Threaded callers that serialize internally do not pass.
- Limits 1, 2, a representative higher N (for example 4 or 7), and N+1 queued requests.
- Transparent SQL, explicit gpu_execution, prepared re-execution on independent connections,
  CPU fallback, discarded materialized results, and connection close.
- Faults during registration, scan setup/I/O/decode, creator hints, operator execution, task
  dispatch/retry, spill, and cleanup; assert one completion/error per query.
- Cancellation while queued, initializing, scanning, waiting for memory, and executing.
- Same/different tables and files; GPU/HOST/compressed pins; transaction visibility; dynamic
  filters; cache hit/miss/reset; unpin/repin; local and configured remote I/O.
- Memory-pressure sorts/joins/aggregates; host full, disk full, no disk, spill failure, OOM
  retries exhausted, and a younger query retaining memory needed by an older query.
- Two or more GPUs, shared device sets, queries spanning devices, disjoint subsets when the
  selection API supports them, affinity-constrained tasks and no-preference tasks, retries,
  peer access and host-staging fallback. Require actual multi-GPU runs, not skipped tests.
- CPU control-plane TSan, ASan lifetime/error tests, and reduced GPU sanitizer scenarios as
  supported by the toolchain. Sanitizers supplement deterministic barriers and result checks.
- Concurrent TPC-H/mixed workloads checked against CPU results. The vendored SQLLogic runner's
  `concurrentloop` creates one Connection per iteration
  (`duckdb/test/sqlite/sqllogic_command.cpp:362`) and can supplement the C++ harness.
- Reuse the watchdog subprocess support in `test_query_lifecycle_slot.cpp`; replace its
  single-flight expectations deliberately. Wire the existing uninvoked concurrent logging
  variant into a test, and add co-tenant completion assertions to component tests.

No timings, reproduction claims, or sanitizer results are implied by this source-only analysis.

## 7. Decisions for discussion

| Decision | Recommended starting point | Why it matters |
|---|---|---|
| Limit configuration and scope | Startup `sirius.max_concurrent_queries`, default 1, per DatabaseInstance; migrate old scan setting | One enforceable limit avoids scan capacity drifting from admission |
| FIFO under memory pressure | Oldest compatible runnable task first; permit spill/memory-releasing progress when older work cannot run | Strictly blocking all younger progress can create dependency cycles |
| PIN/UNPIN/reset/index maintenance | Wait cancellably for exclusive access, blocking newer admissions | Smallest safe initial contract; online generation replacement adds complexity |
| SQL SET semantics | Session-local query settings over YAML defaults; runtime-wide settings explicitly synchronized/startup-only | Prevents one connection changing another query midway through planning/execution |
| FFI/streaming scope | Explicitly decide whether first release includes concurrent fragments | FFI Context shares a connection; Fragment retains a window across build/run; dependent fragments can deadlock bounded admission |
| GPU allocation | Keep shared GPUs and existing subset rules initially; add load-balanced/disjoint placement separately if desired | Correct shared execution does not require dedicated GPUs per query |
| Failure boundary | Ordinary query errors isolated; genuine device/shared corruption follows explicit health policy | A CUDA-context fault may affect all users of that context |

If concurrent FFI/streaming is included, add a dedicated phase: separate execution connection
and catalog ownership, transferable query permits rather than thread-owned retained mutexes,
admission at the appropriate build/run boundary, cancellation of input/output waits, and a
strategy for producer/consumer fragments whose dependencies exceed N. Building dormant
fragments must not consume all permits needed to run their producers. SQL-only qualification
does not establish this API's concurrency safety.

## 8. Deferred optimizations and uncertain claims

Do not make these prerequisites without evidence: equal-share scheduling, preempting running
GPU kernels, load-balanced exclusive GPU assignment, online cache reset, fully concurrent pin
replacement, or enabling multi-GPU late materialization. They can be evaluated after the
required correctness, progress, and prefetch behavior is established.

The ownership design needs a prototype and fault tests; this investigation does not prove it
covers every operator-specific race. In particular, current pin conversion/mask/event contracts,
reservation release, materialized-result ownership, and all asynchronous producer entry points
need verification while integrating the new owner. Keep the historical register as background,
but track new completion claims against these source-level invariants and executable tests.

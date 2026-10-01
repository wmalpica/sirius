# Concurrent query execution: per-query control plumbing (steps 3-12)

Continues the work of making Sirius internals safe for concurrent queries.
The single-flight mutex (`SiriusContext::query_lifecycle_mutex_`) is still in
place, so Sirius still executes one query at a time. What changes here is the
control plumbing behind it: teardown, drains, waits, validation and error paths
were process-wide stop-the-world operations invoked once per query, and they are
now scoped to the query that triggered them.

Range: `ff33f389..b38e6e3f` (9 commits), 20 files, +1071 / -225.

The steps referred to are from docs/concurrency/option-d-recommended.md

Validation: full C++ suite (2255 cases, ~32.5M assertions) run 3 consecutive
times per step, all green. Single-GPU host, so multi-GPU cases self-skip.

---

## Commit by commit

### 1. `ff33f389` docs(concurrency): record C1/C2 pull-forward into step 2

Docs only.

Step 1 deleted the `stop()` calls that used to pre-stop the `task_creator`. That
made `drain_after_error -> stop_thread_pool()` race a live manager thread and
exposed the C1 deadlock: `stop_thread_pool()` held `_global_state_mutex` across
`_manager_thread.join()`, while `manager_loop()` needs that same mutex in
`get_query_task_global_state()`. It hung the full suite in roughly 3 of 8 runs.

The plan's own dependency note said "C1 must be fixed before A5/A6 are
exercised, because both call `stop_thread_pool()` today", and scheduling C1 for
step 4 violated it. C1 and C2 were pulled forward into step 2 and are recorded
there. The generalizable lesson is also recorded: any step that stops
pre-stopping a subsystem makes that subsystem's lifecycle races live in the same
commit.

Files: `docs/concurrency/option-d-recommended.md`

### 2. `7bdf050e` concurrency step 3: per-query completion and error paths

`wait_for_completion` and `drain_after_error` took a `query_id` but their bodies
were process-wide.

- Producer halt: both called `_task_creator->stop_thread_pool()`, tearing down
  the SHARED creation pool and interrupting the shared creation queue on every
  completion and every error. Replaced with the lifecycle gate's
  `quiesce(query_id)`.
- Queue validation: `wait_for_completion` checked the WHOLE queue size, so query
  A completing normally threw "pipeline task queue not empty at query
  completion" because query B had work legitimately queued. Now
  `size(query_index{A})`, with a matching
  `itask_executor::wait_and_validate_empty(query_id)`.
- Queue draining: `drain_after_error` called the unindexed `_task_queue.drain()`
  twice plus `gpu_exec->drain_and_wait()`, destroying every co-tenant query's
  queued tasks and leaving those queries waiting on completions that could never
  arrive. Now `drain(query_index{})` and a new
  `wait_and_drain_query(query_id)`.
- Guards `*_ready_devices.begin()`, which dereferenced an empty vector when the
  management loop woke on a `task_available` event with no ready device.

Known gap carried forward at this point: the in-flight wait still brackets in
`quiesce_manager()`/`resume_manager()`. That bracket is not optional -
`manager_loop()` reserves a pool slot and then blocks in `pop()`, so an idle
manager holds an active slot forever and `wait_all()` (which waits for
`active_ == 0`) never returns. Removing it deadlocked the suite at test 5 in 4
of 4 runs. Because interrupting makes `push()` return false for the duration, a
co-tenant task in transit could still be dropped.

Closes register A5, D4; A6 partially.

Files: `src/include/parallel/task_executor.hpp`, `src/parallel/task_executor.cpp`,
`src/pipeline/task_scheduler.cpp`, `test/cpp/pipeline/test_task_scheduler.cpp`,
`docs/concurrency/option-d-recommended.md`

### 3. `8a09bebc` fix(util): bound the crash handler so a segfault dies instead of spinning

Not in the original plan. Added because it actively obstructed debugging the
rest of this work.

`segfault_handler()` armed a deadline only around the log flush, leaving the
dangerous work unbounded. Only `backtrace()` and `backtrace_symbols_fd()` are
async-signal-safe; `backtrace_symbols()` goes through `_dl_addr()`, which takes
the dynamic loader lock, and `__cxa_demangle()` plus the log flush allocate. If
the crash happens while any thread holds the loader lock, the malloc arena lock
or the logging mutex, none of those calls fail - they spin forever. The process
then looks alive rather than dead: 100% CPU, no output, unresponsive to SIGTERM.

This disguised an ordinary `task_creator` deadlock as a live spinning process
and cost about an hour of misdiagnosis, because every symptom (running thread,
`wchan=0`, other threads idle) pointed at a live process.

Two changes:
- `arm_handler_deadline()` now fires on handler entry and covers everything, so
  a blocked symbolization or flush terminates the process within 10s.
  `flush_logs_best_effort()` no longer manages or cancels the alarm.
- The async-signal-safe backtrace is emitted FIRST via `backtrace_symbols_fd()`,
  so a usable stack reaches stderr even if the demangled pass blocks.

Verified by reproducing the original failure mode (SIGABRT delivered mid-run
with the handler enabled). Before: spun until force-killed, no output. After:
raw frames plus the demangled backtrace, process exits about 1s after the
signal.

Files: `src/util/segfault_backtrace_handler.cpp`

### 4. `5d93da7f` concurrency step 4: close the in-flight tracking holes and unstarve lookahead

- D1: `get_operator_for_next_task()` dereferences the operator recursively, via
  `get_next_task_hint()`, on the manager thread; `enter_in_flight()` ran after
  it. So `drain_pending_tasks(query_id)` could observe `in_flight == 0` and
  return while the manager was still walking that query's operator graph - and
  the caller's next act is to let the plan be destroyed. The counted region now
  opens before that dereference.
- D2: the creation queue's key extractor read `request.node->type` at push time,
  inside the queue mutex, on a queue every query shares. A `schedule()` racing
  its query's teardown would read a freed operator while holding that mutex. The
  operator type is now captured at `schedule()` time into the request, like
  `priority` and `query_id` already were.
- D3: lookahead returned on the first non-open entry instead of skipping it.
  Since the map is ordered oldest-first and the oldest entry is routinely a
  finished-but-not-yet-reset query, that suppressed lookahead for every younger
  query for the whole cleanup window. It now skips to the oldest query that is
  actually accepting work, preserving FIFO.
- A2 (self-deadlock half): `stop()` from one of its own pool workers is now an
  assert, backed by a thread_local marker. `do_stop_thread_pool()` calls
  `wait_all()`, which blocks until `active_ == 0`, but the calling worker IS an
  active slot.

Closes register D1, D2, D3, and the self-deadlock half of A2.

Files: `src/creator/task_creator.cpp`, `src/include/creator/task_creator.hpp`

### 5. `6d22cb6b` concurrency step 5: query-aware thread pool with per-query waits

`bounded_thread_pool` now tracks work per query, so teardown can wait for one
query without stopping the executor - closing the gap step 3 had to carry.

Slots gain an optional query attribution via `slot::attach(query_id)`, applied
once the task's query is known rather than at `reserve()` time. That distinction
is load-bearing: every manager loop reserves a slot and THEN blocks in `pop()`,
so an idle manager holds a slot indefinitely. Counting it against a query would
make that query's wait block until the manager happened to receive work, and
counting it globally is why `wait_all()` cannot be used at all. Untagged slots
are therefore invisible to the per-query waits, and a test pins exactly that.

Two operations, because the two callers need different things:

- `wait_for_query(q)` - waits until q has nothing queued or running, running
  everything it finds. The SUCCESS path. Since `attach()` precedes `dispatch()`,
  the per-query count already covers queued-but-unstarted work, so reaching zero
  means every task actually ran.
- `drain_and_wait(q)` - discards q's queued work, then waits for what is already
  running. The ERROR path, where the query is failing and its plan is about to
  be destroyed.

Wiring `wait_and_validate_empty()` to `drain_and_wait()` was a bug caught by
these tests: on the success path it would have silently discarded work the query
legitimately scheduled and then reported success.

`wait_and_validate_empty(query_id)` now uses `wait_for_query` with no quiesce
bracket, so a successful completion no longer interrupts the shared queue.
`wait_and_drain_query` (error path) keeps the bracket deliberately: a failing
query can still have tasks the manager may pop at any moment, and between
`pop()` and `attach()` a task belongs to neither the queue nor the count.

`task_creator`'s bespoke in-flight counter is deleted in favour of the pool's.
`work_queue_` becomes `std::list` so `drain_and_wait` can remove one query's
items from the middle; dropped items are spliced out under the lock and
destroyed after releasing it, because destroying a work item runs `~slot`, which
re-enters `release_slot()` and would deadlock on a held mutex.

Closes the A6 remainder on the success path.

Files: `src/include/exec/bounded_thread_pool.hpp`, `src/creator/task_creator.cpp`,
`src/include/creator/task_creator.hpp`, `src/parallel/task_executor.cpp`,
`src/pipeline/gpu_pipeline_executor.cpp`,
`test/cpp/exec/test_bounded_thread_pool.cpp`,
`docs/concurrency/option-d-recommended.md`

### 6. `e5078162` docs(concurrency): record steps 6+7 as attempted and backed out

Docs only. No code from steps 6 or 7 is in this branch.

Steps 6 and 7 (shared-ownership `data_repository` with `close()`, and deleting
the global downgrade drain) were implemented in full and then reverted, on a
deterministic SIGSEGV at the 6th test ("pin_table compression - result equality
vs uncompressed pin") in
`sirius_physical_ungrouped_aggregate_merge::get_next_task_input_data()`, on a
`task_creator` pool worker. Faulting instruction `mov 0x8(%rcx),%rdi`; the
source line is `ports.begin()->second->repo->pop_next_data_batch()`.

Ruled out: not pre-existing (verified by stashing both halves and rebuilding at
step 5); not an unhooked virtual override (nothing derives from
`data_repository`); not an out-of-range partition (`pop_next_data_batch`
range-checks and throws).

The work is preserved in named stashes rather than discarded:
- `cucascade/` stash: `step6-cucascade-shared-ptr-repositories`
- sirius stash: `step6b-7-sirius-side`

Consequence: the global `executor->drain()` is still in
`run_mandatory_cleanup`.

Files: `docs/concurrency/option-d-recommended.md`

### 7. `48c114bc` concurrency steps 8+11: shutdown ordering, failed-cleanup leak, loud drops

- B6 shutdown ordering: `terminate()` destroyed `task_scheduler_` second, while
  the downgrade executors and the creation pool were still live. Each downgrade
  executor holds `_pipeline_task_queue`, a raw pointer into
  `task_scheduler::_task_queue`, and dereferences it in `processing_loop`;
  creation lambdas hold `_task_scheduler`. A monitor request or an in-flight
  creation landing in that window dereferenced a freed queue at DB teardown.
  Every producer and borrower is now stopped first.
- B7 declaration order: documented as the backstop for the one path where
  `terminate()` never runs - `initialize()` throwing after `task_scheduler_` is
  constructed leaves `is_initialized_` false, so `~SiriusContext` skips
  `terminate()` and only the member order decides who dies first.
- B10 destructor: `~SiriusContext` is noexcept but called a `terminate()` that
  opens with `throw_if_not_initialized()` and goes on to `reset_all()`,
  `registry.clear()` and `memory_manager_->shutdown()`. Any throw was
  `std::terminate`, during `~DBConfig`-triggered destruction. Now caught and
  logged.
- B8 failed-cleanup leak: `drop_query_runtime_state_best_effort` never erased the
  repository registry. It runs precisely when `run_mandatory_cleanup` threw
  part-way, and once the runtime is latched unavailable no later window runs the
  erase either - so the failed query's manager and every batch in it survived
  until `terminate()`: GPU/host memory never returned, and the downgrade
  executors kept sweeping it every monitor cycle for the life of the process.
- A9 loud drops: `multi_index_priority_queue::push` returns false for exactly one
  reason - the queue is interrupted - and `task_creator::schedule`,
  `schedule_lookahead` and `task_scheduler::schedule` all discarded it. A refused
  push destroys the request, so a live query silently loses a task it is waiting
  on; that silence is what turned every dropped-work bug in this subsystem into
  an unexplained hang. The lifecycle gate makes the check precise: a drop for a
  query the gate still reports as accepting work is an ERROR, while a drop for a
  quiescing query is the documented teardown contract and stays at DEBUG.

Closes register A9, B6, B7, B8, B10.

Files: `src/sirius_context.cpp`, `src/include/sirius_context.hpp`,
`src/creator/task_creator.cpp`, `src/include/creator/task_creator.hpp`,
`src/pipeline/task_scheduler.cpp`

### 8. `a9a85ed4` concurrency steps 9+12 (partial): prefetch-cache UAF, real max_concurrent_queries

- B11 prefetch cache: two sites dereferenced a `_file_cache` iterator after
  releasing the lock. `get_or_create_file_entry` scoped its `unique_lock` to the
  `if` block, so `return *it->second` ran with NO lock held; the other site
  called `lk.unlock()` explicitly and then used `it`. Either way a concurrent
  insert of a DIFFERENT file rehashes the map and invalidates the iterator. Both
  now carry the `file_entry*` out under the lock: the entry is heap-allocated and
  stable across a rehash, only iterators and references into the map are not.
  `fetch_chunks` is still called unlocked, deliberately, which is exactly why the
  pointer rather than the iterator is what crosses. Reachable only with
  concurrency: a single query first-touches every file in one prepare.
- C3 scan-manager bound: `k_max_concurrent_queries` was a compile-time constant
  of 1 carrying a TODO. It is now
  `scan_manager_config::max_concurrent_queries`, parsed from YAML as
  `scan_manager.max_concurrent_queries`, still defaulting to 1 so existing
  single-query deployments are byte-identical. It sizes the scan thread pool as
  `num_threads + max_concurrent_queries`, which is the constraint that matters:
  each query parks one BLOCKING coalescer sequencer on that pool, unblocked only
  by that query's own split tasks running on the same pool, so raising
  concurrency without resizing the pool deadlocks by starvation.

Step 9's per-query batch telemetry (A8) and step 12's concurrency harness are
NOT included.

Closes register B11, C3.

Files: `src/io/cache/prefetching_cache.cpp`,
`src/include/scan_manager/config.hpp`,
`src/include/scan_manager/sirius_scan_manager.hpp`,
`src/scan_manager/sirius_scan_manager.cpp`, `src/sirius_config.cpp`

### 9. `b38e6e3f` docs(concurrency): execution summary and recommended next steps

Docs only. Adds `docs/concurrency/99-execution-summary.md`: what was built, what
was backed out and why, known gaps in what was delivered, and ordered next
steps. Also records three things found that were not in the plan (the crash
handler, the violated dependency note, and a 9.4 GB core dump accidentally
committed during debugging and scrubbed before push).

Files: `docs/concurrency/99-execution-summary.md`, `docs/concurrency/README.md`

---

## What is still left to do

23 of the 44 issues in `docs/concurrency/00-issue-register.md` are closed across
the whole branch. Sirius still runs one query at a time; the single-flight mutex
has not been touched. The items below are in the order they should be tackled.

### 1. Build the concurrency test harness (register G1-G3) - do this first

Everything on this branch was validated at N=1. The per-query code has still
never run under real concurrency, so "2255 tests green" is narrower assurance
than it looks. Every test added here is single-threaded or exercises one
subsystem with synthetic query ids; nothing runs two QUERIES end to end.

Salvage the primitives currently trapped in
`test/cpp/integration/test_query_lifecycle_slot.cpp`'s anonymous namespace
(start gate, `async_query_result`, `scoped_blocking_window_log_sink`) into
`test/cpp/utils/`, and let a test opt out of `shared_env_listener`'s
env-pausing. `concurrentloop` already works in the vendored DuckDB and
`ParallelExecuteLoop` constructs a new `Connection` per parallel iteration on
its own thread against the shared `DatabaseInstance`, which is exactly the shape
needed.

### 2. Finish steps 6+7 (register B1, B2, B3, B4, B9, A7, D6, F8)

Do NOT resume by re-reading the diff. Add a temporary print of `ports.size()`
and `repo == nullptr` immediately before the deref in
`get_next_task_input_data`. That single data point separates the three remaining
hypotheses; none of the black-box debugging did. Then unstash
`step6-cucascade-shared-ptr-repositories` and `step6b-7-sirius-side`.

Remaining hypotheses, in suspicion order:
1. `get_repository()` returning `shared_ptr` BY VALUE while
   `materialize_repository_wiring` takes `.get()` on the temporary - the one
   lifetime change on the exact pointer that crashes.
2. The manager's `add_data_batch` moving per-repository calls outside `_mutex`.
3. `close()` leaving `_data_batches` size 0 rather than one empty partition,
   changing `num_partitions()` for anything reading it after close.

This matters more than its position suggests: FIFO fairness depends on spilling
working, because the livelock case is an older query blocked on memory that only
a newer query could release by finishing. Steps 6/7 are what make the downgrade
path per-query. Until then the global `executor->drain()` remains in
`run_mandatory_cleanup` and is the worst cross-query serialization point.

### 3. Close the error-path window (A6 remainder)

`wait_and_drain_query` still brackets in `quiesce_manager()`/`resume_manager()`,
which interrupts the shared queue and can drop a co-tenant's in-transit task.
The bracket cannot simply be deleted - see step 3 above. Fix: have the manager
publish its in-hand task's query BEFORE the task leaves the queue's lock, e.g. a
per-query "in-hand" counter incremented under the pop, so the `pop()`-to-
`attach()` window is covered. Then the error path can drop the bracket like the
success path did.

### 4. Per-query batch telemetry (A8)

`batch_telemetry_registry::on_query_end()` takes no query id: it consumes every
live placement across all 16 shards and clears `impl_->ports`, so one query's end
silently truncates every co-tenant's telemetry. Add `query_id` to placements and
ports, re-key `ports` on the existing `port::source_port_uuid` instead of a raw
`data_repository*` (a key is not an owner - address recycling silently matches
stale entries), and add an `on_query_end(query_id)` overload.

### 5. Head-of-line blocking (C4)

One manager thread per GPU performs a blocking `make_reservation` AND a blocking
downgrade `.get()` while holding a reserved slot, so one query's memory-hungry
task blocks every other query's dispatch to that GPU. At the 3-7 concurrency
target this is the throughput ceiling. Move reservation acquisition into the
dispatched job, or use `make_reservation_or_null` plus requeue-with-backoff.

### 6. Shared mutable configuration (E1-E7) - required before removing the mutex

`operator_params` is one non-atomic struct per `DatabaseInstance`, written by
about 20 `SET` callbacks and read mid-plan and mid-execution. The single-flight
slot is the ONLY thing serializing those writes today. The `duckdb::Config`
static variables have no guard at all, and `EXPRESSION_EVALUATOR_STRATEGY` is
read as a default argument on every `expression_evaluator` construction, so one
connection's `SET` can change strategy between two operators of another
connection's plan. Move both into per-`ClientContext` settings and snapshot into
the plan at window begin. Also here: `PhysicalSiriusExecution::logical_plan_` is
`mutable` and `reset()` from a `const` source method (E4), and
`plan_register::global()` is a check-then-act keyed on a bare table name (E5).

### 7. Remaining fairness and performance items (F2, F3, F5, F6, F7, F9)

Narrow the plan-time `SlotGuard`; live prefetch-epoch set instead of a single
global ticker; replace the `use_count() > 1` pin guard with a per-entry reader
count; reservation-driven sizing in sort and result-collector; remove
`cudaDeviceSynchronize` and default-stream work from the hash-join dynamic-filter
publish; give the cucascade reservation and stream waits a per-query dimension.

### 8. Then raise the cap and remove the single-flight mutex

`scan_manager.max_concurrent_queries` and the pool sizing are in place. Raise it,
run the harness from item 1 under ASan and TSan, and expect to find more - 23
issues closed is not 44.

### Known loose ends not covered above

- One unexplained intermittent failure: "query lifecycle slot is released for an
  unconsumed result" -> "watchdog child exited abnormally", seen once in about 16
  runs and not reproduced in follow-ups. It is a forked-child test, and the
  `DISABLE_SIRIUS_SIGNAL_HANDLER=1` debugging harness changes how a crashing
  child dies, so it may be an artifact of that.
- Pre-existing and unrelated: running only `[integration]` reproducibly fails
  `test/cpp/integration/test_pin_table_mvcc_foundation.cpp:201`
  (`probe.counts.size() >= 2` yields 1), while the same test passes in
  full-suite runs. Order-dependent test isolation.
- Hygiene (register H): dead members `task_creator::_client_context`,
  `sirius_engine::wait_for_query_finish` and friends, `task_completion.hpp`,
  orphaned `exec::inspectable_mpsc`, the dead `schedule(node, query_id)`
  overload, `task_creation_request::device_id`; the 32-bit query id wrap and
  31-bit priority mask (H8); and five drifted docs under `docs/super-sirius/`
  that still describe deleted APIs.

## Testing notes for reviewers

- Full suite: `pixi run build/release/extension/sirius/test/cpp/sirius_unittest`.
  A healthy run is about 345s on a single-GPU host. Anything much longer is a
  hang, not slowness.
- Do not pipe the run to `tail` while diagnosing - `tail` emits nothing until its
  input closes, which makes a live run look identical to a dead one.
- `DISABLE_SIRIUS_SIGNAL_HANDLER=1` gives real core dumps fast (see
  `docs/super-sirius/debugging.md`).
- Put `timeout` INSIDE `pixi run`, not outside: `timeout -s ABRT 600 pixi run ...`
  signals pixi's wrapper and yields a useless Rust backtrace. Use
  `pixi run bash -c 'exec timeout -s ABRT 600 <binary>'`.
- Validate with at least 3 consecutive full-suite runs. Two of the bugs fixed in
  this branch appeared in roughly 1 run in 3.

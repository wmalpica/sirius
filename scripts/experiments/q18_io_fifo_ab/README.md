# Q18 incoming reactor queue A/B experiment

These are diagnostic branches, based on local commit
`88cff3c074d419143d7646cab33b8605b6fd19b6` (Sirius-owned NUMA-assigned Parquet
coalescing). They include the workstation's existing preparation/task telemetry
and the new incoming-reactor enqueue/dequeue telemetry. They are **not** a rebase
onto current main, nor a claim to reproduce remote-only commit `3c53d0be` byte for
byte. The two branches share the exact same instrumented source except for the
queue change described below.

- A: `codex/q18-io-queue-instrumentation`: original implicit-producer queue.
- B: `codex/q18-io-queue-fifo`: A plus a single explicit producer token shared
  under the existing enqueue mutex. FIFO is per reactor across its producers,
  not a single queue across all reactors. No changes to reactor selection,
  grouping, staging, admission limits, affinity or metadata preparation policy.

The only A-to-B diff is in `src/io/uring/uring_reactor.{hpp,cpp}`. The scripts
are identical on both branches. These branches retain earlier instrumentation;
the unrelated local `.claude/claude-tools` and untracked `vcpkg/` are excluded.

## Experiment

One exclusive Slurm allocation, sequential A then B on the same node. Each case
starts a new DuckDB process and runs Q18 three consecutive times on SF3000.
Uses `/home/wmalpica/repos/sirius_config2g_1.yaml`, two GPUs attached to NUMA 0,
`numactl --cpunodebind=0 --membind=0`, no pinning and cache mode `none`.
The original config is snapshotted once, never edited. Only the previously
approved unsupported `uring.max_inflight_small_reads` and `uring.small_read_bytes`
keys are removed from experiment copies; the telemetry output directory differs
between A and B. The original host-pool values are retained.

No Nsight profiling: `driver.py` reuses the benchmark's CLI SQL generation via
`--precmd nsys`, but removes the profiler launcher and SQL profiler calls. The
actual invocation is saved in `command.json`; generated directory names can
still contain `nsys`. Nsight itself is not used to collect data in this test.

The runner checks source cleanliness and parentage, updates submodules after
every checkout, builds the RelWithDebInfo Sirius library, rebuilds the release
DuckDB wrapper against it once, and explicitly preloads the case's library.
It samples loaded libraries and all thread affinities every second. This can
catch persistent misplacement; it does not prove every transient affinity or
physical page placement. Node-0 CPU numbering 0–71 is assumed for this GB200
cluster and checked; incompatible placements fail instead of silently comparing.
Builds execute on the allocated compute node. Building and running both cases
must fit the 55-minute timebox; a cold build might time out. Preserve its logs
and arrange a separate bounded build job if that happens.

## Fetch and submit (user-operated SSH)

First, on Presto 18 (or the head node, since this repo is shared):

```bash
cd /home/wmalpica/repos/sirius
git status --short
git fetch william codex/q18-io-queue-instrumentation codex/q18-io-queue-fifo
git switch --detach william/codex/q18-io-queue-instrumentation
git submodule update --init --recursive
```

If `william` does not exist in that clone, add it first:
`git remote add william git@github.com:wmalpica/sirius.git`.
Preserve any tracked edits before switching; do not reset them. Keep this
checkout unused by other builds/jobs until this experiment finishes. There is
no need to build manually: the job builds each case immediately before running.

Then on `wmalpica@presto-cp-02`:

```bash
source /etc/profile
module load slurm
squeue -u wmalpica
sinfo -p nvidia-part -N
```

Wait until your other jobs finish. Inspect available nodes; do not submit
concurrently with another benchmark. The following submits one serial A/B job
and pins both cases to immutable commit IDs resolved before submission:

```bash
cd /home/wmalpica/repos/sirius
experiment_root=/home/wmalpica/repos/logs/q18-io-fifo-ab-$(date -u +%Y%m%dT%H%M%SZ)
mkdir "$experiment_root"
a_commit=$(git rev-parse william/codex/q18-io-queue-instrumentation)
b_commit=$(git rev-parse william/codex/q18-io-queue-fifo)
job_id=$(sbatch --parsable \
  --output="$experiment_root/slurm-%j.log" \
  scripts/experiments/q18_io_fifo_ab/benchmark.sbatch \
  "$experiment_root" "$a_commit" "$b_commit")
printf '%s\n' "$job_id" | tee "$experiment_root/job-id.txt"
printf 'Results: %s\n' "$experiment_root"
squeue -j "$job_id"
```

The partition chooses a node; the placement guard requires two visible GPUs on
CPU NUMA 0. If choosing a specific available `presto-gb200-gcn-*` node, add
`--nodelist=<verified-node>` to `sbatch`. Do not assume node 18 is in this partition.
The run aborts if allocated GPUs do not satisfy the topology check. In that case,
inspect the allocation/topology records and adjust Slurm GPU placement before
resubmitting; do not remove the check or override GPU visibility blindly.

After completion, check status on the head node (no performance analysis there):

```bash
sacct -j "$job_id" --format=JobID,JobName,NodeList,State,ExitCode,Elapsed,AllocTRES%80
cat "$experiment_root/job-status.txt"
ls "$experiment_root/A" "$experiment_root/B"
```

Success requires `finished.txt`, zero job/step exit statuses, three finite query
timings per case, and no `AFFINITY_FAILURE` or `BUILD_FAILURE`. The harness can
return zero despite query errors; the runner separately validates `runtimes.csv`.
A failed build/query stops the pair and preserves logs. A wall-time kill may
prevent the shell exit trap and timeline dump. The repo is left detached at B;
do not reset it while a job is active. Use a new result directory for every retry.

## Collect and analyze locally

From this workstation, substitute the printed remote result directory:

```bash
rsync -av --partial wmalpica@presto-cp-02:/home/wmalpica/repos/logs/q18-io-fifo-ab-TIMESTAMP/ \
  /home/wmalpica/repos/logs/q18-io-fifo-ab-TIMESTAMP/
cd /home/wmalpica/PR_repos/PR0/sirius
pixi run python scripts/experiments/q18_io_fifo_ab/analyze-queue.py \
  /home/wmalpica/repos/logs/q18-io-fifo-ab-TIMESTAMP/A/preparation.csv \
  /home/wmalpica/repos/logs/q18-io-fifo-ab-TIMESTAMP/B/preparation.csv
```

Keep all artifacts: `quent/`, `gaps/`, `preparation.csv`, `benchmark/`, config,
source/submodule revisions, binary hashes, build logs, placement/affinity samples,
actual command and Slurm allocation/status records. Confirm transfer completeness
before deleting anything remotely. Label Quent query/iteration display metadata
locally while preserving original records.

`uring_executor` maps TID to GPU/executor. `uring_enqueue` records per-reactor
serialized enqueue sequence and bytes; `uring_dequeue` records dequeue sequence
and queue age in nanoseconds, retaining the original parent context. Request
shape records GPU+1 and slice count. Cancellation and failed enqueue events are
separate. Recorder drops must be zero for complete order reconstruction. Compare
per-executor ages, overtaking, scan durations, cumulative idle time and all three
query times, not just the fastest iteration. B should eliminate accepted-request
reordering within each reactor; if scan stalls remain, investigate downstream
staging/admission/group monopolization. Reordering is a candidate cause, not an
established root cause. A/B order and OS cache effects remain confounders; a B/A
repeat can follow if the first pair provides evidence.

## Local checks

`queue-policy-test.cpp` uses the checked-out DuckDB queue implementation.
Deterministic replenishment: original selects the latest registered producer
800/800 times, FIFO serves each of eight producers 100 times. A concurrent test
verifies 80,000 move-only requests preserve mutex-serialized enqueue order without
loss or duplication. This demonstrates a possible fairness failure, not that a
benchmark necessarily exhibits it. Earlier syntax checks passed for A and B
reactor code and executor instrumentation. Full Presto builds and A/B benchmarks
remain to be run; no performance result is claimed by these branches.

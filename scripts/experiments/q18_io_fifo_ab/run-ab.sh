#!/bin/bash
set -euo pipefail
root=${1:?Pass a new absolute output directory}
a_ref=${2:?Pass the A commit or ref}
b_ref=${3:?Pass the B commit or ref}
repo=/home/wmalpica/repos/sirius
cd "$repo"
export EXPERIMENT_ROOT="$root"
trap 'rc=$?; printf "exit=%s\nfinished_utc=%s\n" "$rc" "$(date -u -Is)" > "$root/job-status.txt"' EXIT
# Scripts have been staged outside the checkout by benchmark.sbatch.
# Do not discard any tracked changes, or change revisions during another job.
git diff --quiet --ignore-submodules=all HEAD -- || { echo 'Tracked checkout changes: stop and preserve them'; exit 1; }
a_ref=$(git rev-parse "$a_ref^{commit}")
b_ref=$(git rev-parse "$b_ref^{commit}")
test "$(git rev-parse "$b_ref^")" = "$a_ref"
printf 'A=%s\nB=%s\n' "$a_ref" "$b_ref" > "$root/refs.txt"
git diff "$a_ref" "$b_ref" > "$root/A-to-B.patch"
test -d /scratch/sirius/datasets/tpch_sf3000
python3 "$root/check-placement.py" > "$root/placement.txt"
{
 hostname; date -u -Is; git rev-parse HEAD; git status --short; git submodule status --recursive
 nvidia-smi; nvidia-smi topo -m; numactl --hardware
 env | sort | grep -E '^(SLURM|CUDA_VISIBLE)'
} > "$root/identity.txt" 2>&1
cp /home/wmalpica/repos/sirius_config2g_1.yaml "$root/original-config.yaml"
mkdir -p "$root/wrapper/extension/sirius"
ln -sf "$repo/sirius-duckdb/build/release/extension/sirius/sirius.duckdb_extension" "$root/wrapper/extension/sirius/sirius.duckdb_extension"
cat > "$root/wrapper/duckdb" <<'WRAPPER'
#!/bin/bash
set -e
printf '%s\n' "$$" > "$EXPERIMENT_ROOT/$EXPERIMENT_CASE/duckdb.pid"
export LD_PRELOAD=/home/wmalpica/repos/sirius/build/relwithdebinfo/libsirius.so
exec /home/wmalpica/repos/sirius/sirius-duckdb/build/release/duckdb "$@"
WRAPPER
chmod +x "$root/wrapper/duckdb"
export SIRIUS_BUILD_PATH="$root/wrapper"
for case in A B; do
 export EXPERIMENT_CASE="$case"
 mkdir -p "$root/$case/gaps"
 ref=$a_ref
 if [ "$case" = B ]; then ref=$b_ref; fi
 git switch --detach "$ref"
 git submodule update --init --recursive
 if git submodule status --recursive | grep -Eq '^[-+U]'; then echo 'Submodule mismatch'; exit 1; fi
 git rev-parse HEAD > "$root/$case/commit.txt"
 git status --short > "$root/$case/source-status.txt"
 git submodule status --recursive > "$root/$case/submodules.txt"
 export CMAKE_BUILD_PARALLEL_LEVEL=32
 /home/wmalpica/.pixi/bin/pixi run make relwithdebinfo TEST_BUILD_TARGET= > "$root/$case/build.log" 2>&1
 grep -q '^CMAKE_BUILD_TYPE:STRING=RelWithDebInfo$' build/relwithdebinfo/CMakeCache.txt
 # Rebuild the wrapper once against this source/ABI. B changes no wrapper API.
 if [ "$case" = A ]; then
  /home/wmalpica/.pixi/bin/pixi run cmake --install build/relwithdebinfo --prefix "$repo/build/relwithdebinfo/install" --component sirius_library >> "$root/$case/build.log" 2>&1
  /home/wmalpica/.pixi/bin/pixi run make -C sirius-duckdb release SIRIUS_DUCKDB_LINKAGE=shared SIRIUS_INSTALL_DIR="$repo/build/relwithdebinfo/install" >> "$root/$case/build.log" 2>&1
 fi
 sha256sum build/relwithdebinfo/libsirius.so sirius-duckdb/build/release/duckdb sirius-duckdb/build/release/extension/sirius/sirius.duckdb_extension > "$root/$case/binary.sha256"
 /home/wmalpica/.pixi/bin/pixi run python - <<'PY'
import os,yaml
from pathlib import Path
r=Path(os.environ['EXPERIMENT_ROOT']);case=os.environ['EXPERIMENT_CASE']
x=yaml.safe_load((r/'original-config.yaml').read_text())
assert x['sirius']['topology']['num_gpus']==2
assert x['sirius']['executor']['scan_manager']['cache']['mode']=='none'
u=x['sirius']['executor']['scan_manager'].get('uring',{})
for key in ('max_inflight_small_reads','small_read_bytes'): u.pop(key,None)
x['sirius'].setdefault('telemetry',{})['output_directory']=str(r/case/'quent')
(r/case/'config.yaml').write_text(yaml.safe_dump(x,sort_keys=False))
PY
 export SIRIUS_PREPARATION_PROBE=1 SIRIUS_PREPARATION_TIMELINE="$root/$case/preparation.csv" SIRIUS_GAP_TRACE_DIR="$root/$case/gaps" SIRIUS_LOG_LEVEL=trace
 unset SIRIUS_DIAGNOSTIC_PREPARATION_CPUS SIRIUS_PROBE_COALESCING_BYPASS SIRIUS_PREPARATION_CTRACK
 numactl --cpunodebind=0 --membind=0 numactl --show > "$root/$case/policy.txt"
 # Reuse the established CLI SQL harness, stripping the nsys launcher in driver.py.
 # This is NOT a profiled run: both cases use identical plain CLI execution.
 set +e
 numactl --cpunodebind=0 --membind=0 /home/wmalpica/.pixi/bin/pixi run python "$root/driver.py" --input /scratch/sirius/datasets/tpch_sf3000 --scale-factor 3000 --engine gpu --queries 18 --iterations 3 --pin none --config "$root/$case/config.yaml" --precmd nsys --query-timeout 1200 --output "$root/$case/benchmark" > "$root/$case/console.log" 2>&1
 rc=$?
 set -e
 printf 'runner_exit=%s\n' "$rc" > "$root/$case/status.txt"
 test "$rc" = 0
 test ! -e "$root/$case/AFFINITY_FAILURE"
 test ! -e "$root/$case/BUILD_FAILURE"
 test -s "$root/$case/preparation.csv"
 /home/wmalpica/.pixi/bin/pixi run python - <<'CHECK'
import csv, math, os
from pathlib import Path
r=Path(os.environ['EXPERIMENT_ROOT'])/os.environ['EXPERIMENT_CASE']
files=list((r/'benchmark').rglob('runtimes.csv'))
assert len(files)==1, f'Expected one runtimes.csv, found {files}'
rows=[x for x in csv.DictReader(files[0].open()) if x['query']=='q18' and x['engine']=='sirius']
assert len(rows)==3 and {int(x['iteration']) for x in rows}=={0,1,2}, rows
assert all(math.isfinite(float(x['runtime_s'])) and float(x['runtime_s'])>0 for x in rows), rows
(r/'validated-times.txt').write_text(''.join(f"iteration {int(x['iteration'])+1}: {x['runtime_s']} s\n" for x in rows))
CHECK
done
printf 'finished_utc=%s\n' "$(date -u -Is)" > "$root/finished.txt"

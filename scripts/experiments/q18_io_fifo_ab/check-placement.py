import os
import subprocess
from pathlib import Path

print('CPU affinity:', sorted(os.sched_getaffinity(0)), flush=True)
print(Path('/proc/self/cgroup').read_text(), flush=True)
for line in Path('/proc/self/cgroup').read_text().splitlines():
    _, controllers, name = line.split(':', 2)
    if controllers and 'cpuset' not in controllers.split(','):
        continue
    base = Path('/sys/fs/cgroup') / ('cpuset' if controllers else '')
    current = base / name.lstrip('/')
    while current == base or base in current.parents:
        for key in ('cpuset.cpus.effective', 'cpuset.effective_cpus', 'cpuset.cpus', 'cpuset.mems.effective', 'cpuset.mems'):
            p = current / key
            if p.exists():
                print(str(p), p.read_text().strip(), flush=True)
        if current == base:
            break
        current = current.parent
# The step's cgroup must constrain the initially unbound task, rather than
# relying solely on a numactl affinity mask that engine workers can widen.
assert set(range(72)) <= os.sched_getaffinity(0), 'CPU NUMA node 0 must be available'
print('Slurm does not enforce a restricted cpuset; numactl affinity will be checked during execution.', flush=True)

topo = subprocess.check_output(["nvidia-smi", "topo", "-m"], text=True)
print(topo, flush=True)
gpus = [line.split() for line in topo.splitlines() if line.startswith("GPU")]
assert len(gpus) == 2 and all(row[-2] == "0" for row in gpus), "Expected exactly two GPUs attached to CPU NUMA 0"
print(subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,pci.bus_id", "--format=csv"], text=True), flush=True)

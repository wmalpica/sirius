import os,sys,subprocess,json,threading,time
from pathlib import Path
root=Path(os.environ['EXPERIMENT_ROOT']);case=os.environ['EXPERIMENT_CASE'];out=root/case
sys.path.insert(0,'/home/wmalpica/repos/sirius/test/tpch_performance')
import performance_test as bench
original=bench.subprocess.run

def wrapped(args,*a,**kw):
 if not isinstance(args,list) or args[:2]!=['nsys','profile']:return original(args,*a,**kw)
 target=args.index(bench.DUCKDB_BIN)
 if case in ('A','B'):
  sql=Path(args[-1]);sql.write_text(sql.read_text().replace('CALL profiler_start();','').replace('CALL profiler_stop();',''))
  args=args[target:]
 (out/'command.json').write_text(json.dumps(args,indent=2))
 return original(args,*a,**kw)
bench.subprocess.run=wrapped
stop=threading.Event()
def monitor():
 with (out/'affinity.ndjson').open('w') as f:
  while not stop.wait(1):
   pidfile=out/'duckdb.pid'
   if not pidfile.exists():continue
   pid=int(pidfile.read_text());base=Path(f'/proc/{pid}')
   try:
    maps=(base/'maps').read_text()
    if 'libsirius.so' in maps:
     (out/'loaded-libraries.txt').write_text(maps)
     assert '/build/relwithdebinfo/libsirius.so' in maps,'Wrong Sirius library'
    rows=[]
    for t in (base/'task').iterdir():
     try:rows.append({'tid':int(t.name),'name':(t/'comm').read_text().strip(),'cpus':sorted(os.sched_getaffinity(int(t.name)))})
     except (FileNotFoundError,ProcessLookupError):pass
    f.write(json.dumps({'epoch_ns':time.time_ns(),'threads':rows})+'\n');f.flush()
    if any(set(t['cpus'])-set(range(72)) for t in rows):
     (out/'AFFINITY_FAILURE').write_text('Thread mask outside node0');os.kill(pid,15);return
   except FileNotFoundError:pass
   except AssertionError as e:
    (out/'BUILD_FAILURE').write_text(str(e));os.kill(pid,15);return
thread=threading.Thread(target=monitor,daemon=True);thread.start()
try:bench.main()
finally:stop.set();thread.join()

"""Analyze local preparation.csv files; no cluster-side analysis required."""
import argparse, collections, csv, heapq, json, statistics
from pathlib import Path

def analyze(path):
    labels = {}; enqueue = {}; dequeues = []; failed = set(); dropped = None
    with path.open() as f:
        rows = csv.DictReader(line for line in f if not line.startswith('#'))
        for row in rows:
            event = row['event']
            if event == 'uring_executor':
                labels[int(row['tid'])] = f"gpu_pipeline-gpu{row['a']}-exec-{row['b']}"
            elif event.startswith('uring_'):
                row = {k: (v if k == 'event' else int(v)) for k,v in row.items()}
                key = row['owner'], row['id']
                if event == 'uring_enqueue': enqueue[key] = row
                elif event in ('uring_dequeue','uring_cancel_dequeue'): dequeues.append(row)
                elif event == 'uring_enqueue_failed': failed.add(key)
    with path.open() as f:
        for line in f:
            if line.startswith('# dropped='): dropped = int(line.split('=')[1])
    per_reactor = collections.defaultdict(list)
    for d in dequeues: per_reactor[d['owner']].append(d)
    groups = collections.defaultdict(list); worst=[]; reorder_count=0; missing=0; canceled=0
    for owner, rows in per_reactor.items():
        pending = [seq for r,seq in enqueue if r == owner and (r,seq) not in failed]
        heapq.heapify(pending); removed=set()
        for d in sorted(rows,key=lambda d:d['a']):
            seq=d['id']; e=enqueue.get((owner,seq))
            if not e: missing+=1; continue
            while pending and pending[0] in removed: heapq.heappop(pending)
            oldest=pending[0] if pending else None
            bypass=oldest is not None and oldest<seq
            reorder_count+=int(bypass); removed.add(seq)
            canceled+=int(d['event']=='uring_cancel_dequeue')
            sample=dict(reactor=owner,enqueue_sequence=seq,dequeue_sequence=d['a'],
                        executor=labels.get(e['tid'],f"tid-{e['tid']}"),submitter_tid=e['tid'],
                        queue_age_s=d['b']/1e9,bytes=e['b'],parent_span=e['parent'],
                        bypassed_oldest_sequence=oldest if bypass else None,
                        enqueued_ns=e['ns'],dequeued_ns=d['ns'],canceled=d['event']=='uring_cancel_dequeue')
            groups[sample['executor']].append(sample);worst.append(sample)
    summaries=[]
    for name, values in sorted(groups.items()):
        ages=sorted(v['queue_age_s'] for v in values)
        summaries.append(dict(executor=name,count=len(ages),median_s=statistics.median(ages),
             p95_s=ages[min(len(ages)-1,int(len(ages)*.95))],max_s=max(ages),
             bypassing_requests=sum(v['bypassed_oldest_sequence'] is not None for v in values)))
    unresolved = len(enqueue) - len(failed) - (len(dequeues) - missing)
    result=dict(file=str(path),dropped=dropped,complete=(dropped==0 and missing==0 and unresolved==0),
                unresolved_requests=unresolved,
                enqueues=len(enqueue),dequeues=len(dequeues),failed=len(failed),canceled=canceled,
                missing_enqueue_records=missing,requests_passing_older_requests=reorder_count,
                executors=summaries,longest_waits=sorted(worst,key=lambda x:x['queue_age_s'],reverse=True)[:100])
    path.with_name('queue-analysis.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='longest_waits'},indent=2))
    return result
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('timeline',nargs='+',type=Path)
    for file in parser.parse_args().timeline: analyze(file)

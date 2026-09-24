#!/usr/bin/env python3
"""Option B probe: m=0.3 abstention + PROCESS-prior tie-break among
validation-mean-tied top masks. Goal: 0/54 run harm (fix WS12-b r2 fold)."""
import sys, os, csv, pickle
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import subgraph_rescoring as sr
CACHE = os.path.join(os.path.dirname(HERE), 'analysis', 'cache_rescore')
OUT = os.path.join(os.path.dirname(HERE), 'analysis', 'safe_tiebreak_probe.csv')
HOSTS = ['SimulatedUbuntu', 'SimulatedW10', 'SimulatedWS12',
         'SysClient0051', 'SysClient0201', 'SysClient0501',
         'cadets', 'theia', 'trace']
M = 0.3
EPS = 1e-6

def select(rd, val_runs, prefer_process):
    val_types = sorted({t for d in val_runs.values() for t in d['types']})
    def vmean(mask, tau):
        return float(np.mean([sr.f1_metric(d, mask, tau) for d in val_runs.values()]))
    def_f1 = vmean(val_types, 10.0)
    apply_def = sr.f1_metric(rd, rd['types'], 10.0)
    best_mean = -1.0
    ties = []  # (mean, mask, tau)
    for mask in sr.masks_for(val_types):
        cand = sorted({sr.score_of(d, i, mask) for d in val_runs.values()
                       for i in range(len(d['subgraphs']))})
        bm = -1.0; bt = None
        for tau in cand + ([cand[-1] + 1] if cand else [1.0]):
            mm = vmean(mask, tau)
            if mm > bm + 1e-12:
                bm, bt = mm, tau
        if bm > best_mean + EPS:
            best_mean, ties = bm, [(bm, mask, bt)]
        elif abs(bm - best_mean) <= EPS:
            ties.append((bm, mask, bt))
    if best_mean <= def_f1 + M:
        return apply_def, 'DEFAULT', True
    # tie-break: prefer PROCESS-only if among ties; else first found
    chosen = ties[0]
    if prefer_process:
        proc = [t for t in ties if t[1] == ('PROCESS',)]
        if proc:
            chosen = proc[0]
    return sr.f1_metric(rd, chosen[1], chosen[2]), \
           f"mask:{','.join(sorted(chosen[1]))}@{chosen[2]}", False

def main():
    rows_out = []
    for host in HOSTS:
        for variant in ['fullz', 'TGN_fullz']:
            rds = {}
            for run in ['0', '1', '2']:
                p = os.path.join(CACHE, f'prep_{host}_{variant}_run{run}.pkl')
                if os.path.exists(p):
                    with open(p, 'rb') as f:
                        rds[run] = pickle.load(f)
            if len(rds) < 3:
                continue
            for run, rd in rds.items():
                val_runs = {r: d for r, d in rds.items() if r != run}
                for tag, pref in [('m0.3', False), ('m0.3_proc', True)]:
                    f1, chosen, abst = select(rd, val_runs, pref)
                    rows_out.append(dict(host=host, variant=variant, run=run, crit=tag,
                                         real_F1=round(f1, 4),
                                         default_F1=round(rd['default_f1'], 4),
                                         chosen=chosen, abstained=int(abst)))
            print(f'{host} {variant} done', flush=True)
    with open(OUT, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['host', 'variant', 'run', 'crit', 'real_F1',
                                          'default_F1', 'chosen', 'abstained'])
        w.writeheader()
        for r in rows_out:
            w.writerow(r)
    print('wrote', OUT, len(rows_out))

if __name__ == '__main__':
    main()

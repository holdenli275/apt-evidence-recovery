#!/usr/bin/env python3
"""Persist per-run prepared data (subgraph aggregates + closure sets) to disk.

Pickles produced by subgraph_rescoring.prep_run for (host,variant,run) so that
policy sweeps over the S_theta family become seconds, not minutes.
"""
import sys, os, pickle
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import subgraph_rescoring as sr

CACHE = os.path.join(os.path.dirname(HERE), 'analysis', 'cache_rescore')
VARIANTS = ['fullz', 'TGN_fullz']
hosts_by_ds = {'nodlink': ['SimulatedUbuntu', 'SimulatedW10', 'SimulatedWS12'],
               'darpa_optc': ['SysClient0051', 'SysClient0201', 'SysClient0501'],
               'darpa_tc3': ['cadets', 'theia', 'trace']}

def main():
    only = set(sys.argv[1:]) if len(sys.argv) > 1 else None
    for ds, hosts in hosts_by_ds.items():
        for host in hosts:
            if only and host not in only:
                continue
            emap = sr.build_eval_maps(host)
            root = os.path.join(os.path.dirname(HERE), 'dataset', ds, host, 'experiments')
            for variant in VARIANTS:
                mdir = os.path.join(root, sr.MODEL_DIR[variant](host))
                summary_path = os.path.join(mdir, 'subgraph_anomaly_results_summary.csv')
                if not os.path.exists(summary_path):
                    continue
                summ = sr.summary_rows(summary_path)
                for run in ['0', '1', '2']:
                    outp = os.path.join(CACHE, f'prep_{host}_{variant}_run{run}.pkl')
                    if os.path.exists(outp):
                        continue
                    if run not in summ:
                        continue
                    rd = sr.prep_run(host, variant, run, emap, mdir)
                    if rd is None:
                        continue
                    rd['types'] = sorted({t for s in rd['subgraphs'] for t in s['tsum']})
                    rd['default_f1'] = sr.f1_metric(rd, rd['types'], 10.0)
                    ref_f1 = float(summ[run].get('f_measure', -1))
                    rd['reproduced'] = abs(rd['default_f1'] - ref_f1) < 1e-6
                    rd['host'] = host; rd['variant'] = variant; rd['run'] = run
                    with open(outp, 'wb') as f:
                        pickle.dump(rd, f, protocol=4)
                    print(f'cached {host} {variant} run{run} (def={rd["default_f1"]:.4f} ok={rd["reproduced"]})',
                          flush=True)
    print('done')

if __name__ == '__main__':
    main()

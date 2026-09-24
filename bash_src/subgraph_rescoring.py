#!/usr/bin/env python3
"""Real-subgraph offline rescoring ablation (S_theta family over cached subgraphs).

For every (host, variant, run) where the real detection pipeline cached
`*_constructed_subgraphs_nx.pt`, we re-score the SAME constructed subgraphs
with candidate keep rules and evaluate with the EXACT real evaluation
(reproduced offline):

  default_tau10   S(C)=sum p_a over alarms in C, keep S>10  (paper default)
  minor_tau1 / significant_tau100 / critical_tau1000        (severity tiers)
  mask_<TYPE>_tauX   type-mask scores (PROCESS/FILE/NET/MEMORY/OTHER)
  oracle_prefix      per-run best prefix on S (6.1 upper bound)
  loocv_tau          tau chosen on the other runs (6.1)
  loocv_mask         binary type mask + tau chosen on other runs (6.2)
  loocv_lambda       + single-base-type penalty lambda (6.2)
  outlier_grubbs/zscore/iqr, topk<K>   self-contained selectors (6.4)

The real subgraph F1 uses 2-hop closure over the full heterogeneous graph
(re_evaluate_anomaly_detection). We reproduce it exactly and verify each run
against its stored subgraph_anomaly_results_summary.csv default row; runs
whose default cannot be reproduced are skipped (recorded).
"""
import sys, os, csv, argparse, pickle, collections, glob
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'src'))
from dataset_pyg_custom import PygNodePropPredDataset_custom

OUT_CSV = os.path.join(os.path.dirname(HERE), 'analysis', 'subgraph_rescoring.csv')
OUT_MEANS = os.path.join(os.path.dirname(HERE), 'analysis', 'subgraph_rescoring_means.csv')
EVAL_CACHE = os.path.join(os.path.dirname(HERE), 'analysis', 'cache_rescore', 'eval_{host}.pkl')

SEG_ALIAS = {
    'net': ['net', 'netflowobject', 'flow'],
    'process': ['process', 'subject_process', 'proc'],
    'file': ['file', 'file_object', 'file_object_file', 'file_object_dir',
             'file_object_link', 'file_object_char', 'file_object_unix_socket'],
    'memory': ['memory', 'memoryobject'],
}
BASE_LABEL = {'net': 'NET', 'process': 'PROCESS', 'file': 'FILE', 'memory': 'MEMORY'}

def base_type(ntype):
    seg = str(ntype).split('_')[-1].lower()
    full = str(ntype).lower()
    for base, aliases in SEG_ALIAS.items():
        if seg in aliases or full in aliases:
            return BASE_LABEL[base]
    return 'OTHER'

HOST_DIR = {'SimulatedUbuntu': 'nodlink', 'SimulatedW10': 'nodlink', 'SimulatedWS12': 'nodlink',
            'SysClient0051': 'darpa_optc', 'SysClient0201': 'darpa_optc', 'SysClient0501': 'darpa_optc',
            'cadets': 'darpa_tc3', 'theia': 'darpa_tc3', 'trace': 'darpa_tc3'}

MODEL_DIR = {
    'fullz': lambda h: f'results/Full_Script_Test_fullz/T3_OCRGCN_{h}_fullz',
    'TGN_fullz': lambda h: f'results/Full_Script_Test_TGN_fullz/T3_OCRGCN_{h}_TGN_fullz',
    'released': lambda h: ('results/Full_Script_Test/'
                           'OCRGCN_Dr0_ly3_bs0_ep100_beta0.5_LR0.005_Hly32_dynConVal0.001To0.05'),
}
NX_PATTERNS = [
    'run{r}_expand_1_hop_MaxEdges5000_K15_ly3_constructed_subgraphs_nx.pt',
    'run{r}_expand_1_hop_MaxEdges5000_K15_constructed_subgraphs_nx.pt',
]

# ---------------- eval-graph (replicates get_edges_mapping_of_full_graph) ----
def build_eval_maps(host):
    cp = EVAL_CACHE.format(host=host)
    if os.path.exists(cp):
        with open(cp, 'rb') as f:
            return pickle.load(f)
    dsdir = HOST_DIR[host]
    root = os.path.join(os.path.dirname(HERE), 'dataset', dsdir, host, 'experiments') + os.sep
    ds = PygNodePropPredDataset_custom(name='Full_Script_Test_TGN_fullz', root=root,
                                       numofClasses=2)
    data = ds[0]
    split = ds.get_idx_split('node_type')
    cache = ds.root
    info = {}
    for t in data.y_dict.keys():
        n = int(data.num_nodes_dict[t])
        y = data.y_dict[t].flatten().numpy()
        test = np.asarray(split['test'][t])
        tm = np.zeros(n, dtype=bool); tm[test] = True
        uuids = [None] * n
        with open(os.path.join(cache, 'mapping', f'{t}_entidx2name.csv')) as f:
            rd = csv.reader(f); next(rd)
            for row in rd:
                uuids[int(row[0])] = row[1].strip().lower()
        info[t] = dict(n=n, y=y, test_mask=tm, uuids=uuids)
    uuid2loc = {}
    for t, d in info.items():
        for i, u in enumerate(d['uuids']):
            if u is not None:
                uuid2loc[u] = (t, i)
    type_keys = [t for t in info if 'test_mask' in info[t]]
    M = set(); test_set = set()
    for t in type_keys:
        d = info[t]
        for i in np.flatnonzero(d['test_mask'] & (d['y'] == 1)):
            u = d['uuids'][i]
            if u is not None:
                M.add(u)
        for i in np.flatnonzero(d['test_mask']):
            u = d['uuids'][i]
            if u is not None:
                test_set.add(u)
    out = dict(info={t: d for t, d in info.items() if t in type_keys},
               uuid2loc=uuid2loc, M=M, test_set=test_set,
               edges=[(k[0], k[2], v) for k, v in data.edge_index_dict.items()])
    with open(cp, 'wb') as f:
        pickle.dump(out, f, protocol=4)
    return out

def two_hop(uuids, emap):
    """uuids within <=2 hops of set uuids over the full graph (incl. starts)."""
    cur = collections.defaultdict(set)
    for u in uuids:
        t, idx = emap['uuid2loc'][u]
        cur[t].add(idx)
    edges = emap['edges']
    for _ in range(2):
        nxt = {t: set(v) for t, v in cur.items()}
        for st, ot, ei in edges:
            row = ei[0].numpy(); col = ei[1].numpy()
            if st in cur and cur[st]:
                arr = np.fromiter(cur[st], dtype=np.int64, count=len(cur[st]))
                sel = np.isin(row, arr)
                if sel.any():
                    nxt.setdefault(ot, set()).update(int(c) for c in col[sel])
            if ot in cur and cur[ot]:
                arr = np.fromiter(cur[ot], dtype=np.int64, count=len(cur[ot]))
                sel = np.isin(col, arr)
                if sel.any():
                    nxt.setdefault(st, set()).update(int(r) for r in row[sel])
        cur = nxt
    out = set()
    for t, idxs in cur.items():
        d = emap['info'][t]
        uu = d['uuids']
        out.update(uu[i] for i in idxs if i < len(uu) and uu[i] is not None)
    return out

# ---------------- per-run precomputed subgraph data --------------------------
def load_alarms(path):
    out = {}
    with open(path) as f:
        r = csv.reader(f); h = next(r)
        iu, ip, it = h.index('node_uuid'), h.index('Prediction_probability'), h.index('node_type')
        for row in r:
            out[row[iu].lower()] = (float(row[ip]), base_type(row[it]))
    return out

def summary_rows(path):
    rows = list(csv.reader(open(path)))
    h = rows[0]
    return {r[0]: dict(zip(h[1:], r[1:])) for r in rows[1:] if r and r[0] not in ('avg', 'std')}

def prep_run(host, variant, run, emap, mdir):
    """-> dict with per-subgraph aggregates + precomputed closure sets."""
    al = load_alarms(os.path.join(mdir, f'run_{run}_raised_alarms.csv'))
    sg_path = None
    for pat in NX_PATTERNS:
        p = os.path.join(mdir, pat.format(r=run))
        if os.path.exists(p):
            sg_path = p
            break
    if sg_path is None:
        return None
    sgs = torch.load(sg_path, map_location='cpu', weights_only=False)
    M = emap['M']; test = emap['test_set']
    stats_path = sg_path.replace('_constructed_subgraphs_nx.pt',
                                 '_correlated_subgraphs_statistics.csv')
    stat_scores = None
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            stat_scores = [float(r['subgraph_anomaly_score'])
                           for r in csv.DictReader(f)]
    subs = []
    for sg in sgs:
        nodes = [u.lower() for u in sg.nodes()]
        A_s = set(nodes) & test
        mal_s = A_s & M
        tsum = collections.Counter()
        for u in nodes:
            if u in al:
                tsum[al[u][1]] += al[u][0]
        bases = list(tsum.keys())
        subs.append(dict(A=A_s, mal=mal_s, tsum=dict(tsum),
                         single=len(bases) == 1, n_alarm=sum(1 for u in nodes if u in al)))
    # rescue: 2-hop of malicious alarms in subgraph, intersected with M
    for s in subs:
        s['rescue'] = two_hop(s['mal'], emap) & M if s['mal'] else set()
    g2 = two_hop(M, emap) & test
    return dict(subgraphs=subs, g2=g2, stat_scores=stat_scores, alarm_rows=al,
                n_mal=len(M), test=test, M=M)

def default_score_all(s):
    return sum(s['tsum'].values())

def eval_selection(run_data, kept):
    """kept: iterable of subgraph indices; returns metric dict (exact real eval)."""
    M = run_data['M']; test = run_data['test']
    MP = set(); TP = set(); rescue = set()
    for i in kept:
        s = run_data['subgraphs'][i]
        MP |= s['A']
        TP |= s['mal']
        rescue |= s['rescue']
    FP_raw = MP - M
    FPL = FP_raw - run_data['g2']
    FN = M - TP
    TPL = TP | (FN & rescue)
    FN2 = FN - rescue
    tp, fp, fn = len(TPL), len(FPL), len(FN2)
    tn = len(test) - len(M) - fp
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, prec=prec, rec=rec, f1=f1)

def score_of(run_data, i, mask, lam=0.0):
    s = run_data['subgraphs'][i]
    if mask is None:
        sc = sum(s['tsum'].values())
    else:
        sc = 0.0
        for m in mask:
            sc += s['tsum'].get(m, 0.0)
    if lam and s['single']:
        sc -= lam
    return sc

def kept_by_tau(run_data, mask, tau, lam=0.0):
    return [i for i in range(len(run_data['subgraphs']))
            if score_of(run_data, i, mask, lam) > tau]

# ---------------- rules -------------------------------------------------------
def f1_metric(run_data, mask, tau, lam=0.0):
    kept = kept_by_tau(run_data, mask, tau, lam)
    return eval_selection(run_data, kept)['f1']

def oracle_f1(run_data, mask):
    scs = sorted(((score_of(run_data, i, mask), i) for i in range(len(run_data['subgraphs']))),
                 reverse=True)
    best = 0.0
    kept = []
    for sc, i in scs:
        kept.append(i)
        best = max(best, eval_selection(run_data, kept)['f1'])
    return best

def topk_f1(run_data, k):
    scs = sorted(((score_of(run_data, i, None), i) for i in range(len(run_data['subgraphs']))),
                 reverse=True)
    kept = [i for _, i in scs[:k]]
    return eval_selection(run_data, kept)['f1']

def outlier_kept(run_data, method):
    vals = np.array([score_of(run_data, i, None) for i in range(len(run_data['subgraphs']))])
    n = len(vals)
    if method == 'grubbs':
        try:
            from outliers import smirnov_grubbs as grubbs
            ol = grubbs.max_test_outliers(vals.tolist(), alpha=0.05)
        except Exception:
            ol = []
        if not ol or n < 3:
            return []
        thr = min(ol)
        return [i for i, v in enumerate(vals) if v >= thr - 1e-9]
    if method == 'zscore':
        sd = float(vals.std())
        if n < 2 or sd == 0 or not np.isfinite(sd):
            return []
        z = (vals - vals.mean()) / sd
        return [i for i, v in enumerate(z) if v > 2.0]
    if method == 'iqr':
        if n < 2:
            return []
        q1, q3 = np.percentile(vals, [25, 75])
        thr = q3 + 1.5 * (q3 - q1)
        return [i for i, v in enumerate(vals) if v >= thr - 1e-9]
    raise ValueError(method)

def masks_for(types):
    n = len(types)
    return [tuple(types[i] for i in range(n) if (b >> i) & 1) for b in range(1, 1 << n)]

def all_masks_union(host_runs, mask):  # place-holder kept minimal
    return mask

# ---------------- main ---------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=['nodlink', 'darpa_optc', 'darpa_tc3'])
    ap.add_argument('--hosts', nargs='+', default=None)
    ap.add_argument('--variants', nargs='+', default=['fullz', 'TGN_fullz', 'released'])
    args = ap.parse_args()

    hosts_by_ds = {'nodlink': ['SimulatedUbuntu', 'SimulatedW10', 'SimulatedWS12'],
                   'darpa_optc': ['SysClient0051', 'SysClient0201', 'SysClient0501'],
                   'darpa_tc3': ['cadets', 'theia', 'trace']}
    all_rows = []
    skipped = []

    for ds in args.datasets:
        for host in hosts_by_ds[ds]:
            if args.hosts and host not in args.hosts:
                continue
            emap = build_eval_maps(host)
            print(f'## {host}: test={len(emap["test_set"])} M={len(emap["M"])} '
                  f'types={sorted(emap["info"])}', flush=True)
            root = os.path.join(os.path.dirname(HERE), 'dataset', ds, host, 'experiments')
            for variant in args.variants:
                mdir = os.path.join(root, MODEL_DIR[variant](host))
                summary_path = os.path.join(mdir, 'subgraph_anomaly_results_summary.csv')
                if not os.path.exists(summary_path):
                    continue
                summ = summary_rows(summary_path)
                runs = ['0', '1', '2'] if variant != 'released' else ['0']
                host_run_data = {}
                for run in runs:
                    if run not in summ:
                        continue
                    rd = prep_run(host, variant, run, emap, mdir)
                    if rd is None:
                        skipped.append((host, variant, run, 'no nx cache'))
                        continue
                    # default mask = all base types present in the run
                    all_types = sorted({t for s in rd['subgraphs'] for t in s['tsum']})
                    kept_def = kept_by_tau(rd, all_types, 10.0)
                    met = eval_selection(rd, kept_def)
                    ref = {k: float(summ[run][k]) for k in ('tp', 'fp', 'fn', 'f_measure')
                           if k in summ[run]}
                    ok = (abs(met['tp'] - ref.get('tp', -1)) < 1e-6 and
                          abs(met['f1'] - ref.get('f_measure', -1)) < 1e-6)
                    rd['types'] = all_types
                    rd['summary_ref'] = ref
                    rd['reproduced'] = ok
                    host_run_data[run] = rd
                    print(f'   {variant} run{run}: n_sub={len(rd["subgraphs"])} '
                          f'default F1={met["f1"]:.4f} ref={ref.get("f_measure", float("nan")):.4f} '
                          f'ok={ok}', flush=True)
                    if not ok:
                        skipped.append((host, variant, run, 'default mismatch'))
                # fixed per-run rules
                for run, rd in host_run_data.items():
                    base_rows = []
                    all_types = rd['types']
                    def add(rule, f1, **kw):
                        all_rows.append(dict(host=host, variant=variant, run=run,
                                             rule=rule, emu_F1=None, real_F1=round(f1, 4),
                                             reproduced=rd['reproduced'], **kw))
                    for tau, name in ((10.0, 'default_tau10'), (1.0, 'minor_tau1'),
                                      (100.0, 'significant_tau100'), (1000.0, 'critical_tau1000')):
                        add(name, f1_metric(rd, all_types, tau))
                    for t in all_types:
                        add(f'mask_{t}_tau10', f1_metric(rd, (t,), 10.0))
                    add('mask_PROCESS_tau1', f1_metric(rd, ('PROCESS',), 1.0))
                    add('oracle_prefix', oracle_f1(rd, all_types))
                    for k in (5, 10, 15, 20):
                        add(f'topk{k}', topk_f1(rd, k))
                    for m in ('grubbs', 'zscore', 'iqr'):
                        kept = outlier_kept(rd, m)
                        add(f'outlier_{m}', eval_selection(rd, kept)['f1'])
                # LOOCV rules (need >=3 runs)
                if len(host_run_data) >= 3:
                    for run, rd in host_run_data.items():
                        val_runs = {r: d for r, d in host_run_data.items() if r != run}
                        val_types = sorted({t for d in val_runs.values() for t in d['types']})
                        # loocv tau on all-types
                        cand = sorted({score_of(d, i, val_types)
                                       for d in val_runs.values()
                                       for i in range(len(d['subgraphs']))})
                        best = (-1.0, None)
                        for tau in cand + [cand[-1] + 1]:
                            mean = float(np.mean([f1_metric(d, val_types, tau)
                                                  for d in val_runs.values()]))
                            if mean > best[0] + 1e-12:
                                best = (mean, tau)
                        add('loocv_tau', f1_metric(rd, val_types, best[1]), chosen_tau=best[1])
                        # loocv mask
                        best2 = (-1.0, None, None)
                        for mask in masks_for(val_types):
                            cand2 = sorted({score_of(d, i, mask)
                                            for d in val_runs.values()
                                            for i in range(len(d['subgraphs']))})
                            for tau in cand2 + [cand2[-1] + 1 if cand2 else 1.0]:
                                mean = float(np.mean([f1_metric(d, mask, tau)
                                                      for d in val_runs.values()]))
                                if mean > best2[0] + 1e-12:
                                    best2 = (mean, mask, tau)
                        mask_star, tau_star = best2[1], best2[2]
                        add('loocv_mask', f1_metric(rd, mask_star, tau_star),
                            chosen_mask=','.join(sorted(mask_star)), chosen_tau=tau_star)
                        # loocv lambda on best mask
                        best3 = (-1.0, None, None)
                        for lam in (0.0, 5.0, 10.0, 20.0, 50.0):
                            cand3 = sorted({score_of(d, i, mask_star, lam)
                                            for d in val_runs.values()
                                            for i in range(len(d['subgraphs']))})
                            for tau in cand3 + [cand3[-1] + 1 if cand3 else 1.0]:
                                mean = float(np.mean([f1_metric(d, mask_star, tau, lam)
                                                      for d in val_runs.values()]))
                                if mean > best3[0] + 1e-12:
                                    best3 = (mean, lam, tau)
                        lam_star, tau_lam = best3[1], best3[2]
                        add('loocv_lambda', f1_metric(rd, mask_star, tau_lam, lam_star),
                            chosen_mask=','.join(sorted(mask_star)),
                            chosen_lambda=lam_star, chosen_tau=tau_lam)

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    fields = ['host', 'variant', 'run', 'rule', 'real_F1', 'reproduced']
    for extra in ('chosen_mask', 'chosen_tau', 'chosen_lambda'):
        if any(extra in r for r in all_rows):
            fields.append(extra)
    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for r in all_rows:
            w.writerow(r)
    agg = collections.defaultdict(list)
    for r in all_rows:
        if r['reproduced']:
            agg[(r['host'], r['variant'], r['rule'])].append(r['real_F1'])
    with open(OUT_MEANS, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['host', 'variant', 'rule', 'n', 'mean_real_F1', 'std', 'min', 'max'])
        for (host, variant, rule), vals in sorted(agg.items()):
            v = np.array(vals)
            w.writerow([host, variant, rule, len(v), round(float(v.mean()), 4),
                        round(float(v.std()), 4), round(float(v.min()), 4),
                        round(float(v.max()), 4)])
    print('wrote', OUT_CSV, len(all_rows), 'rows; skipped:', skipped)

if __name__ == '__main__':
    main()

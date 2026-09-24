#!/usr/bin/env python3
"""Paired significance of FUSED vs STATIC (subgraph 3-run, node 10-run).

Subgraph: 27 (host, run) pairs at the default rule from analysis/subgraph_rescoring.csv.
Node    : 9 hosts x 10 runs from per-host anomaly_results_summary.csv (f_measure).

Reports Wilcoxon signed-rank, sign test, paired t-test and bootstrap CIs
(run-level and host-cluster) for the paired mean difference.
"""
import csv, os, sys, json, collections
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT_CSV = os.path.join(HERE, 'paired_significance.csv')
OUT_MD = os.path.join(HERE, 'paired_significance.md')

HOST_DIR = {'SimulatedUbuntu': 'nodlink', 'SimulatedW10': 'nodlink', 'SimulatedWS12': 'nodlink',
            'SysClient0051': 'darpa_optc', 'SysClient0201': 'darpa_optc', 'SysClient0501': 'darpa_optc',
            'cadets': 'darpa_tc3', 'theia': 'darpa_tc3', 'trace': 'darpa_tc3'}
NODE_PATH = {
    'STATIC': lambda h: os.path.join(ROOT, 'dataset', HOST_DIR[h], h, 'experiments', 'results',
                                     'Full_Script_Test_fullz', f'T3_OCRGCN_{h}_fullz',
                                     'anomaly_results_summary.csv'),
    'FUSED': lambda h: os.path.join(ROOT, 'dataset', HOST_DIR[h], h, 'experiments', 'results',
                                    'Full_Script_Test_TGN_fullz', f'T3_OCRGCN_{h}_TGN_fullz',
                                    'anomaly_results_summary.csv'),
}
RNG = np.random.default_rng(20260911)
NBOOT = 20000


def load_subgraph_pairs():
    rows = list(csv.DictReader(open(os.path.join(HERE, 'subgraph_rescoring.csv'))))
    d = collections.defaultdict(dict)
    for r in rows:
        if r['rule'] != 'default_tau10':
            continue
        if r['variant'] in ('fullz', 'TGN_fullz', 'released'):
            d[(r['host'], int(r['run']))][r['variant']] = float(r['real_F1'])
    hosts = sorted(set(h for h, _ in d))
    out = {h: {'STATIC': [], 'FUSED': [], 'released': []} for h in hosts}
    for (h, run) in sorted(d):
        for k_src, k_dst in (('fullz', 'STATIC'), ('TGN_fullz', 'FUSED'), ('released', 'released')):
            if k_src in d[(h, run)]:
                out[h][k_dst].append((run, d[(h, run)][k_src]))
    return out


def load_node_pairs(n_runs=3):
    """Node-level pairs for the first n_runs seeds (3 = matched to the subgraph level)."""
    out = {}
    for h in HOST_DIR:
        vals = {'STATIC': {}, 'FUSED': {}}
        for name in ('STATIC', 'FUSED'):
            p = NODE_PATH[name](h)
            if not os.path.exists(p):
                raise SystemExit(f'missing {p}')
            with open(p) as f:
                for r in csv.DictReader(f):
                    if r['run'] in ('avg', 'std'):
                        continue
                    vals[name][int(r['run'])] = float(r['f_measure'])
        runs = sorted(set(vals['STATIC']) & set(vals['FUSED']))[:n_runs]
        out[h] = {'STATIC': [vals['STATIC'][r] for r in runs],
                  'FUSED': [vals['FUSED'][r] for r in runs], 'runs': runs}
    return out


def boot_mean_ci(deltas, clusters=None, n=NBOOT):
    """Bootstrap CI for the mean. If clusters given, resample clusters (not points)."""
    deltas = np.asarray(deltas, dtype=float)
    if clusters is None:
        idx = RNG.integers(0, len(deltas), size=(n, len(deltas)))
        means = deltas[idx].mean(axis=1)
    else:
        keys = list(clusters.keys())
        means = np.empty(n)
        for b in range(n):
            pick = RNG.integers(0, len(keys), size=len(keys))
            vals = np.concatenate([np.asarray(clusters[keys[i]], dtype=float) for i in pick])
            means[b] = vals.mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(deltas.mean()), float(lo), float(hi)


def paired_report(name, static_by_host, fused_by_host, cluster):
    """static/fused: dict host -> list of paired values (aligned)."""
    hosts = sorted(static_by_host)
    s = np.concatenate([np.asarray(static_by_host[h], float) for h in hosts])
    f = np.concatenate([np.asarray(fused_by_host[h], float) for h in hosts])
    d = f - s
    n = len(d)
    wins = int((d > 0).sum()); ties = int((d == 0).sum()); losses = int((d < 0).sum())

    w_two = stats.wilcoxon(d, alternative='two-sided') if np.any(d != 0) else None
    w_greater = stats.wilcoxon(d, alternative='greater') if np.any(d != 0) else None
    n_eff = int(np.count_nonzero(d))
    sign_p = stats.binomtest(wins, wins + losses, 0.5, alternative='greater').pvalue if wins + losses else float('nan')
    t_p = stats.ttest_rel(f, s).pvalue

    # host-cluster bootstrap
    cl = {h: (np.asarray(fused_by_host[h], float) - np.asarray(static_by_host[h], float)) for h in hosts}
    m, lo_run, hi_run = boot_mean_ci(d, None)
    _, lo_cl, hi_cl = boot_mean_ci(d, cl)

    # per-host means
    host_means = {h: float(np.mean(np.asarray(fused_by_host[h], float)
                                   - np.asarray(static_by_host[h], float))) for h in hosts}
    hm = np.array([host_means[h] for h in hosts])
    if np.any(hm != 0):
        hw = stats.wilcoxon(hm, alternative='two-sided')
        hw_g = stats.wilcoxon(hm, alternative='greater')
        hw_p, hw_pg = hw.pvalue, hw_g.pvalue
    else:
        hw_p = hw_pg = float('nan')

    res = dict(level=name, n_pairs=n, n_hosts=len(hosts), wins=wins, ties=ties, losses=losses,
               mean_static=float(s.mean()), mean_fused=float(f.mean()),
               mean_delta=m, ci95_run=[lo_run, hi_run], ci95_cluster=[lo_cl, hi_cl],
               wilcoxon_two_p=(float(w_two.pvalue) if w_two else float('nan')),
               wilcoxon_greater_p=(float(w_greater.pvalue) if w_greater else float('nan')),
               wilcoxon_n_eff=n_eff, sign_p=float(sign_p), ttest_p=float(t_p),
               host_wilcoxon_two_p=float(hw_p), host_wilcoxon_greater_p=float(hw_pg),
               static_min=float(s.min()), fused_min=float(f.min()),
               static_below_005=int((s < 0.05).sum()), fused_below_005=int((f < 0.05).sum()),
               host_means=host_means, cluster=cluster)
    return res


def fmt(x):
    return 'n/a' if x != x else f'{x:.4g}'


def main():
    sub = load_subgraph_pairs()
    node = load_node_pairs(3)
    node10 = load_node_pairs(10)

    sub_s = {h: [v for _, v in sub[h]['STATIC']] for h in sub}
    sub_f = {h: [v for _, v in sub[h]['FUSED']] for h in sub}
    sub_rep = paired_report('subgraph', sub_s, sub_f, cluster='host')

    node_s = {h: node[h]['STATIC'] for h in node}
    node_f = {h: node[h]['FUSED'] for h in node}
    node_rep = paired_report('node', node_s, node_f, cluster='host')
    n10_s = {h: node10[h]['STATIC'] for h in node10}
    n10_f = {h: node10[h]['FUSED'] for h in node10}
    node10_rep = paired_report('node10', n10_s, n10_f, cluster='host')

    # collapse-count test: static vs fused runs below F1<0.05
    a = sub_rep['static_below_005']; b = sub_rep['n_pairs'] - a
    c = sub_rep['fused_below_005']; d_ = sub_rep['n_pairs'] - c
    fisher = stats.fisher_exact([[a, b], [c, d_]], alternative='greater')

    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['level', 'metric', 'value'])
        for rep in (sub_rep, node_rep, node10_rep):
            for k, v in rep.items():
                if k in ('host_means',):
                    continue
                if isinstance(v, list):
                    v = json.dumps([round(x, 6) for x in v])
                w.writerow([rep['level'], k, v])
        w.writerow(['subgraph', 'collapse_fisher_p', fisher.pvalue])
        w.writerow(['subgraph', 'collapse_oddsratio', fisher.statistic])
        for h, m in sub_rep['host_means'].items():
            w.writerow(['subgraph', f'host_mean_delta::{h}', round(m, 6)])
        for h, m in node_rep['host_means'].items():
            w.writerow(['node', f'host_mean_delta::{h}', round(m, 6)])

    L = []
    L.append('# Paired significance: FUSED vs STATIC\n')
    L.append('Generated by `analysis/paired_significance.py`; bootstrap 20,000 resamples, seed 20260911.\n')
    for rep, note in ((sub_rep, 'Subgraph level, 3 runs/host, default rule (27 pairs).'),
                      (node_rep, 'Node level, 3 runs/host (27 pairs; matched to the subgraph level).'),
                      (node10_rep, 'Node level, 10 runs/host (90 pairs; robustness check).')):
        L.append(f'\n## {rep["level"].capitalize()} level\n')
        L.append(note + '\n')
        L.append(f'- mean STATIC = **{rep["mean_static"]:.4f}**, mean FUSED = **{rep["mean_fused"]:.4f}**, '
                 f'mean Δ = **{rep["mean_delta"]:+.4f}**')
        L.append(f'- bootstrap 95% CI (run-level) = [{rep["ci95_run"][0]:+.4f}, {rep["ci95_run"][1]:+.4f}]')
        L.append(f'- bootstrap 95% CI (host-cluster) = [{rep["ci95_cluster"][0]:+.4f}, {rep["ci95_cluster"][1]:+.4f}]')
        L.append(f'- wins/ties/losses = **{rep["wins"]}/{rep["ties"]}/{rep["losses"]}**')
        L.append(f'- Wilcoxon signed-rank (two-sided) p = {fmt(rep["wilcoxon_two_p"])} '
                 f'(n_eff={rep["wilcoxon_n_eff"]}); one-sided greater p = {fmt(rep["wilcoxon_greater_p"])}')
        L.append(f'- sign test (one-sided, wins>losses) p = {fmt(rep["sign_p"])}; paired t-test p = {fmt(rep["ttest_p"])}')
        L.append(f'- host-level Wilcoxon (n={rep["n_hosts"]}) two-sided p = {fmt(rep["host_wilcoxon_two_p"])}, '
                 f'one-sided greater p = {fmt(rep["host_wilcoxon_greater_p"])}')
        if rep['level'] == 'subgraph':
            L.append(f'- runs below F1 0.05: STATIC {rep["static_below_005"]}/{rep["n_pairs"]} '
                     f'vs FUSED {rep["fused_below_005"]}/{rep["n_pairs"]}; '
                     f'Fisher exact one-sided p = {fmt(fisher.pvalue)}')
            L.append(f'- min F1: STATIC {rep["static_min"]:.4f} vs FUSED {rep["fused_min"]:.4f}')
        L.append('- per-host mean Δ: ' + ', '.join(f'{h} {v:+.3f}' for h, v in rep['host_means'].items()))
    open(OUT_MD, 'w').write('\n'.join(L) + '\n')

    print('\n'.join(L))
    print(f'\nwrote {OUT_CSV} and {OUT_MD}')


if __name__ == '__main__':
    main()

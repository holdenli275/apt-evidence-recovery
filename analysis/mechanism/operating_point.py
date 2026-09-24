#!/usr/bin/env python3
"""Operating-point robustness (plan P0-1) + a validation-calibrated protocol.

Uses the cached per-run sweep in analysis/knob_sensitivity.csv (no re-training).

Claim tested: the FUSED advantage is not an artefact of the single default
operating point (tau=10 / k=15).
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = f'{ROOT}/analysis/mechanism'
K = pd.read_csv(f'{ROOT}/analysis/knob_sensitivity.csv')

HOSTS = ['cadets', 'theia', 'trace', 'SysClient0051', 'SysClient0201', 'SysClient0501',
         'SimulatedUbuntu', 'SimulatedW10', 'SimulatedWS12']
M = {'fullz': 'STATIC', 'TGN_fullz': 'FUSED'}
RULES = ([f'f1_topk_{k}' for k in [2, 4, 8, 15, 30]]
         + [f'f1_q_{q}' for q in ['0.5', '0.7', '0.8', '0.9']]
         + ['f1_tau_1', 'f1_tau_10'])
d = K[K.tag.isin(M)].copy()
d['V'] = d.tag.map(M)
d = d[d.host.isin(HOSTS)]

rows = []
for r in RULES:
    if r not in d.columns:
        continue
    for v in ['STATIC', 'FUSED']:
        s = d[d.V == v][r].astype(float)
        rows.append(dict(rule=r, variant=v, n=len(s), mean=round(s.mean(), 4),
                         worst=round(s.min(), 4),
                         collapse=int((s < 0.05).sum()),
                         p10=round(float(np.percentile(s, 10)), 4)))
fixed = pd.DataFrame(rows)

# ---- validation-calibrated: pick the rule on run 0, score it on runs 1-2 ----
piv = {}
for v in ['STATIC', 'FUSED']:
    sub = d[d.V == v]
    piv[v] = sub.set_index(['host', 'run'])[RULES].astype(float)

cal = []
for host in HOSTS:
    for v in ['STATIC', 'FUSED']:
        try:
            r0 = piv[v].loc[host, 0]
        except KeyError:
            continue
        best = r0.idxmax()
        val = [float(piv[v].loc[(host, r), best]) for r in (1, 2) if (host, r) in piv[v].index]
        default = [float(piv[v].loc[(host, r), 'f1_tau_10']) for r in (1, 2) if (host, r) in piv[v].index]
        oracle = [float(piv[v].loc[(host, r), RULES].max()) for r in (1, 2) if (host, r) in piv[v].index]
        cal.append(dict(host=host, variant=v, picked_on_run0=best,
                        val_mean=round(np.mean(val), 4),
                        default_mean=round(np.mean(default), 4),
                        oracle_mean=round(np.mean(oracle), 4)))
cal = pd.DataFrame(cal)

fixed.to_csv(f'{OUT}/operating_point_fixed.csv', index=False)
cal.to_csv(f'{OUT}/operating_point_calibrated.csv', index=False)

with open(f'{OUT}/OPERATING_POINT.md', 'w') as f:
    f.write('# Operating-point robustness (P0-1)\n\n')
    f.write('Source: `analysis/knob_sensitivity.csv` (cached candidate subgraphs, exact evaluator).\n')
    f.write('A "collapse" is a run with F1 < 0.05.\n\n## Fixed rule, all 27 (host,run) units\n\n')
    f.write('| rule | STATIC mean / worst / collapse | FUSED mean / worst / collapse |\n|---|---|---|\n')
    for r in RULES:
        a = fixed[(fixed.rule == r) & (fixed.variant == 'STATIC')].iloc[0]
        b = fixed[(fixed.rule == r) & (fixed.variant == 'FUSED')].iloc[0]
        f.write(f'| `{r}` | {a["mean"]:.3f} / {a["worst"]:.3f} / {a.collapse}/27 | '
                f'{b["mean"]:.3f} / {b["worst"]:.3f} / {b.collapse}/27 |\n')
    f.write('\n## Validation-calibrated (rule chosen on run 0, scored on runs 1-2)\n\n')
    f.write('| host | variant | rule picked on run0 | val mean | default (tau=10) | oracle |\n|---|---|---|---|---|---|\n')
    for _, r in cal.iterrows():
        f.write(f'| {r.host} | {r.variant} | `{r.picked_on_run0}` | {r.val_mean:.3f} | '
                f'{r.default_mean:.3f} | {r.oracle_mean:.3f} |\n')
    f.write('\n### Validation-calibrated 9-host mean\n\n')
    g = cal.groupby('variant')[['val_mean', 'default_mean', 'oracle_mean']].mean().round(4)
    f.write('| variant | validation-calibrated | default (tau=10) | oracle |\n|---|---|---|---|\n')
    for v in ['STATIC', 'FUSED']:
        if v in g.index:
            r = g.loc[v]
            f.write(f'| {v} | {r.val_mean:.4f} | {r.default_mean:.4f} | {r.oracle_mean:.4f} |\n')

print(fixed.to_string(index=False))
print()
print(cal.to_string(index=False))
print()
print(cal.groupby('variant')[['val_mean', 'default_mean', 'oracle_mean']].mean().round(4).to_string())

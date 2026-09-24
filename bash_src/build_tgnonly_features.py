#!/usr/bin/env python3
"""Build Full_Script_Test_TGNonly_pyg: node features = the TGN 32-d block ONLY
(drop the fullz/statistical block), for the TGN-only ablation.

TGN_fullz feature layout = [fullz | TGN32] (TGN appended at the end by
encode_to_PyG.concat_tgn_embeddings; verified: prefix == fullz feature file,
and total dim == fullz_dim + 32 on every host/type).

TGN-only features are therefore the *last 32 columns* of the TGN_fullz
feature tensors -> identical TGN input (same checkpoint, same normalisation)
as the reported TGN_fullz runs; only the fullz block is removed.

Sanity per type: dimensions and prefix equality with the fullz feature file.
"""
import os, shutil, glob, argparse
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORP = {'SimulatedUbuntu': 'nodlink', 'SimulatedW10': 'nodlink', 'SimulatedWS12': 'nodlink',
        'cadets': 'darpa_tc3', 'theia': 'darpa_tc3', 'trace': 'darpa_tc3',
        'SysClient0051': 'darpa_optc', 'SysClient0201': 'darpa_optc', 'SysClient0501': 'darpa_optc'}
SRC = 'Full_Script_Test_TGN_fullz_pyg'
DST = 'Full_Script_Test_TGNonly_pyg'
FULLZ = 'Full_Script_Test_fullz_pyg'
TGN_DIM = 32


def load(p):
    return torch.load(p, map_location='cpu', weights_only=False).float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', required=True)
    args = ap.parse_args()
    host = args.host
    exp_dir = os.path.join(ROOT, 'dataset', CORP[host], host, 'experiments')
    src_dir = os.path.join(exp_dir, SRC)
    dst_dir = os.path.join(exp_dir, DST)
    if not os.path.isdir(dst_dir):
        shutil.copytree(src_dir, dst_dir)
        print(f'[{host}] copied {SRC} -> {DST}')
    types = [os.path.basename(os.path.dirname(p))
             for p in sorted(glob.glob(os.path.join(src_dir, 'features', '*', 'node-features.pt')))]
    for typ in types:
        old = load(os.path.join(src_dir, 'features', typ, 'node-features.pt'))
        assert old.shape[1] > TGN_DIM, f'{host}/{typ}: {old.shape}'
        tgnonly = old[:, -TGN_DIM:].contiguous()
        torch.save(tgnonly, os.path.join(dst_dir, 'features', typ, 'node-features.pt'))
        note = []
        fz_p = os.path.join(exp_dir, FULLZ, 'features', typ, 'node-features.pt')
        if os.path.isfile(fz_p):
            fz = load(fz_p)
            note.append(f'fullz={fz.shape[1]} prefix_ok={fz.shape[1] + TGN_DIM == old.shape[1] and torch.equal(fz, old[:, :fz.shape[1]])}')
        print(f'[{host}] {typ}: {tuple(old.shape)} -> {tuple(tgnonly.shape)} ' + ' | '.join(note))
    print(f'[{host}] done: {len(types)} types')


if __name__ == '__main__':
    main()

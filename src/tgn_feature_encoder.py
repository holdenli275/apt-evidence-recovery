#!/usr/bin/env python
"""Compact TGN temporal encoder for OCR-APT (minimal incremental temporal features).

Implements the core of TGN (Rossi et al., 2020; see PyG `TGNMemory` semantics):
node memory (GRU) + learnable time encoding + message MLP + self-supervised
temporal link prediction.  The expensive temporal neighbour-attention
aggregation is omitted so the model trains on CPU for large graphs; node
embeddings are `id_emb + memory`, which is the memory-based embedding used by
TGN.

Training loop follows PyG `examples/tgn.py`:
  * chronological batches; at batch k the forward pass applies a *local* GRU
    update from the previous batch's messages on top of the flushed memory
    buffer, so gradients flow through the message MLP and GRU;
  * after backward/step the same messages are flushed into the memory buffer
    under `no_grad` (persistent state), exactly like `TGNMemory.update_state`.

Output: a dict {node_uuid(str): 1-D tensor} saved via `torch.save`, aligned
with `node-features.pt` because `encode_to_PyG.py --use-tgn` merges it by
`node_uuid` (the per-type `entidx2name.csv` mapping).

Usage:
  .venv/bin/python -B -u src/tgn_feature_encoder.py \\
      --graph-df dataset/nodlink/SimulatedUbuntu/experiments/SimulatedUbuntu_graph_df.csv \\
      --out dataset/nodlink/SimulatedUbuntu/experiments/tgn_embeddings/SimulatedUbuntu_tgn.pt \\
      --dim 32 --epochs 5 --batch-size 65536 --seed 360
"""

import argparse
import math
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


def parse_args():
    p = argparse.ArgumentParser(description="Compact TGN encoder for OCR-APT")
    p.add_argument("--graph-df", type=str, required=True,
                   help="*_graph_df.csv (tab-separated: source-id/source-type/"
                        "destination-id/destination-type/edge-type/timestamp)")
    p.add_argument("--out", type=str, required=True,
                   help="output .pt path (dict node_uuid -> embedding)")
    p.add_argument("--dim", type=int, default=32, help="embedding/memory dim")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=65536)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=360)
    p.add_argument("--max-edges", type=int, default=0,
                   help="0 = use all edges (small value = smoke test only)")
    p.add_argument("--no-id-emb", action="store_true",
                   help="final embedding = GRU memory only (drop per-node id embedding)")
    p.add_argument("--event-gate-k", type=float, default=0.0,
                   help="scale embedding by event-count confidence gate; 0=off")
    p.add_argument("--gate-style", choices=["log", "linear"], default="log",
                   help="gate ramp: log = min(1, log1p(events)/log1p(k)); "
                        "linear = min(1, events/k)")
    p.add_argument("--no-norm", action="store_true",
                   help="do not L2-normalize final embeddings")

    return p.parse_args()


class TimeEncoder(nn.Module):
    """TGN time encoding (PyG-style): cos(Linear(t))."""
    def __init__(self, dim):
        super().__init__()
        self.lin = nn.Linear(1, dim)

    def forward(self, t):
        return torch.cos(self.lin(t.view(-1, 1)))


def mean_aggregate(msg, inv, num_uniq):
    """Mean-aggregate messages by group id (out-of-place: keeps gradients)."""
    dim = msg.size(-1)
    index = inv.unsqueeze(1).expand(-1, dim)
    total = torch.zeros(num_uniq, dim, dtype=msg.dtype, device=msg.device)
    total = total.scatter_add(0, index, msg)
    cnt = torch.bincount(inv, minlength=num_uniq).to(total.dtype).unsqueeze(1)
    return total / cnt.clamp_min(1.0)


class CompactTGN(nn.Module):
    def __init__(self, num_nodes, num_edge_types, dim):
        super().__init__()
        self.num_nodes = num_nodes
        self.dim = dim
        self.id_emb = nn.Embedding(num_nodes, dim)
        self.edge_emb = nn.Embedding(num_edge_types, dim)
        self.time_enc = TimeEncoder(dim)
        # message = MLP(cat(mem_src, mem_dst, edge_emb, time_enc))
        self.msg = nn.Sequential(nn.Linear(4 * dim, dim), nn.ReLU(),
                                 nn.Linear(dim, dim))
        self.gru = nn.GRUCell(dim, dim)
        # link score = MLP(cat(z_src, z_dst, time_enc))
        self.decoder = nn.Sequential(nn.Linear(3 * dim, dim), nn.ReLU(),
                                     nn.Linear(dim, 1))
        self.register_buffer("memory", torch.zeros(num_nodes, dim))
        self.register_buffer("last_update", torch.zeros(num_nodes))

    def updated_memory(self, store):
        """Return (new_mem_for_uniq, uniq, inv, t_all) with gradients flowing
        through the message MLP and GRU (used in the forward pass)."""
        src, dst, t, e = store
        mem = self.memory
        t_rel_s = t - self.last_update[src]
        t_rel_d = t - self.last_update[dst]
        ms = self.msg(torch.cat(
            [mem[src], mem[dst], self.edge_emb(e), self.time_enc(t_rel_s)],
            dim=-1))
        md = self.msg(torch.cat(
            [mem[dst], mem[src], self.edge_emb(e), self.time_enc(t_rel_d)],
            dim=-1))
        node = torch.cat([src, dst])
        msg = torch.cat([ms, md], dim=0)
        tt = torch.cat([t, t])
        uniq, inv = torch.unique(node, return_inverse=True)
        aggr = mean_aggregate(msg, inv, uniq.numel())
        new_mem = self.gru(aggr, mem[uniq])
        return new_mem, uniq, inv, tt

    def flush(self, store):
        """Persist store's messages into the memory buffer (no_grad), matching
        PyG TGNMemory.update_state."""
        with torch.no_grad():
            new_mem, uniq, inv, tt = self.updated_memory(store)
            self.memory[uniq] = new_mem
            last_upd = self.last_update.new_zeros(uniq.numel()).scatter_reduce_(
                0, inv, tt, reduce="amax")
            self.last_update[uniq] = last_upd
            self.memory.detach_()
            self.last_update.detach_()

    def embed(self, store):
        """Node embedding matrix (id_emb + updated memory), gradient-safe."""
        mem = self.memory
        if store is not None:
            new_mem, uniq, _, _ = self.updated_memory(store)
            mask = torch.zeros(self.num_nodes, dtype=torch.bool,
                               device=mem.device)
            mask[uniq] = True
            mem_full = torch.zeros_like(mem).scatter_add(
                0, uniq.unsqueeze(1).expand(-1, self.dim), new_mem)
            mem = torch.where(mask.unsqueeze(1), mem_full, mem)
        return self.id_emb.weight + mem


def train_tgn(args, src, dst, t, e, uuid_list, type_code, type_nodes,
              edge_types, events):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(max(1, os.cpu_count() // 2))

    num_nodes = len(uuid_list)
    model = CompactTGN(num_nodes, len(edge_types), args.dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # chronological batches
    order = torch.argsort(t)
    src, dst, t, e = src[order], dst[order], t[order], e[order]
    n_batches = (src.numel() + args.batch_size - 1) // args.batch_size
    batches = []
    for i in range(n_batches):
        sl = slice(i * args.batch_size, (i + 1) * args.batch_size)
        batches.append((src[sl], dst[sl], t[sl], e[sl]))

    def sample_negatives(dst_t):
        dst_np = dst_t.numpy()
        types = type_code[dst_np]
        neg = np.empty_like(dst_np)
        for tc in np.unique(types):
            mask = types == tc
            pool = type_nodes[tc]
            if len(pool) == 0:  # degenerate subset (e.g. --max-edges smoke)
                neg[mask] = np.random.randint(0, num_nodes, int(mask.sum()))
            else:
                neg[mask] = pool[np.random.randint(0, len(pool), int(mask.sum()))]
        return torch.from_numpy(neg)

    print(f"[tgn] nodes={num_nodes} edges={src.numel()} edge_types="
          f"{len(edge_types)} dim={args.dim} batches/epoch={n_batches}",
          flush=True)
    hist = []
    for epoch in range(1, args.epochs + 1):
        model.memory.zero_()
        model.last_update.zero_()
        model.train()
        store = None
        t0 = time.time()
        tot_loss = 0.0
        for bi, batch in enumerate(batches):
            optimizer.zero_grad()
            b_src, b_dst, b_t, b_e = batch
            neg = sample_negatives(b_dst)
            z = model.embed(store)
            z_src, z_dst, z_neg = z[b_src], z[b_dst], z[neg]
            t_enc = model.time_enc(b_t)
            pos = model.decoder(torch.cat([z_src, z_dst, t_enc], dim=-1))
            negd = model.decoder(torch.cat([z_src, z_neg, t_enc], dim=-1))
            loss = (F.binary_cross_entropy_with_logits(pos, torch.ones_like(pos))
                    + F.binary_cross_entropy_with_logits(negd,
                                                         torch.zeros_like(negd)))
            loss.backward()
            optimizer.step()
            if store is not None:
                model.flush(store)
            store = batch
            tot_loss += float(loss.item())
            if bi % 25 == 0:
                print(f"[tgn] epoch {epoch}/{args.epochs} batch "
                      f"{bi + 1}/{n_batches} loss {loss.item():.4f} "
                      f"({time.time() - t0:.1f}s)", flush=True)
        # flush the final store so memory reflects the whole timeline
        if store is not None:
            model.flush(store)
        avg = tot_loss / n_batches
        hist.append(avg)
        print(f"[tgn] epoch {epoch}/{args.epochs} done avg_loss={avg:.4f} "
              f"({time.time() - t0:.1f}s)", flush=True)

    # final embedding: (id_emb + memory) or memory-only, L2-normalized per node;
    # optional event-count confidence gate (w=1 for active nodes, ->0 for sparse)
    model.eval()
    z = model.memory if args.no_id_emb else model.embed(None)
    if not args.no_norm:
        z = F.normalize(z, p=2, dim=1)
    if args.event_gate_k > 0:
        if args.gate_style == "linear":
            w = torch.clamp(events / args.event_gate_k, 0, 1)
        else:
            w = torch.clamp(torch.log1p(events) / math.log1p(args.event_gate_k), 0, 1)
        z = z * w[:, None]
        print(f"[tgn] event-gate k={args.event_gate_k}: "
              f"w<=0.5 for {(w <= 0.5).float().mean():.1%} nodes, "
              f"mean w={w.mean():.3f}", flush=True)
    z = z.detach()
    emb_dict = {uuid_list[i]: z[i] for i in range(num_nodes)}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(emb_dict, args.out)
    torch.save({"args": vars(args), "loss_hist": hist,
                "num_nodes": num_nodes, "edge_types": edge_types},
               args.out + ".meta.pt")
    print(f"[tgn] saved {num_nodes} embeddings -> {args.out}", flush=True)
    return emb_dict


def main():
    args = parse_args()
    df = pd.read_csv(args.graph_df, sep="\t", nrows=args.max_edges or None,
                     encoding_errors="ignore")
    print(f"[tgn] loaded {len(df)} rows from {args.graph_df}", flush=True)

    df["edge-type"] = (df["edge-type"].astype(str)
                       .str.replace("EVENT_", "", regex=False).str.lower())
    edge_types = sorted(df["edge-type"].unique().tolist())
    edge2idx = {et: i for i, et in enumerate(edge_types)}

    all_uuid = pd.unique(pd.concat([df["source-id"], df["destination-id"]]))
    all_uuid = sorted(str(u) for u in all_uuid)
    uuid2idx = {u: i for i, u in enumerate(all_uuid)}
    print(f"[tgn] nodes={len(all_uuid)} edge_types={edge_types}", flush=True)

    src = df["source-id"].astype(str).map(uuid2idx).to_numpy()
    dst = df["destination-id"].astype(str).map(uuid2idx).to_numpy()
    e = df["edge-type"].map(edge2idx).to_numpy()
    ts_col = df["timestamp"]
    if pd.api.types.is_integer_dtype(ts_col):
        ts = ts_col.to_numpy(dtype=np.int64)
    else:
        ts = pd.to_datetime(ts_col).astype("int64").to_numpy()
    t = ((ts - ts.min()) / 1e9).astype(np.float32)  # seconds since first event

    # per-node type (for type-matched negative sampling)
    src_map = df[["source-id", "source-type"]].drop_duplicates()
    dst_map = df[["destination-id", "destination-type"]].drop_duplicates()
    typ = pd.concat([
        src_map.rename(columns={"source-id": "id", "source-type": "type"}),
        dst_map.rename(columns={"destination-id": "id", "destination-type": "type"}),
    ]).drop_duplicates("id")
    type_of = dict(zip(typ["id"].astype(str), typ["type"]))
    type_names = sorted(set(type_of.values()))
    type2code = {tn: i for i, tn in enumerate(type_names)}
    type_code = np.zeros(len(all_uuid), dtype=np.int64)
    type_nodes = {i: [] for i in range(len(type_names))}
    for u, i in uuid2idx.items():
        tc = type2code[type_of[u]]
        type_code[i] = tc
        type_nodes[tc].append(i)
    type_nodes = {tc: np.array(v, dtype=np.int64)
                  for tc, v in type_nodes.items()}

    src_t = torch.from_numpy(src).long()
    dst_t = torch.from_numpy(dst).long()
    e_t = torch.from_numpy(e).long()
    t_t = torch.from_numpy(t)

    # per-node event counts (in+out), aligned to all_uuid order
    cnt = pd.concat([df["source-id"].astype(str),
                        df["destination-id"].astype(str)]).value_counts()
    events = torch.tensor([cnt.get(u, 0) for u in all_uuid], dtype=torch.float32)
    print(f"[tgn] event counts: median={events.median():.0f} "
          f"p90={events.quantile(.9):.0f} max={events.max():.0f}", flush=True)

    train_tgn(args, src_t, dst_t, t_t, e_t, all_uuid, type_code, type_nodes,
              edge_types, events)


if __name__ == "__main__":
    main()

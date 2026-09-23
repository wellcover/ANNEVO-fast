# -*- coding: utf-8 -*-
"""predict.py — sliding-window per-base probability prediction for
ANNEVO-style base-level gene annotation models.

Given a genome FASTA and a PyTorch checkpoint (see model/ANNEVO_seq.py),
writes an HDF5 file with per-base class probabilities for both strands:
    <out>/<seq_id>/predictions_forward  [L, n_classes]
    <out>/<seq_id>/predictions_reverse  [L, n_classes]

Optimizations for large/fragmented genomes:
- windows from all scaffolds in a chunk are flattened and run through the
  model in two big batches (forward and reverse-complement) instead of two
  per scaffold;
- short scaffolds can be skipped (--min_len); missing H5 groups are skipped
  automatically by the downstream decoder;
- empty/ultra-short scaffolds are handled safely.

Usage:
    python predict.py -g genome.fa -o probs.h5 --checkpoint model.pt \
        [--batch_size 16] [--chunk_mb 200] [--min_len 1024] \
        [--window 30720] [--flank 5120]
"""
import argparse
import gc
import os
import re
import sys
import time

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from Bio import SeqIO
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import ANNEVO_seq  # noqa: E402

COMP = str.maketrans("ATGCatgcNnXx", "TACGtacgNnXx")
_SEQ_LUT = np.zeros((256, 4), dtype=np.float32)
for _i, _b in enumerate("ACGT"):
    _SEQ_LUT[ord(_b)] = np.eye(4, dtype=np.float32)[_i]
    _SEQ_LUT[ord(_b.lower())] = np.eye(4, dtype=np.float32)[_i]


def sequence_encode(seq):
    return _SEQ_LUT[np.frombuffer(seq.encode("ascii"), np.uint8)]


def rev_complement(s):
    return s.translate(COMP)[::-1]


class GenomeDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return torch.tensor(sequence_encode(self.data[idx]), dtype=torch.float)


def predict_windows(model, windows, device, batch_size, num_workers=8, tag=""):
    """A batch of windows → [N, window_size, num_classes] fp16 numpy array (fp32 compute)."""
    if not windows:
        return None
    dataset = GenomeDataset(windows)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, persistent_workers=False)
    all_outputs = None
    w = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Predict{tag}", mininterval=10):
            seqs = batch.to(device).float()
            result = model(seqs)[0]
            out = F.softmax(result, dim=-1).to(torch.float16).cpu().numpy()
            if all_outputs is None:
                all_outputs = np.empty((len(dataset), out.shape[1], out.shape[2]), np.float16)
            all_outputs[w:w + out.shape[0]] = out
            w += out.shape[0]
    return all_outputs


def make_windows(seq, window_size, flank_length):
    """One sequence → (fwd_windows, rc_windows in reverse order, actual length L). L=0 → empty."""
    L = len(seq)
    if L == 0:
        return [], [], 0
    step_size = window_size
    starts = list(range(0, L, step_size))
    total_len_needed = starts[-1] + window_size + 2 * flank_length
    pad_behind = total_len_needed - (L + flank_length)
    padded = 'N' * flank_length + seq + 'N' * pad_behind
    fwd_windows = []
    rc_windows = []
    for os_ in starts:
        s = os_ + flank_length
        e = s + window_size
        w_fwd = padded[s - flank_length:e + flank_length]
        fwd_windows.append(w_fwd)
        rc_windows.append(rev_complement(w_fwd))
    rc_windows = rc_windows[::-1]
    return fwd_windows, rc_windows, L


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--genome", required=True)
    parser.add_argument("-o", "--out", required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--chunk_mb", type=int, default=200)
    parser.add_argument("--min_len", type=int, default=40960,
                        help="Skip scaffolds shorter than this (window+flanks=40960bp for the "
                             "default config; scaffolds without an H5 group are skipped by the "
                             "downstream decoder; 0 disables the filter)")
    parser.add_argument("--checkpoint", required=True, help="PyTorch state_dict checkpoint path")
    parser.add_argument("--window", type=int, default=30720,
                        help="Prediction window length (must match the checkpoint's training config)")
    parser.add_argument("--flank", type=int, default=5120,
                        help="Flank length on both sides (must match the checkpoint's training config)")
    args = parser.parse_args()

    window_size, flank_length = args.window, args.flank
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = ANNEVO_seq.ANNEVO(channels=64, num_classes=15, num_heads=8,
                              window_size=window_size, flank_length=flank_length,
                              num_encoder_layers=6, n_experts=8,
                              local_pattern_size=32, bal_loss_coef=1e-3)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    print(f"[model] checkpoint loaded: {args.checkpoint}")

    print("[genome] reading...")
    t0 = time.time()
    genome = {}
    n_skip, bp_skip = 0, 0
    idx = SeqIO.index(args.genome, "fasta")
    for sid in idx:
        seq = str(idx[sid].seq).upper()
        seq = re.sub(r'[^ATCG]', 'N', seq)
        if len(seq) < max(args.min_len, 1):
            n_skip += 1
            bp_skip += len(seq)
            continue
        genome[sid] = seq
    idx.close()
    print(f"[genome] {len(genome)} sequences kept, {time.time()-t0:.1f}s"
          + (f" | filtered: {n_skip} scaffolds < {args.min_len}bp ({bp_skip/1e6:.2f} Mb)" if n_skip else ""))

    cumulative = 0
    chunk_num = 1
    buffer = {}
    n_buffered = 0

    def flush(buf):
        sids = list(buf.keys())
        fwd_flat, rc_flat, spans = [], [], []
        w0 = 0
        for sid in sids:
            fwd_w, rc_w, L = buf[sid]
            fwd_flat.extend(fwd_w)
            rc_flat.extend(rc_w)
            spans.append((sid, L, w0, w0 + len(fwd_w)))
            w0 += len(fwd_w)
        t0 = time.time()
        fwd_all = predict_windows(model, fwd_flat, device, args.batch_size, tag="(+)")
        rc_all = predict_windows(model, rc_flat, device, args.batch_size, tag="(-)")
        n_cls = fwd_all.shape[-1]
        with h5py.File(args.out, "a") as f:
            for sid, L, a, b in spans:
                fwd_flat_s = fwd_all[a:b].reshape(-1, n_cls)[:L]
                rc_flat_s = rc_all[a:b].reshape(-1, n_cls)[-L:]
                grp = f.create_group(sid)
                grp.create_dataset("predictions_forward", data=fwd_flat_s)
                grp.create_dataset("predictions_reverse", data=rc_flat_s)
        print(f"[chunk] {len(sids)} scaffolds, {w0} windows/strand, {time.time()-t0:.1f}s")

    for sid, seq in genome.items():
        fwd_w, rc_w, L = make_windows(seq, window_size, flank_length)
        buffer[sid] = [fwd_w, rc_w, L]
        n_buffered += 1
        cumulative += len(fwd_w) * (window_size + 2 * flank_length)
        if cumulative > args.chunk_mb * 1e6:
            print(f"\n[chunk {chunk_num}] cumulative {cumulative/1e6:.0f} Mb, {n_buffered} scaffolds")
            flush(buffer)
            buffer = {}
            n_buffered = 0
            cumulative = 0
            chunk_num += 1
            gc.collect()
            torch.cuda.empty_cache()

    if buffer:
        print(f"\n[chunk {chunk_num}] (final) {n_buffered} scaffolds")
        flush(buffer)

    print("[done]")


if __name__ == "__main__":
    main()

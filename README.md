# ANNEVO-Fast

Optimized **prediction and HMM decoding pipeline** for [ANNEVO](https://github.com/xjtu-omics/ANNEVO)-style
base-level ab initio gene annotation. It is a drop-in, faster replacement for the
official inference/decoding steps, producing **identical gene content** with the
reference decoder, plus a few optional accuracy improvements.

Based on the ANNEVO architecture and decoding logic
([Zhang et al., Nat Methods 2026](https://doi.org/10.1038/s41592-026-03036-7));
this repository redistributes a modified derivative under the terms of the
original ANNEVO Non-Commercial License (see `LICENSE`).

## What is different from the official pipeline

**Decoding (`decoding.py`, `src/`)**
- **Sparse-edge Viterbi** — the dense `O(L·S²)` inner loop is replaced by a
  sparse grammar-edge table `O(L·E)`. Gene-grammar transitions have very few
  outgoing edges per state, so at `min_intron_length=20` (170 states) each base
  needs ~hundreds of comparisons instead of ~29 000. Edge evaluation order and
  tie-breaking exactly follow the dense reference, so decoded paths are
  **bit-for-bit identical**.
- **Fully vectorized candidate detection / range parsing / gene scoring**
  (cumsum window scan, run-length encoding, float64 sequential accumulation),
  removing the pandas dependency.
- State/transition/edge tables are **cached per process** instead of being
  rebuilt per region; the H5 file is opened once per batch instead of once per
  segment.
- Deterministic output ordering (chromosomes in lexicographic order).
- Optional accuracy knobs: **AT-AC (U12-type) splice path**, tunable **GC-AG
  donor penalty**, and a recalibrated CDS score filter (`--min_cds_score`,
  worth ~+0.2–0.4 pp gene F1 on plant benchmarks).

**Prediction (`predict.py`)**
- Cross-scaffold chunked batching: windows from all scaffolds in a chunk are
  flattened and run through the model in **two big passes** (forward /
  reverse-complement) instead of two passes per scaffold — a major win on
  fragmented assemblies with hundreds of thousands of contigs.
- Safe handling of empty/ultra-short scaffolds and optional `--min_len` filtering.
- `--window/--flank` are configurable (e.g. 102 kb long-context models).

## Performance

24 CPU threads, identical GFF gene content vs the reference decoder:

| Task | Reference | This repo | Speed-up |
|---|---|---|---|
| Synthetic 2×5 Mb (6 047 genes, short-intron re-decoding path) | 17.9 s | 3.4 s | **5.3×** |
| *Arabidopsis thaliana* whole genome (24 625 genes) | 59.4 s | 22.3 s | **2.7×** |

## Usage

```bash
pip install -r requirements.txt

# 1) per-base probabilities (needs a trained ANNEVO-style checkpoint)
python predict.py -g genome.fa -o probs.h5 --checkpoint model.pt

# 2) HMM decoding to GFF3 (drop-in replacement for the official decoding step)
python decoding.py -g genome.fa -p probs.h5 -o genes.gff -t 24

# optional: enable the AT-AC (U12) splice path
python decoding.py -g genome.fa -p probs.h5 -o genes.gff -t 24 --atac_penalty 5
```

`decoding.py` is model-agnostic: any tool writing per-base class probabilities
in the ANNEVO HDF5 layout (`<seq_id>/predictions_forward|predictions_reverse`,
15 classes, both strands) can feed it directly.

## Repository layout

```
├── predict.py            # sliding-window probability prediction
├── decoding.py           # GFF decoding CLI
├── src/
│   ├── HMM.py            # gene-grammar HMM + sparse-edge Viterbi (numba)
│   ├── gene_decoding.py  # region detection, scoring, GFF3 output
│   └── predict_nucleotide.py
└── model/
    └── ANNEVO_seq.py     # PyTorch architecture definition used by predict.py
```

## License & citation

This project is a derivative work of [ANNEVO](https://github.com/xjtu-omics/ANNEVO)
and is released under the same **ANNEVO Non-Commercial License** (see `LICENSE`,
© 2025 Pengyu Zhang, Kai Ye, Xi'an Jiaotong University): free for academic and
non-profit research; commercial use requires a separate license from the
original copyright holders. Modifications in this repository © 2026 the
ANNEVO-Fast authors.

If you use ANNEVO-Fast, please cite the ANNEVO paper:

> Zhang, P., Xu, T., Wang, S. et al. Highly accurate ab initio gene annotation
> with ANNEVO. *Nat Methods* 23, 740–748 (2026). https://doi.org/10.1038/s41592-026-03036-7

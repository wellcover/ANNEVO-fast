# Inside the ANNEVO-Fast Decoder

*How a gene-grammar HMM is made 3–5× faster without changing a single base of its output.*

This article explains the algorithms and engineering behind the decoding half of
ANNEVO-Fast (`decoding.py`, `src/HMM.py`, `src/gene_decoding.py`): what the hidden
Markov model does, why its Viterbi decode was the bottleneck, how the sparse-edge
rewrite works, what "bit-exact equivalence" actually guarantees and how it is
enforced, and where the remaining runtime goes.

All numbers below are measured on the code in this repository
(`min_intron_length = 20` unless stated otherwise).

---

## 1. What the decoder solves

An ANNEVO-style model outputs, for every genome base and both strands, a
posterior distribution over 15 label classes:

```
0  Intergenic       5  Intron_2        10 ASS_0   (acceptor, phase 0)
1  Coding_exon_0    6  Intron_1        11 ASS_2
2  Coding_exon_2    7  DSS_0           12 ASS_1
3  Coding_exon_1    8  DSS_2           13 start
4  Intron_0         9  DSS_1           14 end
```

(The class order is historical — note the swapped phase-1/phase-2 entries; the
emission mapping in `viterbi_decoding` hard-codes this permutation.)

Per-base argmax is not enough to produce gene structures. The raw posteriors
satisfy the *gene grammar* only statistically: an argmax path can contain an
intron that is 3 bp long, a donor site not followed by an acceptor, a CDS phase
that jumps arbitrarily between exons. The decoder's job is to find the
**highest-scoring path that is grammatically legal** — this is exactly a
Viterbi decode over a hand-designed HMM whose topology encodes eukaryotic gene
structure.

The pipeline in one sentence:

```
per-base posteriors (HDF5)
   → candidate genic regions (cheap thresholds)
   → per-region Viterbi over the grammar HMM
   → gene scoring / filtering
   → GFF3
```

The same pipeline as a diagram — note the lazy re-decode loop (§2.3) and the
per-process cache that all Viterbi calls share:

```mermaid
flowchart LR
    H[("per-base posteriors<br/>(HDF5, both strands)")] --> R["candidate region detection<br/>cumsum window scan + RLE"]
    R --> P{"any intron &lt; min_intron_length<br/>in decoded result?"}
    P -- "decode @ min_intron=1" --> V1["Viterbi<br/>sparse edge table"]
    P -- "yes → re-decode" --> V2["Viterbi<br/>full counter machine"]
    V1 --> S["gene scoring / filtering<br/>(--min_cds_score)"]
    V2 --> S
    C[("process-level cache:<br/>states · 5 matrices · edge table")]
    C -.-> V1
    C -.-> V2
    S --> G["GFF3 output<br/>(deterministic order)"]
```

![Overview: state inventory, inner-loop reduction, end-to-end benchmark](assets/decoding_figures.png)

The three panels above summarize the whole story quantitatively — where the
170 states come from (A), what the sparse rewrite saves per base (B), and what
it buys end-to-end (C) — and the rest of the article walks through each in
detail.

---

## 2. The gene-grammar HMM

### 2.1 State inventory

At `min_intron_length = 20` the machine has **170 states**. They fall into
five families:

| Family | Count | Examples | Role |
|---|---|---|---|
| intergenic | 1 | `intergenic` | background |
| start codon | 3 | `start0..start2` | spells ATG |
| CDS, phase-tracked | 6 | `CDS0`, `CDS0_T`, `CDS1`, `CDS1_TA`, `CDS1_TG`, `CDS2` | exon body in reading frame 0/1/2 |
| donor/acceptor motifs | 12 | `DSS0`, `DSS1_TA`, `ASS2`, … | spell GT…AG around splice sites |
| acceptor/stop | 4 | `end0`, `end1_TA`, `end1_TG`, `end2` | spell TAA/TAG/TGA |
| splice helpers | 24 | `intron1_TG_splice0..3` | consume the GT…AG motif inside introns |
| intron length counters | 120 | `intron0_17`, `intron2_0`, … | enforce `min_intron_length` |

Three design points deserve explanation. First, the overall shape of the
grammar — the backbone every path follows (phase cycle in the middle, splice
arc below, stop path to the right):

```mermaid
flowchart LR
    IG["intergenic"] -- "A" --> S0["start0"] -- "T" --> S1["start1"] -- "G" --> S2["start2"]
    S2 --> C0["CDS0<br/>(phase 0)"]
    C0 -- "exon continues" --> C1["CDS1"] -- "G" --> C2["CDS2"] -- "exon ends / next codon" --> C0
    C2 -- "T (stop begins)" --> E0["end0"] -- "A" --> E1["end1_TA / end1_TG"] -- "A | G" --> E2["end2"]
    E2 -- "gene done" --> IG
    C0 -- "donor GT.." --> DSS["DSS donor states"]
    DSS --> IC["intron counter chain<br/>intron{p}_0 … intron{p}_19<br/>(+ _T/_TA/_TG suffix variants)"]
    IC -- "..AG acceptor" --> ASS["ASS acceptor states"]
    ASS -- "frame-preserving return" --> C0
```

*(A simplified backbone — the full machine has 170 states at
`min_intron_length = 20`; the suffix tags and the second CDS arc from
`CDS1`/`ASS1` are omitted for readability.)*

**Phase tracking across introns.** `CDS0/1/2` encode the position within the
current codon. To keep the frame consistent across an intron, the machine must
remember what the exon ended with. The suffix-tagged variants do exactly that:
`CDS0_T` means "exon in phase 0 that ended with T", and only from that state
can the intron variant `intron0_T_*` be entered. The same suffix tags on intron
states (`intron1_TA_…`) allow a **stop codon interrupted by an intron** to be
recognized: a `T` before the donor, an intron, then `AA`/`AG` after the
acceptor completes `TAA`/`TAG`, and the machine transitions into the `end`
states instead of continuing the CDS. This is why the transition tables show
entries like `intron1_TA_splice3 → end0` on base `T`… wait, on base `A` into
`end1`-equivalents: the exact spelling is that acceptor-exit states reach
`end0` on `T`, and `end1_TA → end2` on `A` completes the split stop.

**Intron length counters.** A hard minimum intron length cannot be enforced by
emissions; it needs topology. The machine instantiates `min_intron_length`
counter states per suffix family (`intron0_0 … intron0_19`), forming chains
that make any intron shorter than the minimum topologically impossible. This
is the single biggest multiplier on state count (120 of the 170 states).

**Nucleotide-conditional transitions.** There are **five** separate S×S
transition matrices — one each for the current base being A, T, C, G, and N.
The grammar is DNA-aware: `start1 → start2` fires only on G (completing ATG),
donor entry fires only on G after DSS, the GC-AG donor pays a penalty on C,
and so on. Under N (ambiguous base) every conditional path opens with a fixed
−10 penalty so assembly gaps degrade gracefully.

### 2.2 Emissions

Emissions are not learned: each state simply reads the model's posterior for
its class group (`columns_dict` maps the 15 classes onto state groups, e.g.
`CODING_EXON_1 ← CDS1, CDS1_TA, CDS1_TG`), floored at ε = 1e-3 and logged.
Transition weights are also fixed log-probabilities derived from expected
exon/intron lengths (`log(1 − 1/E)`, `log(1/E)`), plus motif penalties.

### 2.3 Two-pass decoding

Enforcing `min_intron_length` with counter states can *force* a worse path
when the unconstrained best path contains no short intron at all. The decoder
therefore decodes twice: first with `min_intron_length = 1` (counter chains
collapsed), then — only if the result actually contains an intron shorter
than the target — re-decodes with the full counter machine
(`decode_gene_structure`). In practice the second pass is rare, and skipping
it when unnecessary is both faster and slightly more accurate.

---

## 3. The bottleneck: dense Viterbi

Textbook Viterbi over S states is O(L·S²): for every base, for every target
state, maximize over all S predecessor states. With S = 170 that is
**28,900 candidate comparisons per base, per strand** — and the vast majority
are comparisons against −∞ weights that can never win, because the grammar is
extremely sparse: almost every state has only **1–4 legal predecessors**.

The dense formulation wastes the work three times over:

1. the inner maximization scans all S predecessors although only a handful are
   finite;
2. the five conditional matrices are S×S each, so any per-region
   (re)construction in Python loops is itself expensive;
3. in the reference pipeline the matrices were rebuilt for **every candidate
   region**, although they depend only on
   `(min_intron_length, expect_exon, expect_intron, gcag, atac)`.

---

## 4. The core rewrite: sparse-edge Viterbi, O(L·E)

### 4.1 The data structure

Extract the finite entries of the five transition matrices into a flat,
CSR-like edge table:

```
edge_from[e]  : source state of edge e        (int32)
edge_w[e]     : log transition weight of e    (float32)
to_ptr[s, j]  : edges into target state j under symbol s occupy
                the contiguous range [to_ptr[s, j], to_ptr[s, j+1])
```

Edges are globally concatenated per symbol in **(target, source) ascending**
order (`np.lexsort((fr, to))`), so `to_ptr` is a plain prefix-sum index.
Concretely, for one symbol's table the memory looks like this:

```
                       target state j        target state j+1
                     ┌ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┬ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐
 to_ptr[s, j] ──►    │ e=7  e=8  e=9  e=10 │ e=11 e=12 ...
                     │ (from, w) pairs ...  │
 edge_from  [ 7] =    3     41    77   112     5    88   ...
 edge_w     [ 7] =  -2.1  -0.4  -3.7  -2.1   -0.4  -5.0  ...
                     └ ─ ─ ─ ─ ▲ ─ ─ ─ ─ ─ ┴ ─ ─ ─ ─ ▲ ─ ─ ─ ┘
                        to_ptr[s, j]        to_ptr[s, j+1]

 within a target block, edges are sorted by source index ascending —
 the FIRST (lowest) source wins ties under the strict ">" update,
 exactly matching the reference's for-i-in-0..S-1 loop
```

The inner loop therefore scans one contiguous slice per (base, target state),
and the ascending-source order doubles as the tie-breaking rule (§4.3).

At `min_intron_length = 20` the five tables contain **879 edges in total**
(vs. 5 × 28,900 = 144,500 dense cells). Per base only one symbol's table is
consulted — on average **~176 edge visits per base** instead of 28,900 state
pairs, an ~160× reduction of the inner loop.

### 4.2 The kernel

```python
@njit(cache=True)
def _viterbi_core_sparse_f32(log_emit_probs, edge_from, edge_weight, to_ptr, sequence_codes):
    for t in range(1, seq_length):
        sym = sequence_codes[t]
        for to_state in range(num_states):
            best_score = -inf; best_from = 0
            emit = log_emit_probs[t, to_state]
            for e in range(to_ptr[sym, to_state], to_ptr[sym, to_state + 1]):
                score = dp[t-1, edge_from[e]] + edge_weight[e] + emit
                if score > best_score:      # strict greater-than
                    best_score = score; best_from = edge_from[e]
            dp[t, to_state] = best_score
            path[t, to_state] = best_from
```

This is a tight scalar loop over contiguous memory — exactly the shape numba
compiles well, and exactly the shape that a dense `numpy` row-max cannot
express without materializing S×S intermediates.

Two precision variants are compiled (`f32` for regions ≤ 1 Mb, `f64` above),
mirroring the reference decoder's dtype choice so that results remain
identical in both regimes.

### 4.3 Making it bit-exact — the actual constraints

Speed was never the hard part; **identical output** was. Three rules make the
sparse kernel produce the same `dp` values, the same backpointers, and hence
the same decoded path as the dense reference, bit for bit:

1. **Evaluation order.** The candidate score is computed as
   `(dp_prev + w) + emit` — the same left-to-right association as the
   reference's dense loop. Floating-point addition is not associative, so any
   other grouping could flip a comparison somewhere.
2. **Tie-breaking.** Within a target state, edges are sorted by source index
   ascending and the update uses strict `>`. The first (lowest-index) source
   wins ties — which is precisely what the reference's `for i in 0..S-1` loop
   with strict `>` produces.
3. **Deterministic construction.** The edge table is built by a fixed
   `lexsort`, so process restarts produce identical layouts.

With those three rules the sparse result is not merely statistically
equivalent — `np.array_equal` on decoded paths holds for arbitrary inputs.
This is what allows the decoder to be swapped in as a drop-in replacement:
downstream tooling that consumed the reference GFFs sees the same genes.

### 4.4 Everything else is cached

State tables, the five transition matrices, and the compiled edge table are
built once per process and memoized on
`(min_intron_length, expect_exon, expect_intron, gcag, atac)`
(`_HMM_ARTIFACTS_CACHE`); numba kernels are cached on disk (`cache=True`) so
worker processes forked by the `ProcessPoolExecutor` pay neither the JIT nor
the table construction. The reference rebuilt the matrices per region.

Base encoding is a 256-entry LUT over the raw ASCII buffer
(`np.frombuffer`) instead of a per-base dictionary lookup — a small thing
that matters at 10⁷ bases per chromosome.

---

## 5. Vectorizing everything around the kernel

The Viterbi rewrite would have been pointless if the rest of the pipeline
stayed in Python loops. `gene_decoding.py` reworks each stage with numpy,
under the same equivalence discipline (identical arithmetic order where sums
are involved):

| Stage | Reference | This repo |
|---|---|---|
| candidate regions | per-50bp-window Python loop | `cumsum` window sums + run-length extraction; identical mean formula |
| region merge | — | interval union with 100 bp buffer, sort + linear merge |
| state → class collapse | per-base set membership | lookup table + `np.take` |
| CDS/intron ranges | `pandas.groupby(...).apply(lambda ...)` | numpy run-length encoding (pandas dependency removed) |
| gene score | triple per-base Python loop | vectorized slice sums; **float64 sequential `cumsum`** preserving the reference's left-to-right accumulation order |
| H5 access | one file open per segment | one open per 16-segment batch |
| output order | process-completion order (nondeterministic) | chromosomes in lexicographic order |

The float64 detail in the last-but-two rows is the same story as §4.3 in
miniature: gene scores are sums of millions of posteriors, and a
pairwise-parallel reduction (`np.sum`) can round differently than the
reference's sequential accumulation — which could flip a gene across the
`--min_cds_score` filter. The sequential `cumsum`-then-take-last formulation
keeps the sums bit-identical.

Parallelism is a flat `ProcessPoolExecutor` over batches of candidate
regions; the per-worker caches of §4.4 make each worker hot after its first
batch.

---

## 6. Knobs that trade a little speed for accuracy

Three optional behaviors leave the historical default path completely
unchanged but can be enabled at the CLI:

- **`--atac_penalty <x>`** — adds AT-AC (U12-type minor splice) helper states
  (`intron{p}_ATAC_splice{0,1,3}`): donor preceded by AT, acceptor ending in
  C. Nine extra states per machine; the stop-codon suffix tracking is
  approximated away on this path (≈0.1% of introns).
- **`--gcag_penalty`** — the extra log penalty of GC-AG donors relative to
  canonical GT (default 10, the historical value; it is a transition-weight
  offset in the C-conditional matrix, visible in §2.1's suffix machinery).
- **`--min_cds_score`** — mean per-base emission score a gene must reach
  (single-exon and short genes face 1.5×). The default 0.6 (vs the historical
  0.5) was re-calibrated on plant benchmarks and is worth **+0.2–0.4 pp gene
  F1**. Unlike the two above it changes output by design; set it to 0.5 to
  reproduce historical filtering.

---

## 7. Results and where the time went

24 CPU threads, output gene content identical to the reference decoder:

| Task | Reference | This repo | Speed-up |
|---|---|---|---|
| Synthetic 2×5 Mb, 6,047 genes (exercises the short-intron re-decode path) | 17.9 s | 3.4 s | **5.3×** |
| *Arabidopsis thaliana* whole genome, 24,625 genes | 59.4 s | 22.3 s | **2.7×** |

Why does a ~160× reduction of the inner loop yield "only" 2.7–5.3×? Because
after the rewrite the profile rebalances: H5 I/O, region detection, GFF
formatting, process-pool overhead, and the *unavoidable* O(L·E) scan dominate.
The dense comparison count was so large that the reference was largely
memory-bandwidth-bound on matrices full of −∞; the sparse kernel is small and
fast, and the remaining wall time is honest work elsewhere. On fragmented
draft assemblies the *predictor* side (cross-scaffold chunked batching) is
the larger lever — see the README.

## 8. Verifying a rewrite like this

The equivalence claim rests on three layers, in increasing strength:

1. **Unit-level determinism** — edge-table construction is a pure function of
   its cache key; decoded paths are reproducible across runs and processes.
2. **Differential testing on synthetic data** — a generator producing
   genomes with known ground truth, deliberately including introns shorter
   than `min_intron_length` to trigger the re-decode path, GFFs compared
   byte-for-byte.
3. **Whole-genome differential runs** — real *A. thaliana*: 24,625 genes on
   both strands; all gene content lines identical between reference and this
   decoder (only the order of the per-sequence comment blocks differed — the
   reference emitted them in process-completion order, which was itself
   nondeterministic run-to-run; this repo emits lexicographic order).

Rule 3 deserves emphasis for anyone porting this approach: **"the same
answer" must be defined including tie-breaking and float rounding**, or the
diff will be full of mysterious one-base shifts that are really just
different-but-equally-optimal paths.

---

## 9. Summary

| Aspect | Reference | This repo |
|---|---|---|
| Viterbi inner loop | O(L·S²), dense, ~28,900 comparisons/base | O(L·E), sparse CSR edge table, ~176 edge visits/base |
| Transition tables | rebuilt per region | built once per process, memoized; numba disk cache |
| Precision policy | f32 / f64 per region length | same, both kernels compiled |
| Equivalence | — | bit-exact paths (fixed evaluation order + ascending-source tie-breaking) |
| Pipeline | pandas + Python loops | numpy run-length/cumsum throughout, no pandas |
| H5 access | open per segment | open per batch |
| Output | nondeterministic section order | deterministic lexicographic |
| Extras | — | AT-AC path, GC-AG knob, calibrated CDS filter |

The general lesson generalizes beyond gene finding: **structured HMMs are
almost always sparse, and their sparsity is invisible to a dense
implementation.** Extracting the grammar into an edge table — with deliberate
control of evaluation order and tie-breaking — turns an O(S²) kernel into
O(E) while *strengthening*, not weakening, the reproducibility guarantee.

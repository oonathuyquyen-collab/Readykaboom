# Unified TMIL — Account- & Transaction-Level Ethereum Phishing Detection

> One forward pass. Two granularities. An audited benchmark and an *honest* localization study.

Unified TMIL detects Ethereum phishing at **two levels with a single model**: (i) whether an
**account** is a scammer, and (ii) **which transactions** inside that account's history are the
fraudulent ones. Instead of proposing a new backbone, we **preserve** the published
[BERT4ETH](https://github.com/git-disl/BERT4ETH) transaction embeddings and a temporal
Multiple-Instance-Learning (TMIL) read-out (TCN → Gated-Attention MIL → Bag Aggregation → MLP),
and show how far a careful **unification** and an **honest evaluation protocol** can go.

---

## Highlights

- **Single-weight, two-head architecture.** One shared encoder feeds two attention read-out heads
  that, in *one forward pass*, produce a bag-level account decision (soft head) and a
  direction-aware transaction ranking (localization head).
- **Key modification (backbone-preserving).** We replace C-TMIL's **hard outbound-only mask** with
  the existing BERT4ETH **IN/OUT embedding**, letting the model *learn* directional importance
  rather than assuming outbound-only relevance. No new backbone, no new localization head.
- **Audited benchmark + contract-mediated relabeling.** A receipt-tracing protocol
  (`transferFrom`, Seaport `OrderFulfilled`, internal ETH) lifts clean account-level ground truth
  from **83.4% → 92.7%**, yielding **292** deduplicated scammer EOAs across three mechanisms.
- **Honest evaluation.** Cluster-aware bootstrap CIs, by-source negative breakdowns, activity
  stratification, and cross-mechanism / temporal OOD splits.
- **A finding the field needs.** A content-aware ranker beats a recency prior at Hit@1
  (**0.832 vs 0.693**, paired bootstrap *p* = 0.002), but a controlled ablation shows the **learned
  attention contributes essentially nothing** — removing it leaves Hit@1 unchanged (0.802) and MRR
  slightly higher. We argue *against* treating attention as localization.

## Headline results

| Task | Metric | Result |
|---|---|---|
| Account (cross-domain, combined) | AUC | **0.984** |
| Account (hard DeFi/KOL negatives, n=80) | AUC | 0.725 (95% CI [0.658, 0.791]) |
| Localization (content-aware) | Hit@1 / MRR | **0.832 / 0.880** |
| Localization (recency prior) | Hit@1 / MRR | 0.693 / 0.799 |
| Attention-only localization | Hit@1 | 0.416 (below recency) |

> We are deliberate: simple mean-pooling and BERT4ETH match or exceed us on hard-negative AUC, so
> our account-level contribution is the **unification** (we are the only entry that *also*
> localizes), not a new account-level state of the art.

---

## Repository layout

```
PhishUnified/
├── src/                 # 17 canonical scripts (pipeline + model + baselines + figures)
│   ├── step1_audit.py        # feasibility audit of PTXPhish ground truth
│   ├── step2_crawl.py        # Etherscan history crawler
│   ├── step3_build_bags.py   # bag construction (1 account = 1 bag, L=100)
│   ├── step4_final.py        # account-level evaluation
│   ├── step7_unified.py      # Unified TMIL model (shared encoder + 2 heads)
│   ├── step14_seaport.py     # contract-mediated relabeling protocol
│   ├── sota.py / step16_sota.py   # SOTA + MIL pooling baselines (shared encoder)
│   ├── step21_locwin2.py / loc_ablate_final.py  # localization + attention ablation
│   ├── step9_figures.py / step13_arch.py / fig_loc_honest.py  # figures
│   └── etherscan.py, vocab_def.py, step8b/step15  # utilities
├── data/                # processed datasets (≈17 MB, GitHub-friendly)
│   ├── bert4eth/        # train/test bags + vocab (in-domain training data)
│   ├── ptx_bags.pkl, defi_hard_bags.pkl, normal_eoa_neg.pkl   # test bags
│   ├── seaport_gt.json, step2_gt_map.json, nft_scammers.json  # ground truth / traces
│   └── PTXPhish_source/ # upstream PTXPhish address lists (attribution preserved)
├── results/             # all JSON/NPY result artifacts behind every paper number
├── figures/             # paper figures (architecture, account, localization, OOD)
├── paper/               # paper_en.{tex,pdf}  +  paper_vi.{tex,pdf}
├── slides/              # presentation decks: slides_en.pdf + slides_vi.pdf (23 slides each)
├── PROGRESS.md          # full research log (30+ checkpoints, incl. self-corrections)
├── REPORT.md            # condensed technical report
├── requirements.txt · LICENSE · CITATION.cff
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Reproduce the paper numbers (from cached artifacts)
The `results/` and `data/` artifacts let you regenerate tables/figures without re-crawling:
```bash
python src/step4_final.py        # account-level metrics  -> results/step4_results.json
python src/step7_unified.py      # train Unified TMIL (single forward pass)
python src/sota.py               # baselines on the shared encoder
python src/loc_ablate_final.py   # localization + attention-contribution ablation
python src/fig_loc_honest.py     # regenerate the honest localization figure
```

### Rebuild from scratch (optional, requires an Etherscan API key)
```bash
export ETHERSCAN_API_KEY=...           # used by src/etherscan.py
python src/step1_audit.py              # audit PTXPhish ground-truth feasibility
python src/step2_crawl.py              # crawl full account histories
python src/step14_seaport.py           # contract-mediated relabeling (83.4% -> 92.7%)
python src/step3_build_bags.py         # build bags (L=100, last-tx artifact removed)
```

## Method in one paragraph

Each account is a **bag** of its transactions (single fixed window, L=100). Following BERT4ETH,
every transaction is encoded by counterparty + IN/OUT + value + count + time-delta embeddings and
contextualized by the TMIL TCN — **unchanged**. A soft gated-attention head (Head-C) pools the bag
for the account-level objective; a second head (Head-L) ranks transactions for localization. The
loss is `L = BCE(Head-C) + λ·BCE(Head-L) + β·L_contrast`. We empirically verified that reading
localization *post-hoc* from the soft head underperforms a separately trained masked model — Head-L
must receive its own gradient — so the model is genuinely unified (one set of weights, one forward
pass) rather than two stitched models.

## Datasets

- **Train (in-domain):** BERT4ETH phishing + normal EOA bags (`data/bert4eth/`).
- **Test, account-level:** PTXPhish cashier/recipient scammer EOAs vs. **hard** benign senders
  (active KOL/DeFi) + held-out Normal EOAs.
- **Test, transaction-level:** PTXPhish transaction hashes; contract-mediated cases relabeled to the
  execution receipt (`transferFrom`, Seaport `OrderFulfilled`).

Full histories are crawled via Etherscan. Positives are deduplicated to **unique scammer EOAs** and
used as **zero-shot** test bags (the model trains only on in-domain BERT4ETH data → no train/test
leakage).

## Honest limitations (also in the paper)

- **Small hard-negative pool** (80 DeFi/KOL accounts) → non-trivial run-to-run variance; we report
  cluster-aware CIs and keep PTXPhish as-is for comparability rather than padding with easy negatives.
- **Positional bias:** positive windows end at the detection cutoff, making recency a strong prior;
  we neutralize the trivial last-tx case and benchmark all methods against recency.
- **Attention is not faithful for localization** (shown by ablation) → we make no explanatory claim
  for attention weights.
- **Coverage:** clean localization GT covers 92.7% of PTXPhish; remaining proxy-upgrade
  infrastructure transactions are left for future tracing.

## Upstream credits

This work **preserves and builds on** the following; please cite and comply with their licenses:
- **BERT4ETH** (Hu et al., WWW'23) — transaction embeddings / backbone.
- **C-TMIL / TMIL** — temporal MIL read-out (TCN → Gated-Attention MIL → Bag Agg → MLP).
- **PTXPhish** (BlockSec) — payload-based transaction phishing dataset.

## Citation

See `CITATION.cff`. Papers: `paper/paper_en.pdf` (English) and `paper/paper_vi.pdf` (Vietnamese).

## License

MIT (this repository's code and derived artifacts). Upstream datasets/backbones retain their own
licenses — see `LICENSE`.

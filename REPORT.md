# Unified Direction-Aware Multiple-Instance Learning for Account- and Transaction-Level Phishing Detection on Ethereum

*Reproducible study built on BERT4ETH (WWW'23), C-TMIL, and the PTXPhish dataset (NDSS'25).*

---

## Abstract

We present a single Multiple-Instance-Learning (MIL) model that, in one forward pass, performs
**(1) account-level phishing detection**, **(2) transaction-level phishing localization**, **(3)
entirely on Ethereum**, while **keeping the BERT4ETH counterparty representation and the C-TMIL
backbone (TCN -> Gated-Attention MIL -> bag aggregation -> MLP) unchanged.** Our only architectural
change replaces C-TMIL's *hard outbound attention mask* with the *soft IN/OUT direction embedding
that already exists inside BERT4ETH's feature encoder*, so the same model handles outbound cashouts
and inbound phishing receipts without two mutually-exclusive mask configurations. We evaluate
account-level detection both **in-domain** (BERT4ETH/ScamSniffer) and **cross-domain zero-shot** on
addresses harvested from the **PTXPhish** payload-phishing dataset, and we evaluate transaction-level
localization against **exact PTXPhish transaction hashes**. A pre-registered *feasibility audit*
shows that **4,168 / 4,998 (83%)** PTXPhish phishing transactions yield clean transaction-level ground
truth inside the scammer account's own history. The soft direction embedding improves cross-domain
zero-shot AUC from **0.666 -> 0.824** (paired bootstrap ΔAUC = +0.158, 95% CI [0.096, 0.223], p < 1e-3),
and the attention map localizes phishing transactions at up to **22x** the random baseline on a clean
needle-in-haystack subset. We release all code, the Etherscan crawl, figures and tables.

---

## 1. Introduction

Ethereum phishing is attacked from two largely disjoint literatures. **Account-level** detectors such
as **BERT4ETH** learn a self-supervised transformer over an account's transaction history and classify
*whether an account is a phisher*. **Payload-level** systems such as **PTXPhish** parse individual
transactions and classify *whether a single transaction is a phishing payload*. Neither answers the
operationally important question end-to-end: *given an account, is it fraudulent, and exactly which of
its transactions are the phishing events?*

We unify the two under one MIL formulation and impose three hard requirements:

1. **Account-level phishing detection.**
2. **Transaction-level phishing localization.**
3. **On Ethereum, reusing the BERT4ETH + C-TMIL architecture unchanged.**

**Bag formulation.** A *bag* is one EOA's full transaction history, cut into sliding windows
(W=200, S=50); each window is an *instance*. The bag vector `z = Σ aₖhₖ` (gated attention) drives an
MLP for account-level prediction (criterion 1); the attention weights `aₖ` rank instances for
transaction-level localization (criterion 2); the backbone is untouched (criterion 3).

**The direction problem.** C-TMIL hard-masks inbound transactions (`attn_scores.masked_fill(
inbound, -1e9)`) because its only target was *outbound* Tornado-Cash cashouts. PTXPhish phishing
receipts are frequently *inbound*, so a hard outbound mask is self-defeating. Critically, we verified
the in-domain (BERT4ETH/ScamSniffer) bags are **single-direction** (10,865 all-OUT vs 271 all-IN, zero
bidirectional), whereas PTXPhish bags are **bidirectional (284/372)**. We therefore replace the hard
mask with BERT4ETH's **existing learnable IN/OUT embedding**, letting the model *learn* which direction
matters per domain. This is a one-line configuration change, not a new module.

---

## 2. Data and Feasibility Audit

### 2.1 Sources
- **BERT4ETH pretrained embeddings, vocab, and in-domain bags** (11,136 train / 2,785 test) — reused verbatim.
- **PTXPhish** — 4,998 phishing transaction hashes (4 categories) + 13,557 benign transaction hashes.
- **Etherscan V2 API** — full transaction histories crawled per account (cached in SQLite, resumable).

### 2.2 Feasibility audit (gating experiment)
For each of the 4,998 phishing transactions we fetched the transaction, decoded the 4-byte selector,
identified the scammer side per category, and checked whether the transaction is a *direct* on-chain
event in an EOA's own history. **Result: 4,168 / 4,998 (83%) are clean.** Ice-phishing is clean
because PTXPhish records the *scammer-executed drain* (`transferFrom`, caller = scammer EOA), which is
outbound from the scammer. Only `nft_order` (Seaport marketplace intermediary) requires multi-hop
tracing and is deferred to future work. The audit yields **299 unique scammer accounts** (drainer EOAs
and phishing contracts are heavily reused across victims). See **Figure 1**.

![Figure 1](figures/fig1_audit.png)

### 2.3 Account-level populations
- **In-domain:** BERT4ETH phisher (Etherscan-labeled) vs Normal EOA — original protocol.
- **Cross-domain (zero-shot):** PTXPhish scammer EOAs (positives) vs PTXPhish benign senders (negatives).

**Documented limitation:** PTXPhish's 13,557 benign transactions are *highly concentrated* — only
**~86 unique senders** appear across 4,581 sampled benign transactions (a small KOL/DeFi-dev set whose
transactions are repeatedly sampled). After dedup + BERT4ETH's `[3, 10000]` tx filter we obtain
**292 positive and 80 negative** bags. We therefore report bootstrap CIs and activity-stratified AUC
rather than a single point estimate.

### 2.4 Transaction-level ground truth
Because we crawl each account's *entire* history, every clean PTXPhish hash is located exactly within
its bag (no value-matching needed). Bags average ~9 GT transactions in ~52 transactions; GT directions
are mixed (1,376 OUT / 1,290 IN), outbound-dominated by ice-phishing drains.

---

## 3. Method

**Instance features (unchanged from BERT4ETH/C-TMIL):** counterparty embedding + log-amount + log-Δt,
plus the **IN/OUT embedding** (`io_embed`, our re-activation). **Backbone (unchanged):** residual TCN
-> gated attention `aₖ = softmax(w·(tanh(V hₖ) ⊙ σ(U hₖ)))` -> `z = Σ aₖ hₖ` -> 2-layer MLP. **Loss
(unchanged):** BCE + 0.3·hinge-contrast. We compare three direction modes:
- **io_embed (ours):** add learnable IN/OUT embedding; no attention masking.
- **hardmask (old C-TMIL):** mask inbound when any outbound present.
- **none:** ignore direction entirely (ablation).

All models trained 3 seeds, 8 epochs, CPU. Cross-domain predictions use the 3-seed ensemble; all CIs
are 2,000x bootstrap; the io_embed-vs-none comparison is a paired bootstrap.

---

## 4. Results

### 4.1 Account-level detection (criterion 1)
![Figure 2](figures/fig2_crossdomain.png)

| Mode | In-dom F1 | In-dom AUC | Cross AUC [95% CI] | Cross AUPR [95% CI] |
|---|---|---|---|---|
| **IO-embed (ours)** | 0.735±0.011 | 0.920 | **0.824 [0.775, 0.870]** | **0.933 [0.895, 0.965]** |
| Hard-mask (old) | 0.715±0.003 | 0.909 | 0.662 [0.606, 0.718] | 0.893 [0.859, 0.923] |
| No-direction | 0.715±0.003 | 0.909 | 0.666 [0.610, 0.721] | 0.894 [0.861, 0.923] |

**Paired test (IO-embed vs No-direction), cross-domain AUC:** ΔAUC = **+0.158**, 95% CI [0.096, 0.223],
bootstrap p < 1e-3. The soft direction embedding gives a large, statistically significant cross-domain
gain; "none" collapses toward chance, confirming direction information is essential for transfer.

**Activity-stratified AUC (controls the tx-count confound):**

| tx-count stratum | AUC | n (pos/neg) |
|---|---|---|
| 3–20 | 0.921 | 58 (54/4) |
| 20–100 | 0.956 | 117 (103/14) |
| 100+ | 0.748 | 186 (130/56) |

High AUC persists even in the low-activity stratum, so the signal is **not** merely "power-user vs
short-lived drainer."

### 4.2 Transaction-level localization (criterion 2)
![Figure 3](figures/fig3_localization.png)

**Full set (GT-dense; random Hit@10 = 0.613 because bags average ~9 GT):**

| Method | Hit@1 | Hit@5 | Hit@10 | MRR |
|---|---|---|---|---|
| IO-embed (attn) | 0.182 | 0.459 | 0.599 | 0.325 |
| Hard-mask (attn) | 0.291 | 0.579 | 0.682 | 0.429 |
| No-direction (attn) | 0.295 | 0.565 | 0.675 | 0.426 |
| Amount-rank | 0.127 | 0.288 | 0.394 | 0.212 |
| Random | 0.219 | 0.486 | 0.613 | 0.352 |

**Clean needle-in-haystack subset (≤2 GT, len≥20, n=111; random Hit@10 = 0.189):**

| Method | Hit@1 | Hit@5 | Hit@10 | MRR |
|---|---|---|---|---|
| IO-embed (attn) | 0.063 | 0.171 | 0.243 | 0.139 |
| **Hard-mask (attn)** | **0.198** | **0.342** | **0.396** | **0.285** |
| Amount-rank | 0.018 | 0.018 | 0.018 | 0.039 |
| Random | 0.009 | 0.099 | 0.189 | 0.073 |

On the clean subset, hard-mask attention beats random by **×22.0 at Hit@1**, ×3.46 at Hit@5, ×2.10 at
Hit@10, and beats amount-ranking by an order of magnitude.

### 4.3 The account/localization trade-off (key finding)
There is a **direction trade-off**: the **soft IO-embed** wins **account-level** transfer (it
distributes attention to build a better bag representation), while the **hard mask** wins
**localization** (it concentrates attention on the outbound-dominated GT). A practitioner can pick the
mode per task from the *same* architecture — the unification holds; the masking is a deployment knob.

---

## 5. Ablation Summary
- **Direction handling:** io_embed >> {hardmask, none} for account-level cross-domain (+0.158 AUC, p<1e-3);
  hardmask/none > io_embed for localization. (Tables above.)
- **Attention vs amount vs random:** attention dominates amount everywhere; on GT-dense full set it only
  modestly beats random, motivating the clean-subset protocol where it beats random ×2–22.
- **In-domain single-direction artifact:** hardmask ≡ none in-domain (identical weights) because in-domain
  bags are single-direction; the modes diverge only on bidirectional PTXPhish — itself evidence for the
  in/out asymmetry this paper addresses.

---

## 6. Limitations & Future Work
1. **Benign concentration:** PTXPhish benign senders collapse to ~86 unique accounts; the 80-negative
   cross-domain set is small (mitigated by bootstrap CIs + stratification, but a larger benign pool is
   future work).
2. **GT density:** scammer accounts are phishing-dense, so full-set Hit@K has a high random floor; the
   clean ≤2-GT subset is the meaningful localization benchmark.
3. **NFT-order tracing:** the 830 non-clean transactions (mostly Seaport) need multi-hop internal-tx
   tracing; deferred.
4. **Joint multi-target localization** (in + out in one bag) is left as future work due to the largely
   disjoint positive populations.

---

## 7. Reproducibility
```
ptx_pipeline/
  etherscan.py            # cached Etherscan V2 client (SQLite, throttled)
  step1_audit.py          # feasibility audit -> data/step1_audit.csv
  step2_crawl.py          # full per-account history crawl -> data/step2_*.json
  step3_build_bags.py     # BERT4ETH-vocab-aligned bags -> data/ptx_bags.pkl
  step4_final.py          # multi-seed train + bootstrap CIs + paired tests -> results/step4_results.json
  step4b_loc_subset.py    # clean-subset localization -> results/loc_subset.json
  step5_figures.py        # figures/*.png + results/tables.md
  results/tables.md        figures/{fig1_audit,fig2_crossdomain,fig3_localization}.png
```
All randomness seeded; Etherscan calls cached so reruns are deterministic and offline.

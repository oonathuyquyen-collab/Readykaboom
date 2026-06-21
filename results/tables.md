### Table 0: Dataset composition

| Split | Population | #bags |
|---|---|---|
| In-domain train | BERT4ETH phishers + Normal EOA | 11136 |
| In-domain test | BERT4ETH phishers + Normal EOA (held-out) | 2785 |
| Cross-domain pos | PTXPhish scammer/cashier EOA | 292 |
| Cross-domain neg (PTX) | PTXPhish benign senders (KOL/DeFi) | 80 |
| Cross-domain neg (Normal EOA) | BERT4ETH Normal EOA pool (held-out) | 1324 |
| Cross-domain total neg | combined | 1404 |

### Table 1: Account-level detection (3 seeds; mean +/- std in-domain; bootstrap 95% CI cross-domain)

| Mode | In-dom F1 | In-dom AUC | Cross AUC [95% CI] | Cross AUPR [95% CI] |
|---|---|---|---|---|
| IO-embed (ours) | 0.735±0.011 | 0.920 | 0.990 [0.986,0.993] | 0.932 [0.897,0.965] |
| Hard-mask (old) | 0.715±0.003 | 0.909 | 0.966 [0.958,0.973] | 0.850 [0.813,0.884] |
| No-direction | 0.715±0.003 | 0.909 | 0.966 [0.958,0.973] | 0.852 [0.815,0.886] |

**Paired test (IO-embed vs No-direction), cross-domain AUC:** ΔAUC = +0.024, 95% CI [0.017, 0.030], bootstrap p = 0.0000.

### Table 1b: Cross-domain by NEGATIVE source (positives fixed = PTXPhish scammers)

| Mode | vs PTX-benign (hard) AUC/AUPR | vs Normal-EOA AUC/AUPR | vs combined AUC/AUPR |
|---|---|---|---|
| IO-embed (ours) | 0.824/0.933 | 1.000/0.999 | 0.990/0.932 |
| Hard-mask (old) | 0.662/0.893 | 0.984/0.935 | 0.966/0.850 |
| No-direction | 0.666/0.894 | 0.984/0.936 | 0.966/0.852 |

### Table 2: Activity-stratified cross-domain AUC (IO-embed), controlling tx-count confound

| tx-count stratum | AUC | n (pos/neg) |
|---|---|---|
| 3-20 | 1.000 | 1272 (54/1218) |
| 20-100 | 0.994 | 207 (103/104) |
| 100+ | 0.814 | 206 (130/76) |

### Table 3: Transaction-level localization, full set (GT-dense)

| Mode/Method | Hit@1 | Hit@5 | Hit@10 | MRR | n |
|---|---|---|---|---|---|
| IO-embed (ours) (attn) | 0.182 | 0.459 | 0.599 | 0.325 | 292 |
| Hard-mask (old) (attn) | 0.291 | 0.579 | 0.682 | 0.429 | 292 |
| No-direction (attn) | 0.295 | 0.565 | 0.675 | 0.426 | 292 |
| Amount-rank | 0.127 | 0.288 | 0.394 | 0.212 | 292 |
| Random | 0.219 | 0.486 | 0.613 | 0.352 | 292 |

### Table 4: Localization on clean needle-in-haystack subset (<=2 GT, len>=20)

Subset n = 111 accounts. Random baseline is low here, so lift is meaningful.

| Method | Hit@1 | Hit@5 | Hit@10 | MRR |
|---|---|---|---|---|
| IO-embed (ours) (attn) | 0.063 | 0.171 | 0.243 | 0.139 |
| Hard-mask (old) (attn) | 0.198 | 0.342 | 0.396 | 0.285 |
| No-direction (attn) | 0.198 | 0.342 | 0.396 | 0.283 |
| Amount-rank | 0.018 | 0.018 | 0.018 | 0.039 |
| Random | 0.009 | 0.099 | 0.189 | 0.073 |

**Hard-mask attention lift over random:** Hit@1 ×22.0, Hit@5 ×3.455, Hit@10 ×2.095.

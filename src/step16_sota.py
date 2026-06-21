"""STEP 16 - Evaluate the 7 OFFICIAL-REPO SOTA baselines under our protocol.
Account-level: in-domain (test_bags) P/R/F1/AUC + cross-domain zero-shot AUC/AUPR with
  by-source breakdown (ptx_benign hard negs vs normal_eoa) and bootstrap 95% CI.
Transaction-level (MIL methods only): localization Hit@1/5/10 + MRR vs ground-truth, seed-averaged.
Writes results/sota_results.json (merged into paper tables by step9_paper).
"""
import os, sys, json, pickle, numpy as np
sys.path.insert(0, '.')
import vocab_def; sys.modules['__main__'].Vocab = vocab_def.Vocab
from sklearn.metrics import roc_auc_score, average_precision_score
from step4_final import load, SRC, DATA, RES, boot_ci
from step7_unified import acc_metrics, loc_metrics
import sota
import torch as _t; _t.set_num_threads(4)
SEEDS = [42, 1]

train_bags = load("train_bags.pkl"); test_bags = load("test_bags.pkl")
vocab = pickle.load(open(os.path.join(SRC, "vocab.pkl"), "rb")); V = len(vocab.token2id)
ptx = pickle.load(open(os.path.join(DATA, "ptx_bags.pkl"), "rb"))
norm = pickle.load(open(os.path.join(DATA, "normal_eoa_neg.pkl"), "rb"))
pos = [b for b in ptx if b["label"] == 1]; pb = [b for b in ptx if b["label"] == 0]
allb = pos + pb + norm
y = np.array([b["label"] for b in allb]); src = np.array([b.get("source", "ptx_benign") for b in allb])
y_test = np.array([b["label"] for b in test_bags]); gt = [b["gt_idx"] for b in pos]

def cross_block(p):
    auc = float(roc_auc_score(y, p)); idx = np.arange(len(y))
    def f(rng):
        s = rng.choice(idx, len(idx), replace=True)
        return roc_auc_score(y[s], p[s]) if len(set(y[s])) == 2 else auc
    bysrc = {}
    for sn in ["ptx_benign", "normal_eoa"]:
        sel = (y == 1) | ((y == 0) & (src == sn))
        bysrc[sn] = {"auc": float(roc_auc_score(y[sel], p[sel])),
                     "aupr": float(average_precision_score(y[sel], p[sel])),
                     "n_neg": int(((y == 0) & (src == sn)).sum())}
    return {"auc": auc, "auc_ci": boot_ci(f), "aupr": float(average_precision_score(y, p)),
            "by_source": bysrc}

def acc_eval(trainer, predictor):
    ind = []; cross = []
    for s in SEEDS:
        m = trainer(s); ind.append(acc_metrics(y_test, predictor(m, test_bags))); cross.append(predictor(m, allb))
    indm = {k: [float(np.mean([d[k] for d in ind])), float(np.std([d[k] for d in ind]))] for k in ind[0]}
    return indm, cross_block(np.mean(cross, axis=0))

def mil_eval(trainer, **kw):
    ind = []; cross = []; attn_sum = None
    for s in SEEDS:
        m = trainer(s)
        pt, _ = sota.pred_mil(m, test_bags); ind.append(acc_metrics(y_test, pt))
        pc, _ = sota.pred_mil(m, allb); cross.append(pc)
        _, ap = sota.pred_mil(m, pos); pa = [np.array(a) for a in ap]
        attn_sum = pa if attn_sum is None else [x + y2 for x, y2 in zip(attn_sum, pa)]
    indm = {k: [float(np.mean([d[k] for d in ind])), float(np.std([d[k] for d in ind]))] for k in ind[0]}
    at = [a / len(SEEDS) for a in attn_sum]
    lm = loc_metrics(at, gt)
    return indm, cross_block(np.mean(cross, axis=0)), {k: v for k, v in lm.items() if k != "_bests"}, at

R = {"account": {}, "transaction": {}, "seeds": SEEDS}
print("=== ACCOUNT-LEVEL SOTA ===", flush=True)
for nm, Cls in [("BERT4ETH", sota.BERT4ETH), ("ZipZap", sota.ZipZap), ("TSGN", sota.TSGN)]:
    i, c = acc_eval(lambda s, C=Cls: sota.train_acc(C, train_bags, V, s), sota.pred_acc)
    R["account"][nm] = {"in_domain": i, "cross_domain": c}; print(f"  {nm}: cross AUC {c['auc']:.3f} (ptx {c['by_source']['ptx_benign']['auc']:.3f})", flush=True)
i, c = acc_eval(lambda s: sota.train_acc(sota.LMAE4Eth, train_bags, V, s, expert=True, contrast=True),
                lambda m, b: sota.pred_acc(m, b, expert=True))
R["account"]["LMAE4Eth"] = {"in_domain": i, "cross_domain": c}; print(f"  LMAE4Eth: cross AUC {c['auc']:.3f} (ptx {c['by_source']['ptx_benign']['auc']:.3f})", flush=True)

print("=== TRANSACTION-LEVEL MIL SOTA ===", flush=True)
mil_attn = {}
for nm, Cls, clam in [("GatedMIL", sota.GatedMIL, False), ("TransMIL", sota.TransMIL, False), ("CLAM", sota.CLAM, True)]:
    i, c, lm, at = mil_eval(lambda s, C=Cls, cl=clam: sota.train_mil(C, train_bags, V, s, clam=cl))
    R["account"][nm] = {"in_domain": i, "cross_domain": c}
    R["transaction"][nm] = lm; mil_attn[nm] = [a.tolist() for a in at]
    print(f"  {nm}: cross AUC {c['auc']:.3f} | loc hit@1 {lm['hit@1']:.3f} hit@5 {lm['hit@5']:.3f} mrr {lm['mrr']:.3f}", flush=True)

# low-density localization subset (honest needle-in-haystack), same definition as step7_unified
keep = [k for k, b in enumerate(pos) if len(gt[k]) <= 2 and b["length"] >= 20]
R["transaction_subset"] = {"keep_n": len(keep)}
for nm in mil_attn:
    rl = [mil_attn[nm][k] for k in keep]; gg = [gt[k] for k in keep]
    mm = loc_metrics(rl, gg); R["transaction_subset"][nm] = {k: v for k, v in mm.items() if k != "_bests"}

json.dump(R, open(os.path.join(RES, "sota_results.json"), "w"), indent=2)
print("SAVED results/sota_results.json", flush=True)

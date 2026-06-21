"""step22_headL_strong.py -- Make head-L attention GENUINELY contribute to localization.
Keeps BERT4ETH+TMIL backbone unchanged. Adds ONLY weakly-supervised training signals on
head-L (NO transaction-level GT used in training):
  (a) entropy sharpening  : penalise diffuse attention -> peaked, decisive weights
  (b) instance-max consist: the single highest-attended tx must itself classify the bag label
Evaluates attention-only localization with the SAME artifact-removed protocol as step8b.
Writes results/headL_strong.json and results/unified_attn_strong.json (best config).
"""
import sys, os, json, pickle, time, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import vocab_def
sys.modules['__main__'].Vocab = vocab_def.Vocab
from step4_final import collate, load, iterate, DATA, SRC, RES
from step7_unified import UnifiedTMIL
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support

SEEDS = [42, 1]; DEV = "cpu"

def train_strong(train_bags, V, seed, lam=0.5, beta=0.0, gamma=0.0, epochs=6, bs=64, lr=1e-3):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    m = UnifiedTMIL(V).to(DEV); opt = torch.optim.Adam(m.parameters(), lr=lr); rng = random.Random(seed)
    for ep in range(epochs):
        m.train()
        for batch in iterate(train_bags, bs, rng):
            ids, io, hc, mask, y = collate(batch)
            h = m.enc(ids, io, hc)
            logitC, aC = m.headC(h, mask)
            logitL, aL = m.headL(h, mask, io=io, outmask=True)
            pC = torch.sigmoid(logitC); pL = torch.sigmoid(logitL)
            loss = F.binary_cross_entropy(pC, y) + lam * F.binary_cross_entropy(pL, y)
            pm, nm = y == 1, y == 0
            if pm.sum() > 0 and nm.sum() > 0:
                loss = loss + 0.3 * F.relu(0.3 - (pC[pm].mean() - pC[nm].mean()))
            if beta > 0:
                Hh = -(aL * torch.log(aL + 1e-9)).sum(dim=1)
                loss = loss + beta * Hh.mean()
            if gamma > 0:
                B = h.shape[0]; top = aL.argmax(dim=1)
                h_top = h[torch.arange(B), top]
                inst = torch.sigmoid(m.headL.clf(h_top).squeeze(-1))
                loss = loss + gamma * F.binary_cross_entropy(inst, y)
            opt.zero_grad(); loss.backward(); opt.step()
    return m

@torch.no_grad()
def attn_on(m, bags, bs=128):
    m.eval(); al = []
    for i in range(0, len(bags), bs):
        b = bags[i:i+bs]; ids, io, hc, mask, y = collate(b)
        h = m.enc(ids, io, hc)
        _, aL = m.headL(h, mask, io=io, outmask=True)
        for j, bb in enumerate(b): al.append(aL[j, :bb["length"]].cpu().numpy().astype(float))
    return al

@torch.no_grad()
def pred_acct(m, bags, bs=128):
    m.eval(); pc = []
    for i in range(0, len(bags), bs):
        b = bags[i:i+bs]; ids, io, hc, mask, y = collate(b)
        h = m.enc(ids, io, hc); logitC, _ = m.headC(h, mask)
        pc.extend(torch.sigmoid(logitC).tolist())
    return np.array(pc)

def usable(b):
    n = b["length"]; gt = [g for g in b["gt_idx"] if g < n - 1]
    return n - 1, gt
def best_rank(scores, gt):
    order = np.argsort(-np.asarray(scores)); rankpos = {p: r for r, p in enumerate(order)}
    return min(rankpos[g] for g in gt)
def hitk(pos, attn, idxs, ks=(1, 5, 10)):
    bests = []
    for i in idxs:
        nc, gt = usable(pos[i])
        if nc <= 0 or not gt: continue
        bests.append(best_rank(attn[i][:nc], gt))
    out = {f"hit@{k}": float(np.mean([b < k for b in bests])) for k in ks}
    out["mrr"] = float(np.mean([1.0 / (b + 1) for b in bests])); out["n"] = len(bests)
    return out, bests
def recency_scores(pos): return [np.arange(b["length"], dtype=float) for b in pos]

def main():
    train_bags = load("train_bags.pkl"); test_bags = load("test_bags.pkl")
    vocab = pickle.load(open(os.path.join(SRC, "vocab.pkl"), "rb")); V = len(vocab.token2id)
    ptx = pickle.load(open(os.path.join(DATA, "ptx_bags.pkl"), "rb"))
    norm = pickle.load(open(os.path.join(DATA, "normal_eoa_neg.pkl"), "rb"))
    pos = [b for b in ptx if b["label"] == 1]; pb = [b for b in ptx if b["label"] == 0]
    allb = pos + pb + norm
    y = np.array([b["label"] for b in allb]); src = np.array([b.get("source", "ptx_benign") for b in allb])
    y_test = np.array([b["label"] for b in test_bags])
    allidx = list(range(len(pos)))
    needle = [i for i in allidx if len([g for g in pos[i]["gt_idx"] if g < pos[i]["length"] - 1]) <= 2 and pos[i]["length"] >= 20]
    rec = recency_scores(pos)
    rec_full, _ = hitk(pos, rec, allidx); rec_ndl, _ = hitk(pos, rec, needle)
    configs = {
        "baseline":     dict(beta=0.0,  gamma=0.0),
        "sharpen":      dict(beta=0.02, gamma=0.0),
        "instance":     dict(beta=0.0,  gamma=0.5),
        "sharpen+inst": dict(beta=0.02, gamma=0.5),
    }
    R = {"protocol": "artifact_removed; attention-only; weakly-supervised",
         "recency_full": rec_full, "recency_needle": rec_ndl,
         "counts": {"pos": len(pos), "needle": len(needle), "ptx_neg": len(pb), "normal_eoa_neg": len(norm)},
         "configs": {}}
    best_name, best_h1, best_attn = None, -1, None
    for name, kw in configs.items():
        t = time.time(); attn_sum = None; ind = []; cross = []
        for s in SEEDS:
            m = train_strong(train_bags, V, s, **kw)
            a = attn_on(m, pos)
            attn_sum = a if attn_sum is None else [x + z for x, z in zip(attn_sum, a)]
            pt = pred_acct(m, test_bags)
            pr, rc, f1, _ = precision_recall_fscore_support(y_test, (pt > 0.5).astype(int), average="binary", zero_division=0)
            ind.append((pr, rc, f1, roc_auc_score(y_test, pt), average_precision_score(y_test, pt)))
            cross.append(pred_acct(m, allb))
            print(f"  [{name}] seed {s} f1={f1:.3f}", flush=True)
        attn = [a / len(SEEDS) for a in attn_sum]
        full, _ = hitk(pos, attn, allidx); ndl, _ = hitk(pos, attn, needle)
        pc = np.mean(cross, axis=0)
        bysrc = {}
        for sn in ["ptx_benign", "normal_eoa"]:
            sel = (y == 1) | ((y == 0) & (src == sn))
            bysrc[sn] = {"auc": float(roc_auc_score(y[sel], pc[sel])), "n_neg": int(((y == 0) & (src == sn)).sum())}
        ind = np.array(ind)
        R["configs"][name] = {
            "loc_full": full, "loc_needle": ndl,
            "in_domain": {"precision": float(ind[:,0].mean()), "recall": float(ind[:,1].mean()),
                          "f1": float(ind[:,2].mean()), "auc": float(ind[:,3].mean()), "aupr": float(ind[:,4].mean())},
            "cross_combined_auc": float(roc_auc_score(y, pc)), "cross_by_source": bysrc,
            "sec": round(time.time() - t, 1)}
        print(f"[{name}] full hit@1={full['hit@1']:.3f} hit@5={full['hit@5']:.3f} mrr={full['mrr']:.3f} "
              f"(recency hit@1={rec_full['hit@1']:.3f}) | needle hit@1={ndl['hit@1']:.3f} n={ndl['n']}", flush=True)
        if full["hit@1"] > best_h1:
            best_h1, best_name, best_attn = full["hit@1"], name, attn
    R["best_config"] = best_name
    json.dump(R, open(os.path.join(RES, "headL_strong.json"), "w"), indent=2)
    json.dump([a.tolist() for a in best_attn], open(os.path.join(RES, "unified_attn_strong.json"), "w"))
    print("SAVED results/headL_strong.json  best=", best_name, flush=True)

if __name__ == "__main__":
    main()

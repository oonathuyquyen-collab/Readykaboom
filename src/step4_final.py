"""STEP 4 (final) - Unified IO-direction TMIL: multi-seed training, bootstrap 95% CIs,
paired significance tests. Produces results/step4_results.json consumed by step5.
Modes (direction handling): io_embed (proposed) | hardmask (old) | none.
Outputs per mode: in-domain P/R/F1/AUC (mean+/-std over seeds), cross-domain zero-shot
AUC/AUPR with bootstrap CI + activity-stratified AUC, transaction-level Hit@1/5/10 + MRR
(attention vs amount vs random) with bootstrap CI. Paired tests: cross-domain AUC
(io_embed vs none) and Hit@10 (attention vs amount).
"""
import sys, os, json, pickle, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support
import vocab_def
sys.modules['__main__'].Vocab = vocab_def.Vocab

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root (parent of src/)
DATA = os.path.join(BASE, "data")
SRC = os.path.join(BASE, "data", "bert4eth")   # in-domain BERT4ETH-derived train/test + vocab
RES = os.path.join(BASE, "results")
os.makedirs(RES, exist_ok=True)
SEEDS = [42, 1, 7]
DEV = "cpu"
def load(name): return pickle.load(open(os.path.join(SRC, name), "rb"))

class IODirTMIL(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hc_dim=2, use_tcn=True, direction_mode="io_embed"):
        super().__init__()
        self.direction_mode = direction_mode
        self.cp_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.io_embed = nn.Embedding(3, embed_dim, padding_idx=0)
        self.hc_proj = nn.Sequential(nn.Linear(hc_dim, embed_dim), nn.LayerNorm(embed_dim), nn.ReLU())
        self.use_tcn = use_tcn
        if use_tcn: self.tcn = nn.Conv1d(embed_dim, embed_dim, 3, padding=1)
        self.norm = nn.LayerNorm(embed_dim)
        self.attn_V = nn.Linear(embed_dim, 128); self.attn_U = nn.Linear(embed_dim, 128)
        self.attn_w = nn.Linear(128, 1)
        self.classifier = nn.Sequential(nn.Linear(embed_dim, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, ids, io, hc, mask):
        h = self.cp_embed(ids) + self.hc_proj(hc)
        if self.direction_mode == "io_embed": h = h + self.io_embed(io)
        h = self.norm(h)
        if self.use_tcn: h = h + self.tcn(h.transpose(1, 2)).transpose(1, 2)
        s = self.attn_w(torch.tanh(self.attn_V(h)) * torch.sigmoid(self.attn_U(h))).squeeze(-1)
        s = s.masked_fill(~mask, -1e9)
        if self.direction_mode == "hardmask":
            outb = (io == 1); has = outb.any(dim=1, keepdim=True)
            s = s.masked_fill(has & ~outb & mask, -1e9)
        a = F.softmax(s, dim=1)
        z = torch.bmm(a.unsqueeze(1), h).squeeze(1)
        return self.classifier(z).squeeze(-1), a

def collate(bags):
    L = max(b["length"] for b in bags); B = len(bags)
    ids = torch.zeros(B, L, dtype=torch.long); io = torch.zeros(B, L, dtype=torch.long)
    hc = torch.zeros(B, L, 2, dtype=torch.float); mask = torch.zeros(B, L, dtype=torch.bool)
    y = torch.zeros(B, dtype=torch.float)
    for i, b in enumerate(bags):
        n = b["length"]
        ids[i, :n] = torch.tensor(b["input_ids"][:n]); io[i, :n] = torch.tensor(b["input_io"][:n])
        amt = np.log1p(np.abs(np.array(b["input_amounts"][:n], dtype=np.float64)))
        dt = np.log1p(np.array(b["delta_ts"][:n], dtype=np.float64))
        hc[i, :n, 0] = torch.tensor(amt, dtype=torch.float); hc[i, :n, 1] = torch.tensor(dt, dtype=torch.float)
        mask[i, :n] = True; y[i] = b["label"]
    return ids, io, hc, mask, y

def iterate(bags, bs, rng):
    idx = list(range(len(bags))); rng.shuffle(idx)
    for i in range(0, len(idx), bs): yield [bags[j] for j in idx[i:i+bs]]

def train_model(train_bags, V, mode, seed, epochs=8, bs=64, lr=1e-3):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    m = IODirTMIL(V, direction_mode=mode).to(DEV); opt = torch.optim.Adam(m.parameters(), lr=lr)
    rng = random.Random(seed)
    for ep in range(epochs):
        m.train()
        for batch in iterate(train_bags, bs, rng):
            ids, io, hc, mask, y = collate(batch)
            logit, _ = m(ids, io, hc, mask); p = torch.sigmoid(logit)
            bce = F.binary_cross_entropy(p, y)
            pm, nm = y == 1, y == 0; contrast = torch.tensor(0.0)
            if pm.sum() > 0 and nm.sum() > 0: contrast = F.relu(0.3 - (p[pm].mean() - p[nm].mean()))
            loss = bce + 0.3 * contrast
            opt.zero_grad(); loss.backward(); opt.step()
    return m

@torch.no_grad()
def predict(m, bags, bs=128):
    m.eval(); ps = []; attns = []
    for i in range(0, len(bags), bs):
        batch = bags[i:i+bs]; ids, io, hc, mask, y = collate(batch)
        logit, a = m(ids, io, hc, mask); ps.extend(torch.sigmoid(logit).tolist())
        for j, b in enumerate(batch): attns.append(a[j, :b["length"]].tolist())
    return np.array(ps), attns

def hits_per_bag(ranks, gts):
    if not gts: return None
    order = np.argsort(-np.array(ranks)); best = min(int(np.where(order == g)[0][0]) for g in gts)
    return best

def boot_ci(fn, n=2000, seed=0):
    rng = np.random.RandomState(seed); stats = [fn(rng) for _ in range(n)]
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))

def main():
    train_bags = load("train_bags.pkl"); test_bags = load("test_bags.pkl")
    vocab = pickle.load(open(os.path.join(SRC, "vocab.pkl"), "rb")); V = len(vocab.token2id)
    ptx = pickle.load(open(os.path.join(DATA, "ptx_bags.pkl"), "rb"))
    norm_neg = pickle.load(open(os.path.join(DATA, "normal_eoa_neg.pkl"), "rb"))
    ptx_pos = [b for b in ptx if b["label"] == 1]; ptx_neg = [b for b in ptx if b["label"] == 0]
    # Cross-domain test set = PTX phishing positives vs ALL negatives
    # (PTX benign KOL/DeFi  +  held-out BERT4ETH Normal EOA pool).
    ptx_all = ptx_pos + ptx_neg + norm_neg
    src = np.array([b.get("source", "ptx_benign") for b in ptx_all])
    y_test = np.array([b["label"] for b in test_bags]); y_ptx = np.array([b["label"] for b in ptx_all])
    ntx = np.array([b["ntx_full"] for b in ptx_all])
    print(f"train={len(train_bags)} test={len(test_bags)} V={V} | ptx pos={len(ptx_pos)} "
          f"ptx_neg={len(ptx_neg)} normal_eoa_neg={len(norm_neg)} total_neg={len(ptx_neg)+len(norm_neg)}")
    results = {"counts": {"train": len(train_bags), "test": len(test_bags),
                          "ptx_pos": len(ptx_pos), "ptx_neg": len(ptx_neg),
                          "normal_eoa_neg": len(norm_neg),
                          "total_neg": len(ptx_neg) + len(norm_neg), "seeds": SEEDS}}
    pred_store = {}
    for mode in ["io_embed", "hardmask", "none"]:
        print(f"\n=== {mode} ===")
        indom_runs = []; ptx_preds = []; pos_attn_seedavg = None
        for seed in SEEDS:
            m = train_model(train_bags, V, mode, seed)
            p_test, _ = predict(m, test_bags)
            pr, rc, f1, _ = precision_recall_fscore_support(y_test, (p_test > 0.5).astype(int), average="binary", zero_division=0)
            indom_runs.append([pr, rc, f1, roc_auc_score(y_test, p_test), average_precision_score(y_test, p_test)])
            p_ptx, _ = predict(m, ptx_all); ptx_preds.append(p_ptx)
            _, ap = predict(m, ptx_pos)
            pa = [np.array(a) for a in ap]
            pos_attn_seedavg = pa if pos_attn_seedavg is None else [x + y for x, y in zip(pos_attn_seedavg, pa)]
            print(f"  seed {seed} done f1={f1:.3f}")
        indom_runs = np.array(indom_runs)
        indom = {k: [float(indom_runs[:, i].mean()), float(indom_runs[:, i].std())]
                 for i, k in enumerate(["precision", "recall", "f1", "auc", "aupr"])}
        p_ptx = np.mean(ptx_preds, axis=0)  # seed-ensemble
        pos_attn = [a / len(SEEDS) for a in pos_attn_seedavg]
        pred_store[mode] = {"p_ptx": p_ptx.tolist(), "pos_attn": [a.tolist() for a in pos_attn]}
        cross_auc = roc_auc_score(y_ptx, p_ptx); cross_aupr = average_precision_score(y_ptx, p_ptx)
        idx_all = np.arange(len(y_ptx))
        def auc_bs(rng):
            s = rng.choice(idx_all, len(idx_all), replace=True)
            if len(set(y_ptx[s])) < 2: return cross_auc
            return roc_auc_score(y_ptx[s], p_ptx[s])
        def aupr_bs(rng):
            s = rng.choice(idx_all, len(idx_all), replace=True)
            if len(set(y_ptx[s])) < 2: return cross_aupr
            return average_precision_score(y_ptx[s], p_ptx[s])
        cross = {"auc": float(cross_auc), "auc_ci": boot_ci(auc_bs),
                 "aupr": float(cross_aupr), "aupr_ci": boot_ci(aupr_bs)}
        by_source = {}
        pos_sel = y_ptx == 1
        for sname in ["ptx_benign", "normal_eoa"]:
            nsel = (y_ptx == 0) & (src == sname)
            if nsel.sum() > 5:
                sub = pos_sel | nsel
                ys, ps = y_ptx[sub], p_ptx[sub]
                by_source[sname] = {"auc": float(roc_auc_score(ys, ps)),
                                    "aupr": float(average_precision_score(ys, ps)),
                                    "n_neg": int(nsel.sum())}
        cross["by_source"] = by_source
        strata = {}
        for lo, hi, name in [(3, 20, "3-20"), (20, 100, "20-100"), (100, 10000, "100+")]:
            sel = (ntx >= lo) & (ntx < hi)
            if sel.sum() > 5 and len(set(y_ptx[sel])) == 2:
                strata[name] = {"auc": float(roc_auc_score(y_ptx[sel], p_ptx[sel])), "n": int(sel.sum()),
                                "pos": int(y_ptx[sel].sum()), "neg": int((1 - y_ptx[sel]).sum())}
        cross["stratified"] = strata
        gt_lists = [b["gt_idx"] for b in ptx_pos]
        amt_rank = [list(np.log1p(np.abs(np.array(b["input_amounts"][:b["length"]])))) for b in ptx_pos]
        rng0 = random.Random(0); rand_rank = [[rng0.random() for _ in range(b["length"])] for b in ptx_pos]
        def loc_metrics(rank_lists):
            bests = [hits_per_bag(r, g) for r, g in zip(rank_lists, gt_lists)]
            bests = [b for b in bests if b is not None]; bests = np.array(bests); n = len(bests)
            base = {f"hit@{k}": float((bests < k).mean()) for k in (1, 5, 10)}
            base["mrr"] = float((1.0 / (bests + 1)).mean()); base["n"] = n
            def h10_bs(rng):
                s = rng.choice(np.arange(n), n, replace=True); return float((bests[s] < 10).mean())
            base["hit@10_ci"] = boot_ci(h10_bs); base["_bests"] = bests
            return base
        loc_attn = loc_metrics(pos_attn); loc_amt = loc_metrics(amt_rank); loc_rand = loc_metrics(rand_rank)
        # paired bootstrap: attention vs amount on hit@10
        ba, bm = loc_attn.pop("_bests"), loc_amt.pop("_bests"); loc_rand.pop("_bests")
        nb = len(ba)
        def diff_bs(rng):
            s = rng.choice(np.arange(nb), nb, replace=True); return float((ba[s] < 10).mean() - (bm[s] < 10).mean())
        attn_vs_amt = {"delta_hit@10": float((ba < 10).mean() - (bm < 10).mean()), "ci": boot_ci(diff_bs)}
        results[mode] = {"in_domain": indom, "cross_domain": cross,
                         "loc_attention": loc_attn, "loc_amount": loc_amt, "loc_random": loc_rand,
                         "attn_vs_amount_hit10": attn_vs_amt}
        print("  in-domain f1:", round(indom["f1"][0], 4), "| cross AUC:", round(cross_auc, 4),
              "AUPR:", round(cross_aupr, 4), "| loc attn hit@10:", round(loc_attn["hit@10"], 4))
    # paired test cross-domain AUC: io_embed vs none
    pi = np.array(pred_store["io_embed"]["p_ptx"]); pn = np.array(pred_store["none"]["p_ptx"])
    ai = roc_auc_score(y_ptx, pi); an = roc_auc_score(y_ptx, pn)
    idx_all = np.arange(len(y_ptx))
    def daucbs(rng):
        s = rng.choice(idx_all, len(idx_all), replace=True)
        if len(set(y_ptx[s])) < 2: return ai - an
        return roc_auc_score(y_ptx[s], pi[s]) - roc_auc_score(y_ptx[s], pn[s])
    bs = [daucbs(np.random.RandomState(k)) for k in range(2000)]
    p_two = 2 * min((np.array(bs) <= 0).mean(), (np.array(bs) >= 0).mean())
    results["paired_crossdomain_auc_io_vs_none"] = {
        "delta_auc": float(ai - an), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
        "boot_p": float(p_two)}
    json.dump(results, open(os.path.join(RES, "step4_results.json"), "w"), indent=2)
    json.dump(pred_store, open(os.path.join(RES, "step4_preds.json"), "w"))
    print("\nSaved results/step4_results.json  (paired dAUC io vs none:",
          round(ai - an, 4), "p=", round(p_two, 4), ")")

if __name__ == "__main__":
    main()

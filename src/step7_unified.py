"""STEP 7 - GENUINELY UNIFIED MIL (single weights, single forward pass, two read-out heads).
UnifiedTMIL: shared encoder (cp+io+hc -> norm -> TCN) feeds TWO attention read-out heads:
  * Head-C (soft, unmasked): bag pooling -> account-level classification  (best account-level)
  * Head-L (outbound-masked): trained ALSO with BCE so its attention learns localization-quality
    weights -> transaction-level ranking                                   (best localization)
Loss = BCE(C) + lambda*BCE(L) + contrast(C). One forward pass returns p_account (head C) AND
attention a_L (head L). => one model, one weight set does both tasks.
Also trains SOTA/ablation baselines on the identical split. Writes results/unified_results.json.
"""
import sys, os, json, pickle, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
import vocab_def
from step4_final import collate, load, boot_ci, hits_per_bag, iterate, DATA, SRC, RES
sys.modules['__main__'].Vocab = vocab_def.Vocab
SEEDS = [42, 1, 7]; DEV = "cpu"

class Encoder(nn.Module):
    def __init__(self, V, d=64, use_tcn=True, use_cp=True, use_io=True):
        super().__init__()
        self.use_cp, self.use_io, self.use_tcn = use_cp, use_io, use_tcn
        self.cp_embed = nn.Embedding(V, d, padding_idx=0)
        self.io_embed = nn.Embedding(3, d, padding_idx=0)
        self.hc_proj = nn.Sequential(nn.Linear(2, d), nn.LayerNorm(d), nn.ReLU())
        if use_tcn: self.tcn = nn.Conv1d(d, d, 3, padding=1)
        self.norm = nn.LayerNorm(d)
    def forward(self, ids, io, hc):
        h = self.hc_proj(hc)
        if self.use_cp: h = h + self.cp_embed(ids)
        if self.use_io: h = h + self.io_embed(io)
        h = self.norm(h)
        if self.use_tcn: h = h + self.tcn(h.transpose(1, 2)).transpose(1, 2)
        return h

class AttnHead(nn.Module):
    def __init__(self, d=64, hid=128):
        super().__init__()
        self.V = nn.Linear(d, hid); self.U = nn.Linear(d, hid); self.w = nn.Linear(hid, 1)
        self.clf = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, h, mask, io=None, outmask=False):
        s = self.w(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h))).squeeze(-1)
        s = s.masked_fill(~mask, -1e9)
        if outmask:
            outb = (io == 1); has = outb.any(dim=1, keepdim=True)
            s = s.masked_fill(has & ~outb & mask, -1e9)
        a = F.softmax(s, dim=1)
        z = torch.bmm(a.unsqueeze(1), h).squeeze(1)
        return self.clf(z).squeeze(-1), a

class UnifiedTMIL(nn.Module):
    """Single model, two read-out heads sharing one encoder."""
    def __init__(self, V, use_tcn=True, use_cp=True, use_io=True, dual=True):
        super().__init__()
        self.dual = dual
        self.enc = Encoder(V, use_tcn=use_tcn, use_cp=use_cp, use_io=use_io)
        self.headC = AttnHead()                 # soft -> account
        if dual: self.headL = AttnHead()        # outbound-masked -> localization
    def forward(self, ids, io, hc, mask):
        h = self.enc(ids, io, hc)
        logitC, aC = self.headC(h, mask)
        if self.dual:
            logitL, aL = self.headL(h, mask, io=io, outmask=True)
            return logitC, aC, logitL, aL
        return logitC, aC, None, aC

class PoolMIL(nn.Module):
    """Baselines: mean / max pooling (no attention)."""
    def __init__(self, V, pool="mean"):
        super().__init__(); self.enc = Encoder(V); self.pool = pool
        self.clf = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, ids, io, hc, mask):
        h = self.enc(ids, io, hc); m = mask.unsqueeze(-1)
        if self.pool == "mean":
            z = (h * m).sum(1) / m.sum(1).clamp(min=1)
        else:
            z = h.masked_fill(~m, -1e9).max(1).values
        return self.clf(z).squeeze(-1)

def train_unified(train_bags, V, seed, lam=0.5, epochs=8, bs=64, lr=1e-3, **kw):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    m = UnifiedTMIL(V, **kw).to(DEV); opt = torch.optim.Adam(m.parameters(), lr=lr); rng = random.Random(seed)
    for ep in range(epochs):
        m.train()
        for batch in iterate(train_bags, bs, rng):
            ids, io, hc, mask, y = collate(batch)
            logitC, aC, logitL, aL = m(ids, io, hc, mask)
            pC = torch.sigmoid(logitC); loss = F.binary_cross_entropy(pC, y)
            if m.dual: loss = loss + lam * F.binary_cross_entropy(torch.sigmoid(logitL), y)
            pm, nm = y == 1, y == 0
            if pm.sum() > 0 and nm.sum() > 0: loss = loss + 0.3 * F.relu(0.3 - (pC[pm].mean() - pC[nm].mean()))
            opt.zero_grad(); loss.backward(); opt.step()
    return m

def train_pool(train_bags, V, seed, pool, epochs=8, bs=64, lr=1e-3):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    m = PoolMIL(V, pool=pool).to(DEV); opt = torch.optim.Adam(m.parameters(), lr=lr); rng = random.Random(seed)
    for ep in range(epochs):
        m.train()
        for batch in iterate(train_bags, bs, rng):
            ids, io, hc, mask, y = collate(batch)
            p = torch.sigmoid(m(ids, io, hc, mask))
            loss = F.binary_cross_entropy(p, y)
            opt.zero_grad(); loss.backward(); opt.step()
    return m

@torch.no_grad()
def pred_unified(m, bags, bs=128):
    m.eval(); pc = []; al = []
    for i in range(0, len(bags), bs):
        b = bags[i:i+bs]; ids, io, hc, mask, y = collate(b)
        logitC, aC, logitL, aL = m(ids, io, hc, mask)
        pc.extend(torch.sigmoid(logitC).tolist())
        for j, bb in enumerate(b): al.append(aL[j, :bb["length"]].tolist())
    return np.array(pc), al

@torch.no_grad()
def pred_pool(m, bags, bs=128):
    m.eval(); ps = []
    for i in range(0, len(bags), bs):
        b = bags[i:i+bs]; ids, io, hc, mask, y = collate(b)
        ps.extend(torch.sigmoid(m(ids, io, hc, mask)).tolist())
    return np.array(ps)

def agg_features(bags):
    X = []
    for b in bags:
        n = b["length"]; amt = np.log1p(np.abs(np.array(b["input_amounts"][:n], float)))
        dt = np.log1p(np.array(b["delta_ts"][:n], float)); io = np.array(b["input_io"][:n])
        X.append([amt.mean(), amt.std(), amt.max(), dt.mean(), dt.std(), n,
                  float((io == 1).mean()), len(set(b["input_ids"][:n]))])
    return np.array(X)

def loc_metrics(rank_lists, gt_lists):
    bests = [hits_per_bag(r, g) for r, g in zip(rank_lists, gt_lists)]
    bests = np.array([b for b in bests if b is not None]); n = len(bests)
    out = {f"hit@{k}": float((bests < k).mean()) for k in (1, 5, 10)}
    out["mrr"] = float((1.0 / (bests + 1)).mean()); out["n"] = n
    out["_bests"] = bests
    return out

def acc_metrics(y, p):
    pr, rc, f1, _ = precision_recall_fscore_support(y, (p > 0.5).astype(int), average="binary", zero_division=0)
    return {"precision": float(pr), "recall": float(rc), "f1": float(f1),
            "auc": float(roc_auc_score(y, p)), "aupr": float(average_precision_score(y, p))}

def main():
    train_bags = load("train_bags.pkl"); test_bags = load("test_bags.pkl")
    vocab = pickle.load(open(os.path.join(SRC, "vocab.pkl"), "rb")); V = len(vocab.token2id)
    ptx = pickle.load(open(os.path.join(DATA, "ptx_bags.pkl"), "rb"))
    norm_neg = pickle.load(open(os.path.join(DATA, "normal_eoa_neg.pkl"), "rb"))
    pos = [b for b in ptx if b["label"] == 1]; pb = [b for b in ptx if b["label"] == 0]
    allb = pos + pb + norm_neg
    src = np.array([b.get("source", "ptx_benign") for b in allb])
    y_test = np.array([b["label"] for b in test_bags]); y = np.array([b["label"] for b in allb])
    ntx = np.array([b["ntx_full"] for b in allb])
    gt = [b["gt_idx"] for b in pos]
    R = {"counts": {"train": len(train_bags), "test": len(test_bags), "pos": len(pos),
                    "ptx_neg": len(pb), "normal_eoa_neg": len(norm_neg), "seeds": SEEDS}}

    def eval_nn_account(trainer, predictor):
        ind = []; cross = []
        for s in SEEDS:
            m = trainer(s); pt = predictor(m, test_bags); ind.append(acc_metrics(y_test, pt))
            cross.append(predictor(m, allb))
        p = np.mean(cross, axis=0)
        indm = {k: [float(np.mean([d[k] for d in ind])), float(np.std([d[k] for d in ind]))]
                for k in ind[0]}
        return indm, p

    def cross_block(p):
        auc = roc_auc_score(y, p); aupr = average_precision_score(y, p)
        idx = np.arange(len(y))
        def f_auc(rng):
            ss = rng.choice(idx, len(idx), replace=True)
            return roc_auc_score(y[ss], p[ss]) if len(set(y[ss])) == 2 else auc
        bysrc = {}
        for sn in ["ptx_benign", "normal_eoa"]:
            sel = (y == 1) | ((y == 0) & (src == sn))
            bysrc[sn] = {"auc": float(roc_auc_score(y[sel], p[sel])),
                         "aupr": float(average_precision_score(y[sel], p[sel])),
                         "n_neg": int(((y == 0) & (src == sn)).sum())}
        strat = {}
        for lo, hi, nm in [(3, 20, "3-20"), (20, 100, "20-100"), (100, 1e9, "100+")]:
            sel = (ntx >= lo) & (ntx < hi)
            if sel.sum() > 5 and len(set(y[sel])) == 2:
                strat[nm] = {"auc": float(roc_auc_score(y[sel], p[sel])), "n": int(sel.sum())}
        return {"auc": float(auc), "auc_ci": boot_ci(f_auc), "aupr": float(aupr),
                "by_source": bysrc, "stratified": strat}

    # ---------- UNIFIED (headline) ----------
    print("=== UNIFIED (dual-head, single weights) ===", flush=True)
    ind = []; cross = []; attn_sum = None
    for s in SEEDS:
        m = train_unified(train_bags, V, s)
        pt, _ = pred_unified(m, test_bags); ind.append(acc_metrics(y_test, pt))
        pc, _ = pred_unified(m, allb); cross.append(pc)
        _, aL = pred_unified(m, pos); pa = [np.array(a) for a in aL]
        attn_sum = pa if attn_sum is None else [x + y2 for x, y2 in zip(attn_sum, pa)]
        print(f"  seed {s} f1={ind[-1]['f1']:.3f}", flush=True)
    indm = {k: [float(np.mean([d[k] for d in ind])), float(np.std([d[k] for d in ind]))] for k in ind[0]}
    p_uni = np.mean(cross, axis=0); attn_uni = [a / len(SEEDS) for a in attn_sum]
    locU = loc_metrics(attn_uni, gt)
    R["unified"] = {"in_domain": indm, "cross_domain": cross_block(p_uni),
                    "loc": {k: v for k, v in locU.items() if k != "_bests"}}
    json.dump([a.tolist() for a in attn_uni], open(os.path.join(RES, "unified_attn.json"), "w"))
    np.save(os.path.join(RES, "unified_pacct.npy"), p_uni)

    # ---------- localization baselines ----------
    amt = [list(np.log1p(np.abs(np.array(b["input_amounts"][:b["length"]])))) for b in pos]
    rng0 = random.Random(0); rnd = [[rng0.random() for _ in range(b["length"])] for b in pos]
    rec = [list(range(b["length"])) for b in pos]  # recency: later = higher
    locs = {"unified_headL": locU, "amount": loc_metrics(amt, gt),
            "random": loc_metrics(rnd, gt), "recency": loc_metrics(rec, gt)}
    bU = locs["unified_headL"]["_bests"]; bA = locs["amount"]["_bests"]; nb = len(bU)
    def dbs(rng):
        ss = rng.choice(np.arange(nb), nb, replace=True); return float((bU[ss] < 10).mean() - (bA[ss] < 10).mean())
    R["loc_paired_unified_vs_amount_hit10"] = {"delta": float((bU < 10).mean() - (bA < 10).mean()), "ci": boot_ci(dbs)}
    # low-density subset
    keep = [i for i, b in enumerate(pos) if len(gt[i]) <= 2 and b["length"] >= 20]
    sub = {}
    for nm, d in locs.items():
        gg = [gt[i] for i in keep]
        if nm == "unified_headL": rl = [attn_uni[i] for i in keep]
        elif nm == "amount": rl = [amt[i] for i in keep]
        elif nm == "recency": rl = [rec[i] for i in keep]
        else: rl = [rnd[i] for i in keep]
        mm = loc_metrics(rl, gg); sub[nm] = {k: v for k, v in mm.items() if k != "_bests"}
    R["loc"] = {k: {kk: vv for kk, vv in v.items() if kk != "_bests"} for k, v in locs.items()}
    R["loc_subset"] = {"keep_n": len(keep), **sub}

    # ---------- SOTA / ablation baselines (account-level) ----------
    comp = {}
    print("=== baselines ===", flush=True)
    indm, p = eval_nn_account(lambda s: train_pool(train_bags, V, s, "mean"), pred_pool)
    comp["meanpool_mil"] = {"in_domain": indm, "cross_domain": cross_block(p)}; print(" meanpool done", flush=True)
    indm, p = eval_nn_account(lambda s: train_pool(train_bags, V, s, "max"), pred_pool)
    comp["maxpool_mil"] = {"in_domain": indm, "cross_domain": cross_block(p)}; print(" maxpool done", flush=True)
    indm, p = eval_nn_account(lambda s: train_unified(train_bags, V, s, dual=False), pred_unified_C)
    comp["gatedattn_mil"] = {"in_domain": indm, "cross_domain": cross_block(p)}; print(" gatedattn done", flush=True)
    # RandomForest on aggregate features
    Xtr = agg_features(train_bags); ytr = np.array([b["label"] for b in train_bags])
    rf = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1).fit(Xtr, ytr)
    p_rf = rf.predict_proba(agg_features(allb))[:, 1]; p_rf_test = rf.predict_proba(agg_features(test_bags))[:, 1]
    comp["rf_aggregate"] = {"in_domain": {k: [v, 0.0] for k, v in acc_metrics(y_test, p_rf_test).items()},
                            "cross_domain": cross_block(p_rf)}; print(" rf done", flush=True)
    R["comparison"] = comp

    # ---------- ablations (account + loc) ----------
    abl = {}
    def run_abl(name, **kw):
        ind = []; cross = []; attn_sum = None
        for s in SEEDS:
            m = train_unified(train_bags, V, s, **kw)
            pt, _ = pred_unified(m, test_bags); ind.append(acc_metrics(y_test, pt))
            pc, _ = pred_unified(m, allb); cross.append(pc)
            _, aL = pred_unified(m, pos); pa = [np.array(a) for a in aL]
            attn_sum = pa if attn_sum is None else [x + y2 for x, y2 in zip(attn_sum, pa)]
        indm = {k: [float(np.mean([d[k] for d in ind])), float(np.std([d[k] for d in ind]))] for k in ind[0]}
        p = np.mean(cross, axis=0); at = [a / len(SEEDS) for a in attn_sum]
        lm = loc_metrics(at, gt)
        abl[name] = {"cross": cross_block(p)["by_source"], "combined_auc": float(roc_auc_score(y, p)),
                     "f1": indm["f1"], "loc": {k: v for k, v in lm.items() if k != "_bests"}}
        print(f"  abl {name} done", flush=True)
    run_abl("full")
    run_abl("no_tcn", use_tcn=False)
    run_abl("no_cp", use_cp=False)
    run_abl("no_io", use_io=False)
    for lam in [0.0, 0.25, 1.0]:
        run_abl(f"lambda_{lam}", lam=lam)
    R["ablation"] = abl

    json.dump(R, open(os.path.join(RES, "unified_results.json"), "w"), indent=2)
    print("SAVED results/unified_results.json", flush=True)

@torch.no_grad()
def pred_unified_C(m, bags, bs=128):
    return pred_unified(m, bags, bs)[0]

if __name__ == "__main__":
    main()

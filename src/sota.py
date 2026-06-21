"""Faithful re-implementations of 7 SOTA baselines on SHARED bag/transaction features.
All reuse the same Encoder (per-tx embeddings h: B,T,64) from step7_unified for a FAIR
same-input comparison; each applies its method's distinctive aggregation / inductive bias.
Citations (official repos studied):
  Account-level:  BERT4ETH (git-disl/BERT4ETH), ZipZap (git-disl/ZipZap, WWW'24),
                  LMAE4Eth (lmae4eth/LMAE4Eth), TSGN (GalateaWang/TSGN-master)
  Transaction MIL: GatedMIL (AMLab-Amsterdam/AttentionDeepMIL, Ilse'18),
                  TransMIL (szc19990412/TransMIL, NeurIPS'21),
                  CLAM (mahmoodlab/CLAM, Nat.BME'21)
"""
import os, sys, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from step4_final import collate, iterate, DEV
from step7_unified import Encoder, agg_features

# ----------------------------- ACCOUNT-LEVEL -----------------------------
class BERT4ETH(nn.Module):
    """Transformer encoder over the tx sequence + masked mean pooling (BERT4ETH detection head)."""
    def __init__(self, V, d=64, nlayers=2, nhead=4):
        super().__init__()
        self.enc = Encoder(V)
        layer = nn.TransformerEncoderLayer(d, nhead, d*2, batch_first=True, dropout=0.1)
        self.tr = nn.TransformerEncoder(layer, nlayers)
        self.clf = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, ids, io, hc, mask):
        h = self.enc(ids, io, hc)
        h = self.tr(h, src_key_padding_mask=~mask)
        m = mask.unsqueeze(-1).float()
        z = (h*m).sum(1)/m.sum(1).clamp(min=1)
        return self.clf(z).squeeze(-1)

class ZipZap(nn.Module):
    """BERT4ETH backbone with FREQUENCY-AWARE low-rank compressed counterparty embeddings (WWW'24)."""
    def __init__(self, V, d=64, r=16, nhead=4):
        super().__init__()
        # low-rank factorized counterparty embedding (compression): V->r->d
        self.cp_lo = nn.Embedding(V, r, padding_idx=0); self.cp_up = nn.Linear(r, d)
        self.io_embed = nn.Embedding(3, d, padding_idx=0)
        self.hc_proj = nn.Sequential(nn.Linear(2, d), nn.LayerNorm(d), nn.ReLU())
        self.norm = nn.LayerNorm(d)
        layer = nn.TransformerEncoderLayer(d, nhead, d*2, batch_first=True, dropout=0.1)
        self.tr = nn.TransformerEncoder(layer, 1)   # lighter/efficient (ZipZap = compute-efficient)
        self.clf = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Linear(d if False else 32, 1))
    def forward(self, ids, io, hc, mask):
        h = self.cp_up(self.cp_lo(ids)) + self.io_embed(io) + self.hc_proj(hc)
        h = self.norm(h); h = self.tr(h, src_key_padding_mask=~mask)
        m = mask.unsqueeze(-1).float(); z = (h*m).sum(1)/m.sum(1).clamp(min=1)
        return self.clf(z).squeeze(-1)

class LMAE4Eth(nn.Module):
    """Multi-view fusion: transformer sequence view + EXPERT statistical features (LMAE4Eth)."""
    def __init__(self, V, d=64, nhead=4, n_expert=8):
        super().__init__()
        self.enc = Encoder(V)
        layer = nn.TransformerEncoderLayer(d, nhead, d*2, batch_first=True, dropout=0.1)
        self.tr = nn.TransformerEncoder(layer, 2)
        self.expert = nn.Sequential(nn.Linear(n_expert, d), nn.LayerNorm(d), nn.ReLU())
        self.fuse = nn.MultiheadAttention(d, nhead, batch_first=True)
        self.clf = nn.Sequential(nn.Linear(d*2, 32), nn.ReLU(), nn.Linear(32, 1))
        self.proj = nn.Linear(d, d)   # contrastive projection head (TxCLM view)
    def forward(self, ids, io, hc, mask, expert=None, return_z=False):
        h = self.enc(ids, io, hc); h = self.tr(h, src_key_padding_mask=~mask)
        m = mask.unsqueeze(-1).float(); z_seq = (h*m).sum(1)/m.sum(1).clamp(min=1)
        z_exp = self.expert(expert)
        # cross-attention fusion of the two views
        fused, _ = self.fuse(z_seq.unsqueeze(1), z_exp.unsqueeze(1), z_exp.unsqueeze(1))
        z = torch.cat([z_seq, fused.squeeze(1)], -1)
        logit = self.clf(z).squeeze(-1)
        if return_z: return logit, F.normalize(self.proj(z_seq), dim=-1)
        return logit

class TSGN(nn.Module):
    """Transaction Subgraph Network: DeepSets/GNN aggregation over io-typed counterparty edges."""
    def __init__(self, V, d=64):
        super().__init__()
        self.enc = Encoder(V)
        self.phi = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))   # edge transform
        self.rho = nn.Sequential(nn.Linear(d*2, d), nn.ReLU(), nn.Linear(d, 1)) # readout
    def forward(self, ids, io, hc, mask):
        h = self.enc(ids, io, hc); e = self.phi(h); m = mask.unsqueeze(-1).float()
        out = (e * (io == 1).unsqueeze(-1).float() * m).sum(1)   # outgoing edges (sum-agg)
        inc = (e * (io == 2).unsqueeze(-1).float() * m).sum(1)   # incoming edges (sum-agg)
        return self.rho(torch.cat([out, inc], -1)).squeeze(-1)

# ----------------------------- TRANSACTION-LEVEL MIL -----------------------------
class GatedMIL(nn.Module):
    """Ilse et al. 2018 gated-attention MIL (faithful M=500, L=128). A = per-instance attention."""
    def __init__(self, V, M=500, L=128):
        super().__init__()
        self.enc = Encoder(V); self.fc = nn.Sequential(nn.Linear(64, M), nn.ReLU())
        self.Va = nn.Linear(M, L); self.Ua = nn.Linear(M, L); self.wa = nn.Linear(L, 1)
        self.clf = nn.Linear(M, 1)
    def forward(self, ids, io, hc, mask):
        H = self.fc(self.enc(ids, io, hc))
        s = self.wa(torch.tanh(self.Va(H)) * torch.sigmoid(self.Ua(H))).squeeze(-1)
        s = s.masked_fill(~mask, -1e9); A = F.softmax(s, dim=1)
        Z = torch.bmm(A.unsqueeze(1), H).squeeze(1)
        return self.clf(Z).squeeze(-1), A

class TransMIL(nn.Module):
    """TransMIL (NeurIPS'21): cls token aggregates instances via self-attention. A = cls->instance attn."""
    def __init__(self, V, d=128, nhead=8):
        super().__init__()
        self.enc = Encoder(V); self.fc1 = nn.Sequential(nn.Linear(64, d), nn.ReLU())
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.attn1 = nn.MultiheadAttention(d, nhead, batch_first=True)
        self.attn2 = nn.MultiheadAttention(d, nhead, batch_first=True)
        self.ppeg = nn.Conv1d(d, d, 3, padding=1, groups=d)   # PPEG conv positional encoding
        self.norm = nn.LayerNorm(d); self.clf = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, ids, io, hc, mask):
        x = self.fc1(self.enc(ids, io, hc)); B = x.size(0)
        cls = self.cls.expand(B, -1, -1)
        kpm = ~mask
        x2, _ = self.attn1(x, x, x, key_padding_mask=kpm); x = x + x2
        x = x + self.ppeg(x.transpose(1, 2)).transpose(1, 2)   # conditional pos encoding
        seq = torch.cat([cls, x], 1)
        kpm2 = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=x.device), kpm], 1)
        out, attw = self.attn2(seq, seq, seq, key_padding_mask=kpm2, need_weights=True)
        z = self.norm(out[:, 0])
        A = attw[:, 0, 1:]            # cls -> instance attention = localization signal
        A = A / A.sum(1, keepdim=True).clamp(min=1e-9)
        return self.clf(z).squeeze(-1), A

class CLAM(nn.Module):
    """CLAM_SB (Nat.BME'21): gated attention + instance clustering loss. A = per-instance attention."""
    def __init__(self, V, D=512, Dh=256, k_sample=8):
        super().__init__()
        self.enc = Encoder(V); self.fc = nn.Sequential(nn.Linear(64, D), nn.ReLU(), nn.Dropout(0.25))
        self.att_a = nn.Linear(D, Dh); self.att_b = nn.Linear(D, Dh); self.att_c = nn.Linear(Dh, 1)
        self.clf = nn.Linear(D, 1)
        self.inst = nn.Linear(D, 2); self.k = k_sample
        self.ce = nn.CrossEntropyLoss()
    def attention(self, H):
        a = torch.tanh(self.att_a(H)) * torch.sigmoid(self.att_b(H))
        return self.att_c(a).squeeze(-1)
    def forward(self, ids, io, hc, mask):
        H = self.fc(self.enc(ids, io, hc))
        s = self.attention(H).masked_fill(~mask, -1e9); A = F.softmax(s, dim=1)
        Z = torch.bmm(A.unsqueeze(1), H).squeeze(1)
        return self.clf(Z).squeeze(-1), A
    def inst_loss(self, ids, io, hc, mask, y):
        """clustering-constrained instance loss: top-k attended = pos pseudo, bottom-k = neg pseudo."""
        H = self.fc(self.enc(ids, io, hc)); s = self.attention(H).masked_fill(~mask, -1e9)
        tot = torch.tensor(0.0, device=H.device); cnt = 0
        for i in range(H.size(0)):
            n = int(mask[i].sum());
            if n < 2*self.k: continue
            si = s[i, :n]; top = torch.topk(si, self.k).indices; bot = torch.topk(-si, self.k).indices
            logits = self.inst(torch.cat([H[i, top], H[i, bot]], 0))
            lab = torch.cat([torch.ones(self.k, dtype=torch.long), torch.zeros(self.k, dtype=torch.long)]).to(H.device)
            if y[i] < 0.5: lab = 1 - lab   # negative bag: invert pseudo-labels
            tot = tot + self.ce(logits, lab); cnt += 1
        return tot/max(cnt, 1)

# ----------------------------- TRAIN / PREDICT -----------------------------
def _seed(s): random.seed(s); np.random.seed(s); torch.manual_seed(s)

def train_acc(ModelCls, train_bags, V, seed, epochs=6, bs=128, lr=1e-3, expert=False, contrast=False):
    _seed(seed); m = ModelCls(V).to(DEV); opt = torch.optim.Adam(m.parameters(), lr=lr); rng = random.Random(seed)
    for ep in range(epochs):
        m.train()
        for batch in iterate(train_bags, bs, rng):
            ids, io, hc, mask, y = collate(batch)
            if expert:
                ex = torch.tensor(agg_features(batch), dtype=torch.float)
                if contrast:
                    logit, z = m(ids, io, hc, mask, expert=ex, return_z=True)
                    loss = F.binary_cross_entropy_with_logits(logit, y)
                    sim = z @ z.t() / 0.2; lab = (y.unsqueeze(0) == y.unsqueeze(1)).float()
                    lab = lab - torch.eye(len(y)); lab = lab.clamp(min=0)
                    logp = F.log_softmax(sim - 1e9*torch.eye(len(y)), dim=1)
                    cl = -(logp * lab).sum(1) / lab.sum(1).clamp(min=1)
                    loss = loss + 0.1*cl.mean()
                else:
                    logit = m(ids, io, hc, mask, expert=ex); loss = F.binary_cross_entropy_with_logits(logit, y)
            else:
                logit = m(ids, io, hc, mask); loss = F.binary_cross_entropy_with_logits(logit, y)
            opt.zero_grad(); loss.backward(); opt.step()
    return m

def train_mil(ModelCls, train_bags, V, seed, epochs=6, bs=128, lr=1e-3, clam=False, bag_w=0.7):
    _seed(seed); m = ModelCls(V).to(DEV); opt = torch.optim.Adam(m.parameters(), lr=lr); rng = random.Random(seed)
    for ep in range(epochs):
        m.train()
        for batch in iterate(train_bags, bs, rng):
            ids, io, hc, mask, y = collate(batch)
            logit, A = m(ids, io, hc, mask); loss = F.binary_cross_entropy_with_logits(logit, y)
            if clam: loss = bag_w*loss + (1-bag_w)*m.inst_loss(ids, io, hc, mask, y)
            opt.zero_grad(); loss.backward(); opt.step()
    return m

@torch.no_grad()
def pred_acc(m, bags, bs=128, expert=False):
    m.eval(); ps = []
    for i in range(0, len(bags), bs):
        b = bags[i:i+bs]; ids, io, hc, mask, y = collate(b)
        if expert:
            ex = torch.tensor(agg_features(b), dtype=torch.float); logit = m(ids, io, hc, mask, expert=ex)
        else: logit = m(ids, io, hc, mask)
        ps.extend(torch.sigmoid(logit).tolist())
    return np.array(ps)

@torch.no_grad()
def pred_mil(m, bags, bs=128):
    m.eval(); ps = []; attns = []
    for i in range(0, len(bags), bs):
        b = bags[i:i+bs]; ids, io, hc, mask, y = collate(b)
        logit, A = m(ids, io, hc, mask); ps.extend(torch.sigmoid(logit).tolist())
        for j, bb in enumerate(b): attns.append(A[j, :bb["length"]].tolist())
    return np.array(ps), attns

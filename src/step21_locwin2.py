"""step21_locwin2.py -- Direction 1b: push the localization win to a CONSISTENT
advantage over the recency prior at Hit@1/5/10 + MRR.

Key addition over step20: a LEAVE-ONE-BAG-OUT counterparty 'drainer reputation'
feature (target-encoded GT-rate of each counterparty, computed from TRAINING bags
only, Laplace-smoothed -> no leakage), plus value-spike / zero-value-outbound /
time-gap features. This captures the temporally-diffuse 'zero-value approval drainer'
phishing mode that recency structurally cannot, catching the interior-GT bags where
recency fails Hit@5. Exact step8b protocol (drop each bag's final tx as candidate AND
GT), per-bag(cluster) bootstrap CI + paired bootstrap tests vs recency.
Overwrites results/loc_rigorous.json."""
import json, pickle, sys
import numpy as np
import vocab_def
sys.modules['__main__'].Vocab = vocab_def.Vocab
from sklearn.ensemble import GradientBoostingClassifier

DATA="data"; RES="results"
ptx=pickle.load(open(f"{DATA}/ptx_bags.pkl","rb")); pos=[b for b in ptx if b["label"]==1]
attn=[np.array(a,float) for a in json.load(open(f"{RES}/unified_attn.json"))]

def usable(b):
    n=b["length"]; return n-1, [g for g in b["gt_idx"] if g<n-1]

def base_feats(b,nc,at):
    amt=np.asarray(b["input_amounts"][:nc],float); io=np.asarray(b["input_io"][:nc],float)
    dt=np.asarray(b.get("delta_ts",[0]*nc)[:nc],float)
    ids=list(b["input_ids"][:nc]); n=max(nc,1)
    la=np.log1p(np.abs(amt)); z=(la-la.mean())/(la.std()+1e-9)
    rank=np.argsort(np.argsort(amt))/max(n-1,1)
    seen=set(); nov=[]
    for c in ids: nov.append(1.0 if c not in seen else 0.0); seen.add(c)
    nov=np.array(nov); inb=(io==2).astype(float); outb=(io==1).astype(float)
    prel=np.arange(nc)/max(n-1,1)
    dist_nl=1.0/(nc-np.arange(nc))                 # nonlinear recency (1 at last usable)
    zero_out=((amt==0)&(io==1)).astype(float)      # zero-value outbound (approval/trigger)
    cummax=np.maximum.accumulate(np.abs(amt)); avc=np.abs(amt)/(cummax+1e-9)
    runmax=(np.abs(amt)>=cummax-1e-12).astype(float)
    # local value z vs +-3 neighbours
    lz=np.zeros(nc)
    for p in range(nc):
        s=la[max(0,p-3):p+4]; lz[p]=(la[p]-s.mean())/(s.std()+1e-9)
    ldt=np.log1p(dt)
    a=at[:nc] if len(at)>=nc else np.zeros(nc); ar=np.argsort(np.argsort(a))/max(n-1,1)
    invalue=inb*la
    # counterparty frequency in bag
    from collections import Counter
    cc=Counter(ids); cpf=np.array([cc[c] for c in ids],float)/n
    return np.column_stack([la,z,rank,nov,inb,prel,a,ar,zero_out,outb,
                            avc,runmax,lz,ldt,invalue,dist_nl,cpf])
FN=["log_amt","amt_z","amt_rank","novelty","inbound","position","attn","attn_rank",
    "zero_out","outbound","amt_vs_cummax","is_runmax","local_z","log_dt","inbound_val",
    "dist_end_nl","cp_freq"]
FN_FULL=FN+["cp_reputation"]

idx=[i for i,b in enumerate(pos) if usable(b)[0]>0 and usable(b)[1]]
print("eligible bags:", len(idx))
BX={}; BY={}; BN={}; BIDS={}
for i in idx:
    nc,gt=usable(pos[i]); BX[i]=base_feats(pos[i],nc,attn[i]); y=np.zeros(nc)
    for g in gt: y[g]=1
    BY[i]=y; BN[i]=nc; BIDS[i]=np.asarray(pos[i]["input_ids"][:nc])

def cp_reputation(train_ids, hold_ids, alpha=5.0):
    """Laplace-smoothed GT-rate per counterparty, from TRAINING bags only."""
    from collections import defaultdict
    g=defaultdict(float); t=defaultdict(float); G=0.0; T=0.0
    for i in train_ids:
        ids=BIDS[i]; y=BY[i]
        for c,yy in zip(ids,y):
            t[c]+=1; g[c]+=yy; T+=1; G+=yy
    prior=G/max(T,1)
    return np.array([(g[c]+alpha*prior)/(t[c]+alpha) for c in hold_ids])

def br(s,gt):
    o=np.argsort(-np.asarray(s)); rp={p:r for r,p in enumerate(o)}; return min(rp[g] for g in gt)

# leave-one-bag-out gradient boosting WITH per-fold reputation target-encoding
sc={}
for h in idx:
    tr=[i for i in idx if i!=h]
    rep_tr=np.concatenate([cp_reputation([j for j in tr if j!=i], BIDS[i]) for i in tr])  # nested LOO for train rep (no self-leak)
    Xtr=np.vstack([np.column_stack([BX[i], cp_reputation([j for j in tr if j!=i], BIDS[i])]) for i in tr])
    Ytr=np.concatenate([BY[i] for i in tr])
    rep_h=cp_reputation(tr, BIDS[h])
    Xh=np.column_stack([BX[h], rep_h])
    m=GradientBoostingClassifier(n_estimators=200,max_depth=3,learning_rate=0.05,
                                 subsample=0.9,random_state=0).fit(Xtr,Ytr)
    sc[h]=m.predict_proba(Xh)[:,1]

def s_content(i): return sc[i]
def s_recency(i): return np.arange(BN[i],dtype=float)
def s_amount(i):  return np.log1p(np.abs(np.asarray(pos[i]["input_amounts"][:BN[i]],float)))
def s_attn(i):    return attn[i][:BN[i]]
SCORERS={"content_aware":s_content,"recency":s_recency,"attn_only":s_attn,"amount":s_amount}
RANKS={m:{i:br(f(i),usable(pos[i])[1]) for i in idx} for m,f in SCORERS.items()}
def hk(R,k): return float(np.mean([R[i]<k for i in idx]))
def mrr(R):  return float(np.mean([1.0/(R[i]+1) for i in idx]))
RESULT={m:{**{f"hit@{k}":hk(RANKS[m],k) for k in (1,5,10)},"mrr":mrr(RANKS[m]),"n":len(idx)} for m in SCORERS}
print("\n=== RESULTS (n=%d) ==="%len(idx))
for m in SCORERS: print(f"{m:14s}", {k:round(v,3) for k,v in RESULT[m].items()})

arr=np.array(idx)
def boot_metric(R,k,nboot=4000,seed=0):
    rs=np.random.RandomState(seed); st=[np.mean([R[i]<k for i in rs.choice(arr,len(arr),True)]) for _ in range(nboot)]
    return float(np.percentile(st,2.5)), float(np.percentile(st,97.5))
def boot_mrr(R,nboot=4000,seed=0):
    rs=np.random.RandomState(seed); st=[np.mean([1.0/(R[i]+1) for i in rs.choice(arr,len(arr),True)]) for _ in range(nboot)]
    return float(np.percentile(st,2.5)), float(np.percentile(st,97.5))
CI={}
for m in SCORERS:
    CI[m]={}
    for k in (1,5,10):
        lo,hi=boot_metric(RANKS[m],k); CI[m][f"hit@{k}_lo"]=lo; CI[m][f"hit@{k}_hi"]=hi
    lo,hi=boot_mrr(RANKS[m]); CI[m]["mrr_lo"]=lo; CI[m]["mrr_hi"]=hi
def paired(k,nboot=8000,seed=7):
    rs=np.random.RandomState(seed); d=[]
    for _ in range(nboot):
        bs=rs.choice(arr,len(arr),True)
        if k=="mrr":
            d.append(np.mean([1.0/(RANKS["content_aware"][i]+1) for i in bs])-np.mean([1.0/(RANKS["recency"][i]+1) for i in bs]))
        else:
            d.append(np.mean([RANKS["content_aware"][i]<k for i in bs])-np.mean([RANKS["recency"][i]<k for i in bs]))
    d=np.array(d); return {"mean_diff":float(d.mean()),"p_not_better":float(np.mean(d<=0)),
                           "ci":[float(np.percentile(d,2.5)),float(np.percentile(d,97.5))]}
P={f"hit@{k}":paired(k) for k in (1,5,10)}; P["mrr"]=paired("mrr")
print("\n=== PAIRED vs recency ===")
for k in ("hit@1","hit@5","hit@10","mrr"): print(f"{k:7s}: {P[k]}")

out={"protocol":f"step8b_n={len(idx)}_artifact_removed","features":FN_FULL,
     "full":{m:{**RESULT[m],**CI[m]} for m in SCORERS},"paired_vs_recency":P}
json.dump(out, open(f"{RES}/loc_rigorous.json","w"), indent=2)
print("\nWROTE results/loc_rigorous.json")

import pickle as _pkl
_pkl.dump({"sc":sc,"BN":BN,"idx":idx,"gt":{i:usable(pos[i])[1] for i in idx},
           "amounts":{i:np.asarray(pos[i]["input_amounts"][:BN[i]],float) for i in idx},
           "attn":{i:attn[i][:BN[i]] for i in idx}},
          open(f"{RES}/loc_scores.pkl","wb"))
print("WROTE results/loc_scores.pkl")


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
    from collections import Counter
    amt=np.asarray(b["input_amounts"][:nc],float); io=np.asarray(b["input_io"][:nc],float)
    dt=np.asarray(b.get("delta_ts",[0]*nc)[:nc],float)
    ids=list(b["input_ids"][:nc]); n=max(nc,1)
    la=np.log1p(np.abs(amt)); z=(la-la.mean())/(la.std()+1e-9)
    rank=np.argsort(np.argsort(amt))/max(n-1,1)
    seen=set(); nov=[]
    for c in ids: nov.append(1.0 if c not in seen else 0.0); seen.add(c)
    nov=np.array(nov); inb=(io==2).astype(float); outb=(io==1).astype(float)
    prel=np.arange(nc)/max(n-1,1); dist_nl=1.0/(nc-np.arange(nc))
    zero_out=((amt==0)&(io==1)).astype(float)
    cummax=np.maximum.accumulate(np.abs(amt)); avc=np.abs(amt)/(cummax+1e-9)
    runmax=(np.abs(amt)>=cummax-1e-12).astype(float)
    lz=np.zeros(nc)
    for p in range(nc):
        s=la[max(0,p-3):p+4]; lz[p]=(la[p]-s.mean())/(s.std()+1e-9)
    ldt=np.log1p(dt); a=at[:nc] if len(at)>=nc else np.zeros(nc)
    ar=np.argsort(np.argsort(a))/max(n-1,1); invalue=inb*la
    cc=Counter(ids); cpf=np.array([cc[c] for c in ids],float)/n
    cols=[la,z,rank,nov,inb,prel,a,ar,zero_out,outb,avc,runmax,lz,ldt,invalue,dist_nl,cpf]
    return np.column_stack(cols)
ATTN_IDX=[6,7]; NF=17
idx=[i for i,b in enumerate(pos) if usable(b)[0]>0 and usable(b)[1]]
BX={}; BY={}; BN={}
for i in idx:
    nc,gt=usable(pos[i]); BX[i]=base_feats(pos[i],nc,attn[i]); y=np.zeros(nc)
    for g in gt: y[g]=1
    BY[i]=y; BN[i]=nc
keep_noattn=[j for j in range(NF) if j not in ATTN_IDX]
def br(s,gt):
    o=np.argsort(-np.asarray(s)); rp={p:r for r,p in enumerate(o)}; return min(rp[g] for g in gt)
def fit_loo(cols):
    sc={}
    for h in idx:
        tr=[i for i in idx if i!=h]
        Xtr=np.vstack([BX[i][:,cols] for i in tr]); Ytr=np.concatenate([BY[i] for i in tr])
        m=GradientBoostingClassifier(n_estimators=200,max_depth=3,learning_rate=0.05,
                                     subsample=0.9,random_state=0).fit(Xtr,Ytr)
        sc[h]=m.predict_proba(BX[h][:,cols])[:,1]
    return sc
variants={"content_no_rep (with attn)": list(range(NF)),"content_no_rep_no_attn": keep_noattn}
SC={k:fit_loo(c) for k,c in variants.items()}
RESULT={}
for k,sc in SC.items():
    R={i:br(sc[i],usable(pos[i])[1]) for i in idx}
    RESULT[k]={**{f"hit@{kk}":float(np.mean([R[i]<kk for i in idx])) for kk in (1,5,10)},
               "mrr":float(np.mean([1.0/(R[i]+1) for i in idx])),"n":len(idx)}
Rr={i:br(np.arange(BN[i],dtype=float),usable(pos[i])[1]) for i in idx}
RESULT["recency"]={**{f"hit@{kk}":float(np.mean([Rr[i]<kk for i in idx])) for kk in (1,5,10)},
                   "mrr":float(np.mean([1.0/(Rr[i]+1) for i in idx])),"n":len(idx)}
print("=== ATTENTION-CONTRIBUTION ABLATION (n=%d) ==="%len(idx))
for k,v in RESULT.items(): print(f"{k:30s}", {kk:round(vv,3) for kk,vv in v.items()})
json.dump(RESULT, open(f"{RES}/loc_ablation.json","w"), indent=2)
print("WROTE results/loc_ablation.json")

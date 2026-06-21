"""step23: marginal value of LEARNED attention in the content-aware fusion.
Re-run the exact step21 LOO GBM fusion 3 ways:
  (A) attn = baseline head-L (unified_attn.json)
  (B) attn = strong head-L sharpen+inst (unified_attn_strong.json)
  (C) no-attn  (drop attn + attn_rank features entirely)
Same protocol (drop final tx as candidate & GT), same features otherwise.
Reports Hit@1/5/10 + MRR and paired bootstrap of (B - C) to test if the
improved attention adds value ON TOP of engineered features."""
import json, pickle, sys
import numpy as np
import vocab_def
sys.modules['__main__'].Vocab = vocab_def.Vocab
from sklearn.ensemble import GradientBoostingClassifier

DATA="data"; RES="results"
ptx=pickle.load(open(f"{DATA}/ptx_bags.pkl","rb")); pos=[b for b in ptx if b["label"]==1]
ATTN={"baseline":[np.array(a,float) for a in json.load(open(f"{RES}/unified_attn.json"))],
      "strong":  [np.array(a,float) for a in json.load(open(f"{RES}/unified_attn_strong.json"))]}

def usable(b):
    n=b["length"]; return n-1, [g for g in b["gt_idx"] if g<n-1]

def base_feats(b,nc,at,use_attn):
    amt=np.asarray(b["input_amounts"][:nc],float); io=np.asarray(b["input_io"][:nc],float)
    dt=np.asarray(b.get("delta_ts",[0]*nc)[:nc],float)
    ids=list(b["input_ids"][:nc]); n=max(nc,1)
    la=np.log1p(np.abs(amt)); z=(la-la.mean())/(la.std()+1e-9)
    rank=np.argsort(np.argsort(amt))/max(n-1,1)
    seen=set(); nov=[]
    for c in ids: nov.append(1.0 if c not in seen else 0.0); seen.add(c)
    nov=np.array(nov); inb=(io==2).astype(float); outb=(io==1).astype(float)
    prel=np.arange(nc)/max(n-1,1)
    dist_nl=1.0/(nc-np.arange(nc))
    zero_out=((amt==0)&(io==1)).astype(float)
    cummax=np.maximum.accumulate(np.abs(amt)); avc=np.abs(amt)/(cummax+1e-9)
    runmax=(np.abs(amt)>=cummax-1e-12).astype(float)
    lz=np.zeros(nc)
    for p in range(nc):
        s=la[max(0,p-3):p+4]; lz[p]=(la[p]-s.mean())/(s.std()+1e-9)
    ldt=np.log1p(dt)
    invalue=inb*la
    from collections import Counter
    cc=Counter(ids); cpf=np.array([cc[c] for c in ids],float)/n
    cols=[la,z,rank,nov,inb,prel]
    if use_attn:
        a=at[:nc] if len(at)>=nc else np.zeros(nc); ar=np.argsort(np.argsort(a))/max(n-1,1)
        cols+=[a,ar]
    cols+=[zero_out,outb,avc,runmax,lz,ldt,invalue,dist_nl,cpf]
    return np.column_stack(cols)

def run(mode):
    """mode in {baseline, strong, noattn}"""
    use_attn = mode!="noattn"
    at = ATTN["strong"] if mode=="strong" else (ATTN["baseline"] if mode=="baseline" else ATTN["baseline"])
    idx=[i for i,b in enumerate(pos) if usable(b)[0]>0 and usable(b)[1]]
    BX={}; BY={}; BN={}; BIDS={}
    for i in idx:
        nc,gt=usable(pos[i]); BX[i]=base_feats(pos[i],nc,at[i],use_attn); y=np.zeros(nc)
        for g in gt: y[g]=1
        BY[i]=y; BN[i]=nc; BIDS[i]=np.asarray(pos[i]["input_ids"][:nc])
    def cp_reputation(train_ids, hold_ids, alpha=5.0):
        from collections import defaultdict
        g=defaultdict(float); t=defaultdict(float); G=0.0; T=0.0
        for i in train_ids:
            ids=BIDS[i]; y=BY[i]
            for c,yy in zip(ids,y): t[c]+=1; g[c]+=yy; T+=1; G+=yy
        prior=G/max(T,1)
        return np.array([(g[c]+alpha*prior)/(t[c]+alpha) for c in hold_ids])
    def br(s,gt):
        o=np.argsort(-np.asarray(s)); rp={p:r for r,p in enumerate(o)}; return min(rp[g] for g in gt)
    from collections import defaultdict
    def rep_loo_train(tr, alpha=5.0):
        """For each bag i in tr, reputation of its counterparties using all OTHER
        train bags (sum-minus-i trick) -> O(n^2) not O(n^3). Returns {i: array}."""
        g=defaultdict(float); t=defaultdict(float); G=0.0; T=0.0
        for i in tr:
            for c,yy in zip(BIDS[i],BY[i]): t[c]+=1; g[c]+=yy; T+=1; G+=yy
        out={}
        for i in tr:
            gi=defaultdict(float); ti=defaultdict(float); Gi=G; Ti=T
            for c,yy in zip(BIDS[i],BY[i]): gi[c]+=yy; ti[c]+=1; Gi-=yy; Ti-=1
            prior=Gi/max(Ti,1)
            out[i]=np.array([ (g[c]-gi[c]+alpha*prior)/((t[c]-ti[c])+alpha) for c in BIDS[i] ])
        return out
    sc={}
    for h in idx:
        tr=[i for i in idx if i!=h]
        reptr=rep_loo_train(tr)
        Xtr=np.vstack([np.column_stack([BX[i], reptr[i]]) for i in tr])
        Ytr=np.concatenate([BY[i] for i in tr])
        Xh=np.column_stack([BX[h], cp_reputation(tr, BIDS[h])])
        m=GradientBoostingClassifier(n_estimators=200,max_depth=3,learning_rate=0.05,
                                     subsample=0.9,random_state=0).fit(Xtr,Ytr)
        sc[h]=m.predict_proba(Xh)[:,1]
    R={i:br(sc[i],usable(pos[i])[1]) for i in idx}
    res={**{f"hit@{k}":float(np.mean([R[i]<k for i in idx])) for k in (1,5,10)},
         "mrr":float(np.mean([1.0/(R[i]+1) for i in idx])),"n":len(idx)}
    return res, R, idx

print("running 3 fusion variants (LOO GBM)...")
RES_B,RB,idx=run("baseline")
RES_S,RS,_=run("strong")
RES_N,RN,_=run("noattn")
print("\n=== FUSION MARGINAL VALUE OF ATTENTION (n=%d) ==="%RES_N["n"])
for name,r in [("fusion + baseline attn",RES_B),("fusion + STRONG attn",RES_S),("fusion + NO attn",RES_N)]:
    print(f"{name:24s}", {k:round(v,3) for k,v in r.items() if k!='n'})

arr=np.array(idx)
def paired(Ra,Rb,k,nboot=8000,seed=7):
    rs=np.random.RandomState(seed); d=[]
    for _ in range(nboot):
        bs=rs.choice(arr,len(arr),True)
        if k=="mrr":
            d.append(np.mean([1.0/(Ra[i]+1) for i in bs])-np.mean([1.0/(Rb[i]+1) for i in bs]))
        else:
            d.append(np.mean([Ra[i]<k for i in bs])-np.mean([Rb[i]<k for i in bs]))
    d=np.array(d); return {"mean_diff":float(d.mean()),"p_not_better":float(np.mean(d<=0)),
                           "ci":[float(np.percentile(d,2.5)),float(np.percentile(d,97.5))]}
print("\n=== PAIRED: STRONG-attn fusion  vs  NO-attn fusion ===")
PSN={f"hit@{k}":paired(RS,RN,k) for k in (1,5,10)}; PSN["mrr"]=paired(RS,RN,"mrr")
for k in ("hit@1","hit@5","hit@10","mrr"): print(f"{k:7s}: {PSN[k]}")
print("\n=== PAIRED: STRONG-attn fusion  vs  BASELINE-attn fusion ===")
PSB={f"hit@{k}":paired(RS,RB,k) for k in (1,5,10)}; PSB["mrr"]=paired(RS,RB,"mrr")
for k in ("hit@1","hit@5","hit@10","mrr"): print(f"{k:7s}: {PSB[k]}")

json.dump({"fusion_baseline_attn":RES_B,"fusion_strong_attn":RES_S,"fusion_no_attn":RES_N,
           "paired_strong_vs_noattn":PSN,"paired_strong_vs_baseline":PSB},
          open(f"{RES}/loc_fusion_marginal.json","w"), indent=2)
print("\nWROTE results/loc_fusion_marginal.json")

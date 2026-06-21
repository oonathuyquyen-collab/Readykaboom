"""step8b_final.py -- honest A* analysis from CACHED artifacts (no retrain).
Fixes: (1) bootstrap over 292 UNIQUE scammer EOAs (true independent units; PTX is zero-shot
 -> no train/test leakage). (2) REMOVE detection-cutoff artifact: every bag's LAST tx is a GT
 (crawl stopped at detection) -> exclude final tx from candidates AND gt for ALL methods so
 the recency baseline is fair. Keeps Hit@K. NO random baseline.
"""
import sys, os, json, pickle, csv, collections
import numpy as np
import vocab_def
sys.modules['__main__'].Vocab = vocab_def.Vocab
from sklearn.metrics import roc_auc_score, average_precision_score

DATA="data"; RES="results"
ptx=pickle.load(open(f"{DATA}/ptx_bags.pkl","rb"))
norm=pickle.load(open(f"{DATA}/normal_eoa_neg.pkl","rb"))
pos=[b for b in ptx if b["label"]==1]; ptx_neg=[b for b in ptx if b["label"]==0]
allb=pos+ptx_neg+norm
y=np.array([b["label"] for b in allb]); src=np.array([b.get("source","ptx_benign") for b in allb])
ntx=np.array([b["ntx_full"] for b in allb])
pacct=np.load(f"{RES}/unified_pacct.npy")
attn=[np.array(a,float) for a in json.load(open(f"{RES}/unified_attn.json"))]
aud={}
for r in csv.DictReader(open(f"{DATA}/step1_audit.csv")): aud[r["txhash"].lower()]=r
def cat(b):
    c=collections.Counter()
    for gi in b["gt_idx"]:
        h=str(b["hashes"][gi]).lower()
        if h in aud: c[aud[h]["scam_type"]]+=1
    return c.most_common(1)[0][0] if c else "unknown"
def ts(b):
    t=[int(aud[str(h).lower()]["ts"]) for h in b["hashes"] if str(h).lower() in aud]
    return min(t) if t else 0
pcat=np.array([cat(b) for b in pos]); pts=np.array([ts(b) for b in pos])

# ---- ARTIFACT REMOVAL: drop each bag's final tx from candidates + GT ----
def usable(i):
    n=pos[i]["length"]; gt=[g for g in pos[i]["gt_idx"] if g < n-1]  # exclude last index
    return n-1, gt   # candidate count = n-1, interior GT
def best_rank(scores, gt):
    if not gt: return None
    order=np.argsort(-np.asarray(scores))   # high score first
    rankpos={p:r for r,p in enumerate(order)}
    return min(rankpos[g] for g in gt)
def hitk(idxs, scorer, ks=(1,5,10)):
    bests=[]
    for i in idxs:
        ncand,gt=usable(i)
        if ncand<=0 or not gt: continue
        s=scorer(i,ncand); bests.append(best_rank(s,gt))
    bests=[b for b in bests if b is not None]
    out={f"hit@{k}": float(np.mean([b<k for b in bests])) for k in ks}
    out["mrr"]=float(np.mean([1.0/(b+1) for b in bests])); out["n"]=len(bests)
    return out,bests
# scorers over first ncand txs (exclude last)
def s_headL(i,nc): return attn[i][:nc]
def s_amount(i,nc): return np.log1p(np.abs(np.array(pos[i]["input_amounts"][:nc],float)))
def s_recency(i,nc): return np.arange(nc,dtype=float)
def s_novelty(i,nc):
    seen=set(); o=[]
    for cid in pos[i]["input_ids"][:nc]:
        o.append(1.0 if cid not in seen else 0.0); seen.add(cid)
    return np.array(o)
def s_degree(i,nc):
    cnt=collections.Counter(pos[i]["input_ids"][:nc]); return np.array([-cnt[c] for c in pos[i]["input_ids"][:nc]],float)
methods={"headL_unified":s_headL,"amount":s_amount,"recency":s_recency,"novelty":s_novelty,"degree":s_degree}

allidx=list(range(len(pos)))
needle=[i for i in allidx if len([g for g in pos[i]["gt_idx"] if g<pos[i]["length"]-1])<=2 and pos[i]["length"]>=20]
def locblock(idxs):
    return {m:hitk(idxs,f)[0] for m,f in methods.items()}
full_loc=locblock(allidx); needle_loc=locblock(needle)

# ---- account bootstrap over 292 unique EOAs (and all neg) ----
def auc_ci(ysub,psub,nboot=2000):
    rs=np.random.RandomState(0); n=len(ysub); st=[]
    for _ in range(nboot):
        idx=rs.randint(0,n,n)
        if len(set(ysub[idx]))==2: st.append(roc_auc_score(ysub[idx],psub[idx]))
    return [float(np.percentile(st,2.5)),float(np.percentile(st,97.5))]
cross_auc=float(roc_auc_score(y,pacct)); cross_aupr=float(average_precision_score(y,pacct))
cross_ci=auc_ci(y,pacct)
by_source={}
for sn in ["ptx_benign","normal_eoa"]:
    sel=(y==1)|((y==0)&(src==sn)); 
    by_source[sn]={"auc":float(roc_auc_score(y[sel],pacct[sel])),"auc_ci":auc_ci(y[sel],pacct[sel]),
                   "aupr":float(average_precision_score(y[sel],pacct[sel])),"n_neg":int(((y==0)&(src==sn)).sum())}
strat={}
for lo,hi,nm in [(3,20,"3-20"),(20,100,"20-100"),(100,10000,"100+")]:
    sel=(ntx>=lo)&(ntx<hi)
    if sel.sum()>5 and len(set(y[sel]))==2:
        strat[nm]={"auc":float(roc_auc_score(y[sel],pacct[sel])),"n":int(sel.sum()),"pos":int(y[sel].sum())}

# ---- permutation tests: headL vs each baseline (needle, hit@1, paired sign-flip) ----
def perm(idxs,m1,m2,k=1,nperm=5000):
    _,b1=hitk(idxs,methods[m1]); _,b2=hitk(idxs,methods[m2])
    # align: recompute paired on common usable bags
    pairs=[]
    for i in idxs:
        nc,gt=usable(i)
        if nc<=0 or not gt: continue
        r1=best_rank(methods[m1](i,nc),gt); r2=best_rank(methods[m2](i,nc),gt)
        pairs.append(((r1<k)-(r2<k)))
    d=np.array(pairs,float); obs=d.mean(); rs=np.random.RandomState(0); c=0
    for _ in range(nperm):
        if (d*rs.choice([-1,1],len(d))).mean()>=obs: c+=1
    return float(obs),float((c+1)/(nperm+1))
perm_res={m:dict(zip(["delta_hit@1","p"],perm(needle,"headL_unified",m))) for m in ["amount","recency","novelty","degree"]}
# headL needle CI (bootstrap over usable needle bags)
def hit_ci(idxs,m,k,nboot=2000):
    _,bests=hitk(idxs,methods[m]); bests=np.array(bests); rs=np.random.RandomState(0); st=[]
    for _ in range(nboot):
        idx=rs.randint(0,len(bests),len(bests)); st.append((bests[idx]<k).mean())
    return [float(np.percentile(st,2.5)),float(np.percentile(st,97.5))]
headL_ci={f"hit@{k}":hit_ci(needle,"headL_unified",k) for k in (1,5,10)}

# ---- OOD cross-mechanism + temporal (zero-shot partitions) ----
nneg=len(allb)-len(pos)
def ood_block(pidx):
    if len(pidx)<5: return None
    pc=np.concatenate([pacct[:len(pos)][pidx],pacct[len(pos):]])
    yy=np.concatenate([np.ones(len(pidx)),np.zeros(nneg)])
    nd=[i for i in pidx if len([g for g in pos[i]["gt_idx"] if g<pos[i]["length"]-1])<=2 and pos[i]["length"]>=20]
    lb=hitk(nd,s_headL)[0] if len(nd)>=5 else {}
    return {"n_pos":int(len(pidx)),"auc":float(roc_auc_score(yy,pc)),"aupr":float(average_precision_score(yy,pc)),
            "n_needle":len(nd),"headL_hit@1":lb.get("hit@1"),"headL_hit@5":lb.get("hit@5"),"headL_hit@10":lb.get("hit@10")}
ood_cat={c:ood_block(np.where(pcat==c)[0]) for c in ["payable_function","ice_phishing","address_poisoning"]}
ood_cat={k:v for k,v in ood_cat.items() if v}
med=int(np.median(pts[pts>0]))
ood_time={nm:ood_block(np.where(sel&(pts>0))[0]) for nm,sel in [("early",pts<=med),("late",pts>med)]}
ood_time={k:v for k,v in ood_time.items() if v}

out={"counts":{"pos_unique_eoa":len(set(str(b['account']).lower() for b in pos)),"pos":len(pos),
      "ptx_neg":len(ptx_neg),"normal_eoa_neg":len(norm),"needle_n":len(needle),
      "artifact":"last tx of every bag is GT (detection cutoff) -> EXCLUDED from candidates+GT for all methods"},
     "account":{"cross_domain":{"auc_combined":cross_auc,"auc_ci":cross_ci,"aupr_combined":cross_aupr,
                 "by_source":by_source,"stratified":strat}},
     "localization":{"full":full_loc,"needle":needle_loc,"headL_needle_ci":headL_ci,"permutation_vs_baselines":perm_res},
     "ood":{"cross_mechanism":ood_cat,"temporal":ood_time,"temporal_median_ts":med},
     "category_dist":dict(collections.Counter(pcat))}
json.dump(out,open(f"{RES}/final_results.json","w"),indent=2)
print("WROTE results/final_results.json (artifact-corrected, no random)")
print("FULL  headL:",full_loc["headL_unified"]," recency:",full_loc["recency"])
print("NEEDLE headL:",needle_loc["headL_unified"])
for m in ["amount","recency","novelty","degree"]: print("  needle",m,needle_loc[m])
print("PERM:",perm_res)
print("cross combined",round(cross_auc,3),cross_ci," by_source:",{k:(round(v['auc'],3),v['n_neg']) for k,v in by_source.items()})
print("OOD cat:",{k:(round(v['auc'],3),v['headL_hit@5']) for k,v in ood_cat.items()})
print("OOD time:",{k:(round(v['auc'],3),v['headL_hit@5']) for k,v in ood_time.items()})

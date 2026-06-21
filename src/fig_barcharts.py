"""Academic bar charts for every comparison in the paper.
Numbers match the paper tables exactly (account: 3-seed; localization: n=101 protocol)."""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":10,"figure.dpi":170,"axes.grid":True,
                     "grid.alpha":0.25,"grid.linestyle":":","font.family":"DejaVu Sans",
                     "axes.axisbelow":True})
OURS="#c0392b"; BASE="#5b8fb9"; HL="#9b59b6"; REC="#e67e22"

def label_bars(ax,bars,vals,fs=7.0,fmt="%.3f",dy=0.006):
    for b,v in zip(bars,vals):
        ax.text(b.get_x()+b.get_width()/2, v+dy, (fmt%v).lstrip("0") if v<1 else fmt%v,
                ha="center",va="bottom",fontsize=fs)

# ---------- 1. Account-level Precision / Recall / F1 (grouped) ----------
methods=["BERT4ETH","ZipZap","TSGN","LMAE4Eth","GatedMIL","TransMIL","CLAM","Unified\n(ours)"]
P=[0.749,0.737,0.722,0.736,0.736,0.719,0.729,0.709]
R=[0.722,0.687,0.733,0.765,0.721,0.745,0.721,0.766]
F=[0.734,0.711,0.727,0.750,0.728,0.729,0.724,0.735]
x=np.arange(len(methods)); w=0.27
fig,ax=plt.subplots(figsize=(9.2,3.6))
b1=ax.bar(x-w,P,w,label="Precision",color="#7fb3d5")
b2=ax.bar(x,R,w,label="Recall",color="#f0b27a")
b3=ax.bar(x+w,F,w,label="F1",color="#82c596")
ax.set_xticks(x); ax.set_xticklabels(methods,fontsize=8.5)
ax.set_ylim(0.60,0.80); ax.set_ylabel("Score")
ax.set_title("Account-level in-domain Precision / Recall / F1 (3-seed mean)")
# highlight ours
ax.axvspan(x[-1]-0.45,x[-1]+0.45,color=OURS,alpha=0.07)
ax.legend(ncol=3,fontsize=8.5,loc="upper center",bbox_to_anchor=(0.5,1.16),frameon=False)
for bb,vv in [(b1,P),(b2,R),(b3,F)]: label_bars(ax,bb,vv,fs=6.0,dy=0.003)
fig.tight_layout(); fig.savefig("figures/fig_bar_account_prf.png",bbox_inches="tight"); plt.close(fig)

# ---------- 2. Account-level cross-domain hard AUC ----------
m2=["Mean-pool","BERT4ETH","Max-pool","Gated-MIL","ZipZap","TSGN","Unified\n(ours)","LMAE4Eth"]
auc=[0.885,0.836,0.779,0.803,0.776,0.760,0.725,0.708]
cols=[BASE]*len(m2); cols[m2.index("Unified\n(ours)")]=OURS
fig,ax=plt.subplots(figsize=(8.2,3.5))
bars=ax.bar(m2,auc,0.62,color=cols)
ax.set_ylim(0.6,0.95); ax.set_ylabel("AUC (hard PTXPhish benign, n=80)")
ax.set_title("Cross-domain hard-negative AUC (honest: ours is competitive, not top)")
label_bars(ax,bars,auc,fs=7.5,dy=0.004)
ax.set_xticklabels(m2,fontsize=8.5)
fig.tight_layout(); fig.savefig("figures/fig_bar_account_auc.png",bbox_inches="tight"); plt.close(fig)

# ---------- 3. Transaction localization Hit@1 / Hit@5 / MRR (n=101) ----------
lm=["Amount","Novelty","Degree","TransMIL","Gated-MIL","CLAM","Head-L\nbase",
    "Head-L\nstrong","Recency","Content-aware\n(ours)"]
h1=[0.347,0.356,0.465,0.406,0.446,0.455,0.416,0.515,0.693,0.832]
h5=[0.584,0.782,0.772,0.673,0.752,0.792,0.752,0.851,0.921,0.931]
mr=[0.447,0.551,0.607,0.533,0.598,0.599,0.576,0.646,0.799,0.880]
x=np.arange(len(lm)); w=0.27
fig,ax=plt.subplots(figsize=(10.4,3.8))
b1=ax.bar(x-w,h1,w,label="Hit@1",color="#7fb3d5")
b2=ax.bar(x,h5,w,label="Hit@5",color="#f0b27a")
b3=ax.bar(x+w,mr,w,label="MRR",color="#82c596")
ax.axvspan(x[-1]-0.45,x[-1]+0.45,color=OURS,alpha=0.08)
ax.set_xticks(x); ax.set_xticklabels(lm,fontsize=8)
ax.set_ylim(0,1.05); ax.set_ylabel("Score")
ax.set_title("Transaction-level localization on PTXPhish (n=101, artifact-removed)")
ax.legend(ncol=3,fontsize=8.5,loc="upper left",frameon=False)
ax.text(x[-1]-w,0.832+0.02,"$p{=}0.002$",fontsize=7.5,color=OURS,ha="center")
label_bars(ax,b1,h1,fs=5.6,dy=0.004)
fig.tight_layout(); fig.savefig("figures/fig_bar_loc.png",bbox_inches="tight"); plt.close(fig)

# ---------- 4. Attention marginal value in fusion ----------
av=["Head-L\nbase","Head-L\nstrong","Fusion\n+base attn","Fusion\n+strong attn","Fusion\n$-$attn"]
hv=[0.416,0.515,0.832,0.782,0.792]
cols4=[HL,"#7d3c98",OURS,"#a93226","#cd6155"]
fig,ax=plt.subplots(figsize=(7.6,3.5))
bars=ax.bar(av,hv,0.6,color=cols4)
ax.axhline(0.693,ls="--",color=REC,lw=1.4)
ax.text(4.45,0.693+0.012,"recency 0.693",fontsize=7.5,color=REC,ha="right")
ax.set_ylim(0,1.0); ax.set_ylabel("Hit@1")
ax.set_title("Attention helps alone (+24%) but is subsumed by features in fusion ($p{=}0.68$)")
label_bars(ax,bars,hv,fs=7.5,dy=0.006)
ax.set_xticklabels(av,fontsize=8)
fig.tight_layout(); fig.savefig("figures/fig_bar_attn.png",bbox_inches="tight"); plt.close(fig)

print("wrote 4 bar charts")

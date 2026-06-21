import json,numpy as np,matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
d=json.load(open("results/final_results.json"))
plt.rcParams.update({"font.size":11,"figure.dpi":150,"axes.grid":True,"grid.alpha":0.3})

# FIG A: cross-domain account-level AUC by negative source
cd=d["account"]["cross_domain"]
labels=["PTX-benign\n(hard, n=%d)"%cd["by_source"]["ptx_benign"]["n_neg"],
        "Normal-EOA\n(n=%d)"%cd["by_source"]["normal_eoa"]["n_neg"],"Combined"]
vals=[cd["by_source"]["ptx_benign"]["auc"],cd["by_source"]["normal_eoa"]["auc"],cd["auc_combined"]]
fig,ax=plt.subplots(figsize=(5.2,3.4))
b=ax.bar(labels,vals,color=["#c0392b","#7f8c8d","#2980b9"])
ax.axhline(0.5,ls="--",c="k",lw=0.8,label="chance")
ax.set_ylim(0.4,1.02); ax.set_ylabel("ROC-AUC"); ax.set_title("Zero-shot cross-domain account detection")
for r,v in zip(b,vals): ax.text(r.get_x()+r.get_width()/2,v+0.01,f"{v:.3f}",ha="center",fontsize=9)
ax.legend(fontsize=8); fig.tight_layout(); fig.savefig("figures/fig_account.png"); plt.close()

# FIG B: OOD generalization
cm=d["ood"]["cross_mechanism"]; tm=d["ood"]["temporal"]
names=list(cm.keys())+["early","late"]; aucs=[cm[k]["auc"] for k in cm]+[tm[k]["auc"] for k in tm]
nice={"payable_function":"payable-fn","ice_phishing":"ice-phish","address_poisoning":"poisoning","early":"time:early","late":"time:late"}
fig,ax=plt.subplots(figsize=(5.6,3.4))
cols=["#27ae60"]*len(cm)+["#8e44ad"]*len(tm)
b=ax.bar([nice[n] for n in names],aucs,color=cols)
ax.set_ylim(0.4,1.02); ax.axhline(0.5,ls="--",c="k",lw=0.8)
ax.set_ylabel("ROC-AUC"); ax.set_title("OOD generalization (zero-shot): mechanism & time")
for r,v in zip(b,aucs): ax.text(r.get_x()+r.get_width()/2,v+0.01,f"{v:.3f}",ha="center",fontsize=8)
plt.xticks(rotation=15); fig.tight_layout(); fig.savefig("figures/fig_ood.png"); plt.close()

# FIG C: localization Hit@K (full set, interior GT) head-L vs baselines
fl=d["localization"]["full"]; order=["headL_unified","recency","amount","novelty","degree"]
nice2={"headL_unified":"Head-L (ours)","recency":"recency","amount":"amount","novelty":"novelty","degree":"degree"}
ks=["hit@1","hit@5","hit@10"]; x=np.arange(len(ks)); w=0.16
fig,ax=plt.subplots(figsize=(6.4,3.6))
cmap=["#2980b9","#e67e22","#95a5a6","#16a085","#9b59b6"]
for j,m in enumerate(order):
    ax.bar(x+(j-2)*w,[fl[m][k] for k in ks],w,label=nice2[m],color=cmap[j])
ax.set_xticks(x); ax.set_xticklabels([k.upper() for k in ks]); ax.set_ylabel("Hit@K")
ax.set_title("Transaction localization (n=%d interior-GT bags)"%fl["headL_unified"]["n"])
ax.legend(fontsize=8,ncol=2); ax.set_ylim(0,1.0); fig.tight_layout(); fig.savefig("figures/fig_loc.png"); plt.close()
print("figures written:",[f for f in __import__("os").listdir("figures") if f.startswith("fig_")])

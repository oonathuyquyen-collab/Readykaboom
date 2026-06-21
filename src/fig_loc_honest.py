import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":10,"figure.dpi":160,"axes.grid":True,"grid.alpha":0.25,
                     "font.family":"DejaVu Sans"})
rig=json.load(open("results/loc_rigorous.json"))["full"]
abl=json.load(open("results/loc_ablation.json"))
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10.6,3.9))

# Panel A: ranker comparison at Hit@1/5/10
order=["amount","attn_only","recency","content_aware"]
nice={"amount":"Amount prior","attn_only":"Head-L attn only",
      "recency":"Recency prior","content_aware":"Content-aware (ours)"}
cols={"amount":"#95a5a6","attn_only":"#9b59b6","recency":"#e67e22","content_aware":"#2980b9"}
ks=["hit@1","hit@5","hit@10"]; x=np.arange(len(ks)); w=0.2
for j,m in enumerate(order):
    ax1.bar(x+(j-1.5)*w,[rig[m][k] for k in ks],w,label=nice[m],color=cols[m])
ax1.set_xticks(x); ax1.set_xticklabels(["Hit@1","Hit@5","Hit@10"])
ax1.set_ylabel("Score"); ax1.set_ylim(0,1.02)
ax1.set_title("(a) Transaction localization (n=101)")
ax1.legend(fontsize=7.5,ncol=2,loc="upper left")
ax1.text(0.0-0.30,rig["content_aware"]["hit@1"]+0.015,"$p{=}0.002$",fontsize=8,color="#2980b9")

# Panel B: attention-contribution ablation (Hit@1 + MRR)
labels=["Content\n(+attn)","Content\n($-$attn)","Recency"]
keys=["content_no_rep (with attn)","content_no_rep_no_attn","recency"]
h1=[abl[k]["hit@1"] for k in keys]; mrr=[abl[k]["mrr"] for k in keys]
xx=np.arange(len(labels)); w2=0.34
b1=ax2.bar(xx-w2/2,h1,w2,label="Hit@1",color="#2980b9")
b2=ax2.bar(xx+w2/2,mrr,w2,label="MRR",color="#16a085")
ax2.set_xticks(xx); ax2.set_xticklabels(labels); ax2.set_ylim(0,1.02)
ax2.set_title("(b) Removing learned attention does not hurt")
ax2.legend(fontsize=8,loc="lower left")
for b in list(b1)+list(b2):
    ax2.text(b.get_x()+b.get_width()/2,b.get_height()+0.012,f"{b.get_height():.3f}",
             ha="center",fontsize=7.5)
fig.tight_layout(); fig.savefig("figures/fig_loc.png"); print("wrote figures/fig_loc.png")

import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":10,"figure.dpi":160,"axes.grid":True,"grid.alpha":0.25,
                     "font.family":"DejaVu Sans"})
rig=json.load(open("results/loc_rigorous.json"))["full"]
abl=json.load(open("results/loc_ablation.json"))
mar=json.load(open("results/loc_fusion_marginal.json"))
hls=json.load(open("results/headL_strong.json"))
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

# Panel B: attention helps alone, but is subsumed by features in the fusion
hb=hls["configs"]["baseline"]["loc_full"]["hit@1"]
hs2=hls["configs"]["sharpen+inst"]["loc_full"]["hit@1"]
fb=mar["fusion_baseline_attn"]["hit@1"]; fs=mar["fusion_strong_attn"]["hit@1"]
fn=mar["fusion_no_attn"]["hit@1"]
labels=["Head-L\nbase","Head-L\nstrong","Fusion\n+base","Fusion\n+strong","Fusion\n$-$attn"]
vals=[hb,hs2,fb,fs,fn]
cols2=["#9b59b6","#8e44ad","#2980b9","#2471a3","#5dade2"]
xx=np.arange(len(labels))
bb=ax2.bar(xx,vals,0.62,color=cols2)
ax2.axhline(hls["recency_full"]["hit@1"],ls="--",color="#e67e22",lw=1.3)
ax2.text(4.4,hls["recency_full"]["hit@1"]+0.012,"recency",fontsize=7.5,color="#e67e22",ha="right")
ax2.set_xticks(xx); ax2.set_xticklabels(labels,fontsize=8); ax2.set_ylim(0,1.02)
ax2.set_ylabel("Hit@1")
ax2.set_title("(b) Attention helps alone, subsumed in fusion")
ax2.annotate("",xy=(1,hs2+0.03),xytext=(0,hb+0.03),
             arrowprops=dict(arrowstyle="->",color="#8e44ad",lw=1.4))
ax2.text(0.5,max(hb,hs2)+0.07,"+24%",fontsize=8,color="#8e44ad",ha="center")
for b in bb:
    ax2.text(b.get_x()+b.get_width()/2,b.get_height()+0.012,f"{b.get_height():.2f}",
             ha="center",fontsize=7.5)
fig.tight_layout(); fig.savefig("figures/fig_loc.png"); print("wrote figures/fig_loc.png")

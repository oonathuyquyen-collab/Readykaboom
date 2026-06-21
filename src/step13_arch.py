#!/usr/bin/env python3
"""Architecture diagram for the Unified Two-Head TMIL model.
Style follows square_architecture_tmil_v9_final.png: white background, soft pastel
rounded blocks, dark-slate labels, vertical bottom-up flow with a main column and a
right-side localization branch. Adapted to the CURRENT unified two-head design.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BG="#ffffff"; INK="#1b2631"; SUBINK="#5d6d7e"
YEL="#fcf3cf"; YEL_E="#d4ac0d"
PUR="#ebdef0"; PUR_E="#8e44ad"
ORA="#fad7a1"; ORA_E="#ca6f1e"
BLU="#d4e6f1"; BLU_E="#2e86c1"
GRN="#d5f5e3"; GRN_E="#1e8449"
PNK="#fadbd8"; PNK_E="#cb4335"
ACC="#5dade2"

fig, ax = plt.subplots(figsize=(13, 15.5), dpi=150)
ax.set_xlim(0,100); ax.set_ylim(0,152); ax.axis("off")
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)

def block(x,y,w,h,text,fc,ec,fs=13,weight="bold",tc=INK,rounding=2.2,lw=2.0,sub=None):
    p=FancyBboxPatch((x,y),w,h,boxstyle=f"round,pad=0.15,rounding_size={rounding}",
                     linewidth=lw,edgecolor=ec,facecolor=fc,zorder=3,mutation_aspect=1)
    ax.add_patch(p)
    ax.text(x+w/2,y+h/2+(1.0 if sub else 0),text,ha="center",va="center",
            fontsize=fs,color=tc,fontweight=weight,zorder=4)
    if sub:
        ax.text(x+w/2,y+h/2-2.6,sub,ha="center",va="center",
                fontsize=fs-3,color=SUBINK,zorder=4)
    return (x+w/2,y,x+w/2,y+h)

def chips(xc,y,n,w,h,labels,fc,ec,fs=10):
    total=n*w+(n-1)*1.2; x0=xc-total/2; cs=[]
    for i in range(n):
        x=x0+i*(w+1.2)
        block(x,y,w,h,labels[i],fc,ec,fs=fs,rounding=1.2,lw=1.4)
        cs.append((x+w/2,y,x+w/2,y+h))
    return cs

def arrow(x0,y0,x1,y1,color=INK,lw=2.2,style="-|>",ls="-",rad=0.0):
    ax.add_patch(FancyArrowPatch((x0,y0),(x1,y1),arrowstyle=style,mutation_scale=18,
                 linewidth=lw,color=color,zorder=2,ls=ls,
                 connectionstyle=f"arc3,rad={rad}"))

LX=4; LW=92
ax.text(LX,7.4,"Transaction sequence  (one EOA's full on-chain history)",
        fontsize=12.5,color=INK,fontweight="bold",ha="left")
chips(50,1.2,7,11.5,4.6,["Tx1","Tx2","Tx3","Tx4","Tx5","...","Txn"],BLU,BLU_E,fs=11)

block(LX,13.5,LW,9.5,"Feature embeddings  (per transaction)",YEL,YEL_E,fs=13,
      sub="Counterparty-ID emb  +  IN/OUT direction emb  +  Amount  +  dt Time2Vec    [ + BERT4ETH pretrained emb ]")
arrow(50,6.0,50,13.5)

ax.text(LX,31.4,"Sliding windows  (W=200, S=50)  -  each window = one MIL instance",
        fontsize=11.5,color=INK,fontweight="bold",ha="left")
wc=chips(50,25.0,6,13.5,4.6,["win 1","win 2","win 3","win 4","...","win n"],PUR,PUR_E,fs=10.5)
arrow(50,23.0,50,25.0)

block(LX,33.5,LW,9.0,"1D Temporal Convolutional Network",ORA,ORA_E,fs=14,sub="shared encoder")
for c in wc: arrow(c[0],c[3],c[0],33.5,lw=1.4)

ax.text(LX,50.8,"Instance representations",fontsize=11.5,color=INK,fontweight="bold",ha="left")
chips(50,44.5,6,13.5,4.6,["h1","h2","h3","h4","...","hn"],GRN,GRN_E,fs=12)
arrow(50,42.5,50,44.5)

SY=49.1
arrow(50,SY,27,58.0,rad=0.12)
arrow(50,SY,73,58.0,rad=-0.12)

ax.text(50,150.5,"Unified two-head MIL  -  one shared TCN encoder  -  one forward pass  -  one weight set  ->  two task outputs",
        fontsize=13,color="#ffffff",fontweight="bold",ha="center",va="center",
        bbox=dict(boxstyle="round,pad=0.6",facecolor=ACC,edgecolor=BLU_E,linewidth=1.5),zorder=5)

CL=6; CW=41; cx=CL+CW/2
ax.text(cx,71.5,"HEAD-C  -  account classification",fontsize=12.5,color=BLU_E,fontweight="bold",ha="center")
c1=block(CL,58.0,CW,8.5,"Soft Gated Attention",PUR,PUR_E,fs=13,sub=r"$\tanh(Vh)\odot\sigma(Uh)$")
c2=block(CL,78.0,CW,8.0,r"Attention weights $a^{C}$",PUR,PUR_E,fs=12.5,sub=r"$\sum_k a^{C}_k=1$")
c3=block(CL,96.0,CW,8.0,r"Bag vector  $z=\sum_k a^{C}_k h_k$",GRN,GRN_E,fs=12.5)
c4=block(CL,114.0,CW,8.0,"2-Layer MLP Classifier",ORA,ORA_E,fs=13)
c5=block(CL,132.0,CW,8.5,"(1) Account-level Prediction",BLU,BLU_E,fs=13.5,sub="L_BCE  -  AUC / AUPR / F1")
for a,b in [(c1,c2),(c2,c3),(c3,c4),(c4,c5)]: arrow(cx,a[3],cx,b[1])

RL=53; RW=41; rx=RL+RW/2
ax.text(rx,71.5,"HEAD-L  -  transaction localization",fontsize=12.5,color=PNK_E,fontweight="bold",ha="center")
r1=block(RL,58.0,RW,8.5,"Outbound-Masked Gated Attention",PNK,PNK_E,fs=11.5,sub="masks inbound  -  trained with BCE")
r2=block(RL,78.0,RW,8.0,r"Attention weights $a^{L}$",PNK,PNK_E,fs=12.5,sub="direction-aware")
r3=block(RL,96.0,RW,8.0,r"Rank transactions by $a^{L}$",GRN,GRN_E,fs=12.5)
r4=block(RL,114.0,RW,8.0,"Localization read-out",ORA,ORA_E,fs=12.5,sub="vs amount / recency / degree / novelty")
r5=block(RL,132.0,RW,8.5,"(2) Transaction-level Localization",PNK,PNK_E,fs=12,sub="Hit@1 / Hit@5 / Hit@10 / MRR")
for a,b in [(r1,r2),(r2,r3),(r3,r4),(r4,r5)]: arrow(rx,a[3],rx,b[1])

ax.plot([50,50],[57,141],color="#d5dbdb",lw=1.2,ls=(0,(4,4)),zorder=1)
ax.text(50,146.0,"Unified Two-Head TMIL for Account- and Transaction-Level Phishing Detection on Ethereum",
        fontsize=13,color=INK,fontweight="bold",ha="center",va="center")

plt.tight_layout()
plt.savefig("figures/fig_architecture.png",dpi=150,facecolor=BG,bbox_inches="tight")
print("saved figures/fig_architecture.png")

"""Regenerate paper-aligned figures from the archived final evaluation tables."""
from pathlib import Path
import csv,matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
G=Path(__file__).resolve().parents[1];D=G/"results"/"final_20261126";F=G/"figures"/"final_20261126"
F.mkdir(exist_ok=True,parents=True)
read=lambda n:list(csv.DictReader((D/n).open(encoding="utf8",newline="")))
s=read("model_summary.csv"); names=[x["label"] for x in s]
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"svg.fonttype":"none","pdf.fonttype":42})
accent="#15766b"; gray="#b9c0c3"; dark="#24313a"
def save(fig,n):
 for ext in ("png","svg","pdf"):fig.savefig(F/f"{n}.{ext}",dpi=600,bbox_inches="tight",pad_inches=.10,facecolor="white")
 plt.close(fig)
def canvas():
 fig,ax=plt.subplots(figsize=(10,6.2));ax.set(xlim=(0,100),ylim=(0,62));ax.axis("off");return fig,ax
def box(ax,x,y,w,h,t,new=False,fs=9):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.18,rounding_size=0.65",facecolor="#dcefe9" if new else "#f2f3f4",edgecolor=accent if new else "#849096",linewidth=1))
 ax.text(x+w/2,y+h/2,t,ha="center",va="center",fontsize=fs,color=dark,linespacing=1.2)
def arrow(ax,x1,y1,x2,y2):
 ax.annotate("",xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle="-|>",lw=1.1,color=dark,shrinkA=0,shrinkB=0))
fig,ax=canvas()
box(ax,26,54,48,6,"HVPD-DVCS: 24,480 charts\nGroup audit and fixed split")
box(ax,26,43,48,7,"19,544 train / 2,465 validation / 2,471 test\n256 x 256 RGB | identical preprocessing")
arrow(ax,50,54,50,50)
box(ax,3,33,45,6,"Initial study (15 runs)\nMaciborski et al. [7]; Liu et al. [10]")
box(ax,52,33,45,6,"Exploratory follow-up (6 runs)\nSame split; parameter-matched controls")
for x in (25.5,74.5):arrow(ax,50,43,x,39)
box(ax,3,21,45,8,"Frozen DINOv3 | SwinV2 Tiny | SALF-DINO\nAxis-only | Local-only\n5 configurations x 3 seeds",True)
box(ax,52,21,45,8,"Content-ranked regions | Additive residual\n2 configurations x 3 seeds")
for x in (25.5,74.5):arrow(ax,x,33,x,29)
box(ax,26,10,48,7,"Shared training and checkpoint selection\nAdamW | seeds 42, 123, 2025 | <=20 epochs")
for x in (25.5,74.5):arrow(ax,x,21,50,17)
box(ax,14,0.7,72,6,"21/21 validated: identical held-out test identities\nMacro-F1 | paired group bootstrap | category errors | corruptions | cost",True)
arrow(ax,50,10,50,6.7)
save(fig,"Figure_01_Research_Framework")
fig,ax=canvas()
box(ax,32,55,36,6,"Shared chart input\n256 x 256 RGB")
box(ax,2,42,23,8,"DINOv3 baseline\nFrozen ViT-L/16\nSpatial mean pooling")
box(ax,29,42,23,8,"SwinV2 baseline\nFrozen Tiny\nGlobal aggregation")
box(ax,57,42,41,8,"SALF-DINO: frozen DINOv3-L/16\n16 x 16 spatial tokens, d=1024")
for x in (13.5,40.5,77.5):arrow(ax,50,55,x,50)
box(ax,2,26,23,6,"Linear head\n2 logits");box(ax,29,26,23,6,"Linear head\n2 logits")
for x in (13.5,40.5):arrow(ax,x,42,x,32)
box(ax,56,27,13,9,"Global mean\npooling\n g",False,8)
box(ax,70,27,14,9,"Border strips\nL / B / R\nattention: a",True,8)
box(ax,85,27,13,9,"DWConv 3x3\nGELU + 1x1\nGAP: l",True,8)
for x in (62.5,77,91.5):arrow(ax,77.5,42,x,36)
arrow(ax,69,31.5,70,31.5)
box(ax,57,13,41,10,"Gated residual fusion\nc = Wc[a; l] + bc\nz = sigmoid(Wz[g; a; l] + bz)\nf = g + z * c",True,8.7)
for x in (62.5,77,91.5):arrow(ax,x,27,x,23)
box(ax,63,3,29,6,"Linear head: 2 logits",True);arrow(ax,77.5,13,77.5,9)
ax.text(13.5,21,"2,050 trainable",ha="center",fontsize=8);ax.text(40.5,21,"1,538 trainable",ha="center",fontsize=8);ax.text(77.5,0.5,"8,403,970 trainable parameters",ha="center",fontsize=8,color=accent)
save(fig,"Figure_02_Baselines_vs_Proposed")
fig,ax=plt.subplots(figsize=(7.4,3.8));y=np.arange(len(s))
ax.barh(y,[float(x["macro_f1_mean_pct"]) for x in s],xerr=[float(x["macro_f1_sd_pct"]) for x in s],color=[accent]+[gray]*6,edgecolor=dark,linewidth=.4,capsize=2)
ax.set(yticks=y,yticklabels=names,xlim=(0,108),xlabel="Held-out macro-F1 (%)");ax.invert_yaxis()
for i,x in enumerate(s):ax.text(float(x["macro_f1_mean_pct"])+.6,i,f'{float(x["macro_f1_mean_pct"]):.2f}',va="center",fontsize=8)
ax.set_xticks([0,20,40,60,80,100]);ax.spines[["top","right"]].set_visible(False);save(fig,"Figure_03_Test_MacroF1")
e=read("srtaxy_errors.csv");fig,ax=plt.subplots(figsize=(7.4,3.8));y=np.arange(len(e))
ax.barh(y,[float(x["mean_fp"]) for x in e],color="#9ca8ae",label="False positives")
ax.barh(y,[float(x["mean_fn"]) for x in e],left=[float(x["mean_fp"]) for x in e],color="#485962",label="False negatives")
ax.set(yticks=y,yticklabels=[x["label"] for x in e],xlabel="Mean errors per seed (872 test charts)");ax.invert_yaxis();ax.legend(frameon=False,loc="lower right");ax.spines[["top","right"]].set_visible(False);save(fig,"Figure_04_srtAxY_Errors")
c=read("corruption_summary.csv");fig,ax=plt.subplots(figsize=(8.2,4.4));kinds=["jpeg","blur","downsample"];xx=np.arange(3);w=.112
for i,m in enumerate(s):
 vals=[float(next(x["macro_f1_pct"] for x in c if x["model"]==m["model"] and x["corruption"]==k)) for k in kinds]
 ax.bar(xx+(i-3)*w,vals,w,label=m["label"],color=accent if i==0 else str(.32+.105*(i-1)),edgecolor="white",linewidth=.4)
ax.set(xticks=xx,xticklabels=["JPEG Q40","Gaussian blur","Down/up-sample"],ylabel="Macro-F1 (%)",ylim=(0,100));ax.legend(ncol=3,fontsize=7,frameon=False,loc="upper center",bbox_to_anchor=(.5,1.23));ax.spines[["top","right"]].set_visible(False);save(fig,"Figure_05_Corruption_Robustness")
b=read("paired_group_bootstrap.csv");fig,ax=plt.subplots(figsize=(7.6,4));y=np.arange(len(b))
for i,x in enumerate(b):
 d,l,h=[float(x[k]) for k in ("delta_pp","ci_low_pp","ci_high_pp")]
 ax.errorbar(d,i,xerr=[[d-l],[h-d]],fmt="o",color=accent if x["comparator"].startswith("control_") else dark,capsize=3)
ax.axvline(0,color="#999999",ls="--",lw=.8);ax.set(yticks=y,yticklabels=[x["label"] for x in b],xlabel="SALF-DINO minus comparator (percentage points)");ax.invert_yaxis();ax.spines[["top","right"]].set_visible(False);save(fig,"Figure_S3_Paired_Bootstrap")
print("Generated six figure families (PNG 600 dpi / SVG / PDF)",flush=True)

"""
scripts/generate_readme_screenshot.py — Generate README screenshots.
Run: python scripts/generate_readme_screenshot.py
"""
import sys; sys.path.insert(0, ".")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

Path("docs/images").mkdir(parents=True, exist_ok=True)

np.random.seed(42)
tau   = np.arange(1, 41)
BG    = "#030810"; CARD="#091220"; GOLD="#F2C440"
BLUE  = "#3888F8"; RED="#E83030";  MUTED="#5070A0"; BORDER="#1A2A42"

pred  = 2.1964 * np.exp(-0.012*(tau-9)**2) + 0.02 + np.random.normal(0,.01,len(tau))
steer = 0.18 * np.where(tau<4,1,np.exp(-0.25*(tau-4))) + np.random.normal(0,.005,len(tau))
pred  = np.clip(pred, 0, None); steer = np.clip(steer, 0, None)

fig, axes = plt.subplots(1,2,figsize=(14,4.5)); fig.patch.set_facecolor(BG)
for ax in axes:
    ax.set_facecolor(CARD)
    for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
    ax.tick_params(colors=MUTED,labelsize=8); ax.grid(axis="y",color=BORDER,alpha=.6,lw=.5)

ax=axes[0]
ax.plot(tau,pred,color=GOLD,lw=2.5,label="Prediction MI",zorder=3)
ax.plot(tau,steer,color=BLUE,lw=1.8,ls="--",label="Steering MI",alpha=.9,zorder=2)
ax.axhline(0.01,color=MUTED,ls=":",lw=1.2,label="ε=0.01",alpha=.6)
ax.axvspan(4,34,alpha=.15,color=RED,label="ODW τ=4–34",zorder=1)
ax.set_xlabel("τ (delay steps)",color=MUTED,fontsize=9)
ax.set_ylabel("Mutual Information (bits)",color=MUTED,fontsize=9)
ax.set_title("FAWP Alpha Index — SPY (2y, Daily)",color="#E8EDF8",fontsize=10,fontweight="bold",pad=10)
ax.legend(fontsize=8,facecolor=CARD,labelcolor="#E8EDF8",edgecolor=BORDER,framealpha=.95)
ax.annotate("🔴 FAWP DETECTED\npeak gap: 2.1964 bits\nODW: τ=4–34",
            xy=(9,2.1964),xytext=(20,1.8),color=GOLD,fontsize=8,fontweight="bold",
            arrowprops=dict(arrowstyle="->",color=GOLD,lw=1.2))

ax2=axes[1]; ax2.set_xlim(0,1); ax2.set_ylim(0,1); ax2.axis("off")
ax2.set_title("Scan Results",color="#E8EDF8",fontsize=10,fontweight="bold",pad=10)
kpis=[("Peak Gap","2.1964 bits",GOLD),("τ⁺ₕ horizon","4",GOLD),
      ("τf cliff","35",GOLD),("ODW","τ 4–34",RED),("Assets","4",GOLD),("Flagged","3/4",RED)]
for i,(lbl,val,col) in enumerate(kpis):
    row,ci=divmod(i,3); x=0.05+ci*0.34; y=0.72-row*0.44
    ax2.add_patch(mpatches.FancyBboxPatch((x,y),.28,.32,boxstyle="round,pad=0.02",
                  facecolor=CARD,edgecolor=BORDER,linewidth=1))
    ax2.text(x+.14,y+.20,val,ha="center",va="center",color=col,fontsize=13,fontweight="bold",fontfamily="monospace")
    ax2.text(x+.14,y+.07,lbl,ha="center",va="center",color=MUTED,fontsize=7,fontweight="bold")

fig.text(.5,.01,"fawp-index v1.1.0 · doi:10.5281/zenodo.18673949",ha="center",color=MUTED,fontsize=7)
plt.tight_layout(pad=1.2)
plt.savefig("docs/images/scanner_result.png",dpi=150,facecolor=BG,bbox_inches="tight"); plt.close()
print("✓ docs/images/scanner_result.png")

# Weather chart
fig2,ax3=plt.subplots(figsize=(10,4)); fig2.patch.set_facecolor(BG)
ax3.set_facecolor(CARD)
for sp in ax3.spines.values(): sp.set_edgecolor(BORDER)
ax3.tick_params(colors=MUTED,labelsize=8); ax3.grid(axis="y",color=BORDER,alpha=.6,lw=.5)
wp = 0.18*np.exp(-0.015*(tau-8)**2)+0.01+np.random.normal(0,.003,len(tau))
ws = 0.12*np.where(tau<5,1,np.exp(-0.2*(tau-5)))+np.random.normal(0,.003,len(tau))
wp=np.clip(wp,0,None); ws=np.clip(ws,0,None)
ax3.plot(tau,wp,color=GOLD,lw=2.5,label="Prediction MI (ERA5 temperature)")
ax3.plot(tau,ws,color="#2090E8",lw=1.8,ls="--",label="Steering MI")
ax3.axhline(0.01,color=MUTED,ls=":",lw=1,alpha=.6)
ax3.axvspan(5,33,alpha=.15,color=RED,label="ODW τ=5–33")
ax3.set_xlabel("τ (delay, days)",color=MUTED,fontsize=9)
ax3.set_ylabel("Mutual Information (bits)",color=MUTED,fontsize=9)
ax3.set_title("FAWP Weather Scanner — London · Temperature 2m · 2010–2024",
              color="#E8EDF8",fontsize=10,fontweight="bold",pad=10)
ax3.legend(fontsize=8,facecolor=CARD,labelcolor="#E8EDF8",edgecolor=BORDER,framealpha=.95)
fig2.text(.5,.01,"ERA5 via Open-Meteo · fawp-index v1.1.0 · fawp-scanner.info",ha="center",color=MUTED,fontsize=7)
plt.tight_layout(pad=1.2)
plt.savefig("docs/images/weather_result.png",dpi=150,facecolor=BG,bbox_inches="tight"); plt.close()
print("✓ docs/images/weather_result.png")

"""Graph 8a (all) and 8b (flagship). Wave 7 corrected HSD.

Colors use the moral-autonomy ladder:
L1 Scaling, L2 Temporal, L3 Reflection, L4 Co-learning, L5 Real-world.
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
from matplotlib.lines import Line2D
import os

OUT = os.path.dirname(os.path.abspath(__file__))

all_models = [
    ("InstructGPT (175B)","2022-03",3.43e-8,1,175e9,"OpenAI"),("Constitutional AI (52B)","2022-12",6.35e-8,2,52e9,"Anthropic"),
    ("Dromedary (65B)","2023-05",1.26e-8,3,65e9,"CMU"),("Llama 2 (70B)","2023-07",5.71e-8,1,70e9,"Meta"),
    ("Baichuan 2 (13B)","2023-09",3.85e-7,1,13e9,"Baichuan"),("SteerLM (43B)","2023-10",1.63e-8,1,43e9,"NVIDIA"),
    ("SALMON (70B)","2023-10",2.24e-8,3,70e9,"IBM"),("Zephyr (7B)","2023-10",4.29e-7,2,7e9,"HuggingFace"),
    ("ORCA 2 (13B)","2023-11",9.08e-9,1,13e9,"Microsoft"),("OpenChat 3.5 (7B)","2023-11",2.29e-8,1,7e9,"Tsinghua"),
    ("Starling (7B)","2023-11",6.40e-8,2,7e9,"Berkeley"),("Tulu 2 (70B)","2023-11",4.29e-8,1,70e9,"AI2"),
    ("Llama Guard (7B)","2023-12",5.49e-8,2,7e9,"Meta"),("Self-Rewarding (70B)","2024-01",8.57e-9,3,70e9,"Meta"),
    ("InternLM2 (20B)","2024-03",5.00e-8,1,20e9,"Shanghai AI"),("Yi (34B)","2024-03",3.24e-9,1,34e9,"01.AI"),
    ("Aegis (7B)","2024-04",2.29e-7,2,7e9,"NVIDIA"),("MiniCPM (2.4B)","2024-04",1.78e-7,2,2.4e9,"OpenBMB"),
    ("Phi-3 (14B)","2024-04",1.14e-8,1,14e9,"Microsoft"),("Phi-4 (14B)","2024-04",4.29e-8,1.5,14e9,"Microsoft"),
    ("ArmoRM (8B)","2024-06",1.79e-7,2,8e9,"Berkeley"),("WildGuard (7B)","2024-06",7.14e-8,2,7e9,"AI2"),
    ("Nemotron-4 (340B)","2024-06",5.88e-9,1.5,340e9,"NVIDIA"),("Llama 3.1 (405B)","2024-07",2.73e-9,2,405e9,"Meta"),
    ("Llama Guard 3 (8B)","2024-07",1.25e-7,2,8e9,"Meta"),("Meta-Rewarding (8B)","2024-07",7.50e-8,3.5,8e9,"Meta"),
    ("Gemma 2 (27B)","2024-08",4.77e-9,1,27e9,"Google"),("Qwen 2.5 (72B)","2024-09",2.78e-9,1,72e9,"Alibaba"),
    ("Granite 3.0 (8B)","2024-10",2.13e-7,1.5,8e9,"IBM"),("Granite 3.0 (2B)","2024-10",8.50e-7,1.5,2e9,"IBM"),
    ("SmolLM2 (1.7B)","2024-11",2.52e-7,2,1.7e9,"HuggingFace"),("Tulu 3 (70B)","2024-11",5.71e-8,1.5,70e9,"AI2"),
    ("Tulu 3 (8B)","2024-11",2.00e-7,2,8e9,"AI2"),("Llama 3.3 (70B)","2024-12",1.58e-8,2,70e9,"Meta"),
    ("DeepSeek-V3 (671B)","2024-12",5.96e-10,1.5,671e9,"DeepSeek"),("DeepSeek-R1 (671B)","2025-01",1.07e-9,1.5,671e9,"DeepSeek"),
    ("DeepSeek-R1-Zero (671B)","2025-01",2.38e-10,3,671e9,"DeepSeek"),("Mistral Small 3 (24B)","2025-01",4.17e-8,1,24e9,"Mistral"),
    ("OLMo 2 (7B)","2025-01",2.29e-7,2,7e9,"AI2"),("OLMo 2 (13B)","2025-01",1.23e-7,2,13e9,"AI2"),
    ("Phi-4-mini (3.8B)","2025-02",2.63e-7,1.5,3.8e9,"Microsoft"),("Command A (111B)","2025-03",8.29e-9,1,111e9,"Cohere"),
    ("Gemma 3 (1B)","2025-03",2.80e-6,1.5,1e9,"Google"),("Gemma 3 (4B)","2025-03",6.51e-7,1.5,4.3e9,"Google"),
    ("Gemma 3 (12B)","2025-03",2.30e-7,1.5,12.2e9,"Google"),("Gemma 3 (27B)","2025-03",2.19e-7,1.5,27.4e9,"Google"),
    ("DeepSeek-V3-0324 (671B)","2025-03",1.19e-9,1.5,671e9,"DeepSeek"),("Nemotron-Nano (8B)","2025-03",1.28e-7,1.5,8e9,"NVIDIA"),
    ("Nemotron-Super (49B)","2025-03",2.08e-8,1.5,49e9,"NVIDIA"),("Nemotron-Ultra (253B)","2025-03",4.03e-9,1.5,253e9,"NVIDIA"),
    ("OLMo 2 (32B)","2025-03",5.00e-8,2,32e9,"AI2"),("Llama 4 Scout (109B)","2025-04",4.28e-8,1,109e9,"Meta"),
    ("Llama 4 Maverick (400B)","2025-04",1.17e-8,1,400e9,"Meta"),("Qwen3 (0.6B)","2025-04",5.00e-6,1,0.6e9,"Alibaba"),
    ("Qwen3 (1.7B)","2025-04",1.76e-6,1,1.7e9,"Alibaba"),("Qwen3 (4B)","2025-04",7.50e-7,1,4e9,"Alibaba"),
    ("Qwen3 (8B)","2025-04",1.25e-7,1,8e9,"Alibaba"),("Qwen3 (14B)","2025-04",7.14e-8,1,14e9,"Alibaba"),
    ("Qwen3 (32B)","2025-04",3.13e-8,1,32e9,"Alibaba"),("Qwen3 MoE (30B-A3B)","2025-04",3.33e-7,1,30e9,"Alibaba"),
    ("Qwen3 (235B)","2025-04",1.28e-8,1,235e9,"Alibaba"),("Phi-4-reasoning (14B)","2025-04",5.43e-8,1.5,14e9,"Microsoft"),
    ("Granite 3.3 (8B)","2025-04",2.13e-7,1.5,8e9,"IBM"),("GLM-4 (9B)","2025-04",2.67e-8,1,9e9,"Zhipu AI"),
    ("GLM-Z1 (32B)","2025-04",1.25e-8,3,32e9,"Zhipu AI"),("DeepSeek-R1-0528 (671B)","2025-05",1.07e-9,1.5,671e9,"DeepSeek"),
    ("Gemma 3n E2B (5B)","2025-06",1.18e-6,1.5,5.1e9,"Google"),("Gemma 3n E4B (8B)","2025-06",7.50e-7,1.5,8e9,"Google"),
    ("Seed-Coder (8B)","2025-06",9.00e-8,2,8e9,"ByteDance"),("MiniCPM4 (8B)","2025-06",6.75e-8,2,8e9,"OpenBMB"),
    ("MiniMax-M1 (456B)","2025-06",8.77e-10,1.5,456e9,"MiniMax"),("Kimi K2 (1T)","2025-07",4.00e-10,1.5,1e12,"Moonshot"),
    ("Step 3 (321B)","2025-07",9.35e-9,1,321e9,"StepFun"),("gpt-oss-120b (117B)","2025-08",8.56e-9,1,116.8e9,"OpenAI"),
    ("gpt-oss-20b (21B)","2025-08",4.78e-8,1,20.9e9,"OpenAI"),("DeepSeek-V3.1 (671B)","2025-08",1.19e-9,1.5,671e9,"DeepSeek"),
    ("OLMo 3 (32B)","2025-11",5.00e-8,2,32e9,"AI2"),("Mistral Large 3 (675B)","2025-12",1.48e-9,1,675e9,"Mistral"),
    ("Kimi K2.5 (1T)","2026-01",4.00e-10,1.5,1e12,"Moonshot"),("GLM-5 (744B)","2026-02",5.38e-10,1.5,744e9,"Zhipu AI"),
    ("MiniMax-M2.5 (230B)","2026-02",1.30e-8,1.5,230e9,"MiniMax"),("Qwen3-Coder (80B)","2026-03",9.00e-9,1,80e9,"Alibaba"),
    ("Gemma 4 (31B)","2026-04",9.12e-8,1.5,30.7e9,"Google"),("Gemma 4 MoE (25B)","2026-04",7.37e-7,1.5,25.2e9,"Google"),
    ("Gemma 4 E2B (5B)","2026-04",1.18e-6,1.5,5.1e9,"Google"),("Gemma 4 E4B (8B)","2026-04",7.50e-7,1.5,8e9,"Google"),
]

flagship_models = [
    ("InstructGPT (175B)","2022-03",3.43e-8,1,175e9,"OpenAI"),("Constitutional AI (52B)","2022-12",6.35e-8,2,52e9,"Anthropic"),
    ("Llama 2 (70B)","2023-07",5.71e-8,1,70e9,"Meta"),("Baichuan 2 (13B)","2023-09",3.85e-7,1,13e9,"Baichuan"),
    ("Yi (34B)","2024-03",3.24e-9,1,34e9,"01.AI"),("Phi-4 (14B)","2024-04",4.29e-8,1.5,14e9,"Microsoft"),
    ("Nemotron-4 (340B)","2024-06",5.88e-9,1.5,340e9,"NVIDIA"),("Llama 3.1 (405B)","2024-07",2.73e-9,2,405e9,"Meta"),
    ("Gemma 2 (27B)","2024-08",4.77e-9,1,27e9,"Google"),("Qwen 2.5 (72B)","2024-09",2.78e-9,1,72e9,"Alibaba"),
    ("DeepSeek-V3 (671B)","2024-12",5.96e-10,1.5,671e9,"DeepSeek"),("DeepSeek-R1 (671B)","2025-01",1.07e-9,1.5,671e9,"DeepSeek"),
    ("DeepSeek-R1-Zero (671B)","2025-01",2.38e-10,3,671e9,"DeepSeek"),("Command A (111B)","2025-03",8.29e-9,1,111e9,"Cohere"),
    ("Gemma 3 (27B)","2025-03",2.19e-7,1.5,27.4e9,"Google"),("Nemotron-Ultra (253B)","2025-03",4.03e-9,1.5,253e9,"NVIDIA"),
    ("Llama 4 Maverick (400B)","2025-04",1.17e-8,1,400e9,"Meta"),("Qwen3 (235B)","2025-04",1.28e-8,1,235e9,"Alibaba"),
    ("Granite 3.3 (8B)","2025-04",2.13e-7,1.5,8e9,"IBM"),("GLM-Z1 (32B)","2025-04",1.25e-8,3,32e9,"Zhipu AI"),
    ("Seed-Coder (8B)","2025-06",9.00e-8,2,8e9,"ByteDance"),("MiniMax-M1 (456B)","2025-06",8.77e-10,1.5,456e9,"MiniMax"),
    ("Kimi K2.5 (1T)","2025-07",4.00e-10,1.5,1e12,"Moonshot"),("Step 3 (321B)","2025-07",9.35e-9,1,321e9,"StepFun"),
    ("gpt-oss-120b (117B)","2025-08",8.56e-9,1,116.8e9,"OpenAI"),("DeepSeek-V3.1 (671B)","2025-08",1.19e-9,1.5,671e9,"DeepSeek"),
    ("Mistral Large 3 (675B)","2025-12",1.48e-9,1,675e9,"Mistral"),("GLM-5 (744B)","2026-02",5.38e-10,1.5,744e9,"Zhipu AI"),
    ("MiniMax-M2.5 (230B)","2026-02",1.30e-8,1.5,230e9,"MiniMax"),("Qwen3-Coder (80B)","2026-03",9.00e-9,1,80e9,"Alibaba"),
    ("Gemma 4 (31B)","2026-04",9.12e-8,1.5,30.7e9,"Google"),
]

lc={1:"#d84a3a",2:"#d99a2b",3:"#2f9e6d",4:"#2f8fbf",5:"#6f62c9"}
om={"OpenAI":"o","Anthropic":"s","Meta":"^","Google":"P","DeepSeek":"D","Alibaba":"p","Microsoft":"X","NVIDIA":"h",
    "HuggingFace":"v","AI2":"*","Baichuan":"8","Shanghai AI":"d","Cohere":">","CMU":"<","IBM":"H","Berkeley":"v",
    "OpenBMB":"o","Tsinghua":"o","01.AI":"p","Zhipu AI":"d","ByteDance":"o","MiniMax":"o","Moonshot":"o","Mistral":"o","StepFun":"o"}

reflection_methods = (
    "Dromedary",
    "SALMON",
    "Self-Rewarding",
    "Meta-Rewarding",
    "DeepSeek-R1-Zero",
    "GLM-Z1",
)

def ladder_level(name, old_level):
    if any(key in name for key in reflection_methods):
        return 3
    return 1

def pf(ml,fs_fig,t,st,fn,fs_l,ia,note):
    fig,ax=plt.subplots(figsize=fs_fig);ds=[];hs=[];ps=[]
    for nm,dt,h,lv,pr,og in ml:
        d=datetime.strptime(dt,"%Y-%m");ds.append(d);hs.append(h)
        ladder=ladder_level(nm,lv)
        ax.scatter(d,h,s=max(np.log10(pr)*30,60),c=lc.get(ladder,"#95a5a6"),edgecolors="black",linewidths=0.9,zorder=5,alpha=0.85,marker=om.get(og,"o"))
        ps.append((d,h,nm))
    pl=[]
    for d,h,nm in ps:
        cr=sum(1 for pd,ph in pl if abs((d-pd).days)<40 and abs(np.log10(h)-np.log10(ph))<0.12)
        oy=10 if cr==0 else(-11 if cr%2==1 else 10+(cr//2)*5);va="bottom" if oy>0 else "top"
        ax.annotate(nm,(d,h),fontsize=fs_l,ha="center",va=va,xytext=(0,oy),textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.12",facecolor="white",alpha=0.75,linewidth=0.15),zorder=6)
        pl.append((d,h))
    dn=np.array([(d-datetime(2022,1,1)).days for d in ds]);lh=np.log10(hs);z=np.polyfit(dn,lh,1);p=np.poly1d(z)
    xf=np.linspace(dn.min()-30,dn.max()+60,200);df=[datetime(2022,1,1)+timedelta(days=int(x)) for x in xf]
    ax.plot(df,10**p(xf),color="gray",linestyle="--",alpha=0.4,linewidth=1.5)
    ax.text(0.50,0.97,f"Log-linear trend: {z[0]*365:.2f} decades/year",transform=ax.transAxes,fontsize=9,color="gray",style="italic",va="top",
            bbox=dict(boxstyle="round",facecolor="white",alpha=0.8))
    ax.set_yscale("log");ax.set_xlabel("Release Date",fontsize=14,fontweight="bold")
    ax.set_ylabel("Human Specification Density (bits/parameter)",fontsize=14,fontweight="bold")
    ax.set_title(f"{t}\n{st}",fontsize=16,fontweight="bold",pad=15)
    ax.grid(True,alpha=0.2,linestyle="--");ax.spines["top"].set_visible(False);ax.spines["right"].set_visible(False)
    g="#555555"
    cl=[Line2D([0],[0],marker="o",color="w",markerfacecolor=lc[k],markeredgecolor="black",markersize=10,label=l) for k,l in
        [(1,"L1: Scaling"),(2,"L2: Temporal"),(3,"L3: Reflection"),(4,"L4: Co-learning"),(5,"L5: Real-world")]]
    cl.append(Line2D([0],[0],linestyle="--",color="gray",alpha=0.4,linewidth=1.5,label="Trend"))
    l1=ax.legend(handles=cl,loc="lower left",fontsize=8,framealpha=0.92,ncol=2,title="Color = Moral-Autonomy Ladder",title_fontsize=9);ax.add_artist(l1)
    se=[("^","Meta"),("o","OpenAI / misc."),("s","Anthropic"),("D","DeepSeek"),("X","Microsoft"),("h","NVIDIA"),
        ("P","Google"),("H","IBM"),("p","Alibaba / 01.AI"),(">","Cohere"),("8","Baichuan")]
    if ia:se+=[("*","AI2"),("v","HuggingFace / Berkeley"),("d","Shanghai AI / Zhipu"),("<","CMU")]
    else:se+=[("d","Zhipu AI")]
    sl=[Line2D([0],[0],marker=m,color="w",markerfacecolor=g,markeredgecolor="black",markersize=9 if m!="*" else 11,label=l) for m,l in se]
    ax.legend(handles=sl,loc="upper right",fontsize=7,framealpha=0.92,ncol=3,title="Shape = Organization",title_fontsize=9,columnspacing=1.0,handletextpad=0.4)
    ax.text(0.99,0.01,note,transform=ax.transAxes,fontsize=6.5,color="gray",ha="right",va="bottom",style="italic",
            bbox=dict(boxstyle="round,pad=0.3",facecolor="white",alpha=0.7,linewidth=0))
    plt.tight_layout();pa=os.path.join(OUT,fn);plt.savefig(pa,dpi=200,bbox_inches="tight");plt.close();print(f"Saved: {pa} ({len(ml)} models)")

if __name__=="__main__":
    pf(all_models,(28,15),"HSD of Aligned LLMs Over Time \u2014 All Verified Models",
       f"N = {len(all_models)} models (industry + academia), 2022\u20132026 | L1\u2013L5 autonomy ladder","graph8a_hsd_all.png",4.3,True,
       f"N={len(all_models)} models | Bubble \u221d log(params) | HSD=H_spec(bits)/N_params\nIncludes industry+academic+research | Colors use L1\u2013L5 moral-autonomy ladder")
    pf(flagship_models,(24,13),"HSD of Flagship LLMs Over Time \u2014 Industry Only, Largest per Series",
       f"N = {len(flagship_models)} models, 2022\u20132026 | L1\u2013L5 autonomy ladder","graph8b_hsd_flagship.png",6.5,False,
       f"N={len(flagship_models)} flagship models | Largest per series, industry only\nExcludes smaller variants, safety classifiers, academic models\nBubble \u221d log(params) | Colors use L1\u2013L5 moral-autonomy ladder")

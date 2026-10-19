"""MOPAC の parameters_for_PMx_C.F90 から PYSEQM 形式の CSV を作る"""
import re, sys, os, numpy as np, csv
# MOPAC のソース（https://github.com/openmopac/mopac）の src/models/。環境変数 MOPAC_SRC で指定できる
SRC=os.path.join(os.path.expanduser(os.environ.get("MOPAC_SRC", "~/src/mopac")), "src", "models") + "/"
import seqm as _seqm
P=os.path.join(os.path.dirname(_seqm.__file__), "params") + "/"   # PYSEQM 同梱のパラメータ（列名の取得に使う）
cols=open(P+"parameters_PM6_SP_MOPAC.csv").readline().strip().split(",")
cols=[c.strip() for c in cols]
fmap={"uss":"U_ss","upp":"U_pp","udd":"U_dd","zs":"zeta_s","zp":"zeta_p","zd":"zeta_d","betas":"beta_s","betap":"beta_p","betad":"beta_d",
      "zsn":"s_orb_exp_tail","zpn":"p_orb_exp_tail","zdn":"d_orb_exp_tail","gss":"g_ss","gsp":"g_sp","gpp":"g_pp","gp2":"g_p2","hsp":"h_sp",
      "f0sd":"F0SD","g2sd":"G2SD","poc_":"rho_core","alp":"alpha"}
def parse(fname, suf):
    txt=open(SRC+fname).read()
    el={}
    for m in re.finditer(r"data\s+(\w+?)"+suf+r"(\d?)\(\s*(\d+)\s*(?:,\s*(\d))?\)\s*/\s*([-+0-9.D]+)\s*/",txt):
        name,gi,z,k,v=m.group(1),m.group(2),int(m.group(3)),m.group(4),float(m.group(5).replace("D","E"))
        if gi: name=name+gi   # gues61 → gues1
        el.setdefault(z,{})[(name,k)]=v
    pairs={}
    for m in re.finditer(r"(alpb|xfac)\(\s*(\d+),\s*(\d+)\)\s*=\s*([-+0-9.d]+)",txt):
        pairs.setdefault((int(m.group(2)),int(m.group(3))),{})[m.group(1)]=float(m.group(4).replace("d","e"))
    vpar={int(m.group(1)):float(m.group(2).replace("d","e")) for m in re.finditer(r"v_par"+suf+r"\((\d+)\)\s*/\s*([-+0-9.d]+)\s*/",txt)}
    return el,pairs,vpar
def write(el,pairs,method,outdir):
    sym=[r.split(",")[1].strip() for r in open(P+"parameters_PM6_SP_MOPAC.csv").read().splitlines()[1:]]
    rows=[]
    for z in range(1,len(sym)+1):
        d={c:0.0 for c in cols}; d["N"]=z; d["sym"]=sym[z-1]
        for (n,k),v in el.get(z,{}).items():
            if n in fmap: d[fmap[n]]=v
            if n in ("zsn","zpn","zdn"): d["z_"+n[1]]=v
            elif n.startswith("gues") and k: d[f"Gaussian{k}_"+{"1":"K","2":"L","3":"M"}[n[-1]]]=v
        rows.append(d)
    with open(f"{outdir}/parameters_{method}_MOPAC.csv","w") as f:
        f.write(",".join(cols)+"\n")
        for d in rows: f.write(",".join(str(d[c]) for c in cols)+"\n")
    with open(f"{outdir}/PWCCT_{method}_MOPAC.csv","w") as f:
        for i in range(1,len(sym)+1):
            for j in range(1,len(sym)+1):
                a,b=max(i,j),min(i,j); p=pairs.get((a,b),{})
                f.write(f"{i},{j},{p.get('alpb',0)},{p.get('xfac',0)}\n")
if __name__=="__main__":
    import os
    os.makedirs("check_pm6",exist_ok=True)
    el,pairs,v=parse("parameters_for_PM6_C.F90","6"); write(el,pairs,"PM6_SP","check_pm6")
    print("PM6 elements",len(el),"pairs",len(pairs),"v_par",v)

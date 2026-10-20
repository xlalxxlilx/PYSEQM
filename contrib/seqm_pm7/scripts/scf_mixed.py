import sys,os,time,json,itertools,numpy as np,ase.io,torch
W=os.path.join(os.path.dirname(os.path.abspath(__file__)),".."); sys.path.insert(0,W)
os.environ.setdefault("SEQM_PM7_OZ2_LIB", W+"/ozaki/liboz2shim.so")
import seqm_pm7.calculator as C, seqm_pm7.eig_mixed as EM, seqm_pm7.scf_fast as SF
print("oz2 available", EM.available())
def load(nm): a=ase.io.read(os.path.join(os.environ.get("SEQM_PM7_MOLS",W+"/validation/mols2"),nm,"xtbopt.xyz")); return a.get_chemical_symbols(), a.positions-a.positions.mean(0)
def grid(s,x,n,d): offs=[np.array(v)*d for v in itertools.product(range(n),repeat=3)]; return s*len(offs), np.vstack([x+o for o in offs])
R=json.load(open(W+"/validation/mopac_ref.json")); w=R["cl:HF_w20_start_neutral"]
s1,x1=load("TDMAZ"); s2,x2=load("Zr4O6OH4")
sets={"27x HF(H2O)20 (1674)":(grid(w["sym"],np.array(w["x"]),3,12.0),10.0),
      "27x TDMAZ (999)":(grid(s1,x1,3,11.0),10.0),
      "27x Zr4O6(OH)4 (486)":(grid(s2,x2,3,9.0),10.0)}
out=[]
for nm,((s,x),rc) in sets.items():
    res={}
    for eig in ("fp64","auto"):
        for k in EM.STATS: EM.STATS[k]=0
        SF.STATS["fock"]=0
        torch.cuda.synchronize(); t=time.time()
        q=C.compute(s,x,"PM7",device="cuda",cutoff=rc,eig=eig); torch.cuda.synchronize(); dt=time.time()-t
        res[eig]=(q,dt,dict(EM.STATS),SF.STATS["fock"])
        print(f"{nm:22s} eig={eig:4s} {dt:6.1f}s fock {SF.STATS['fock']} conv {q['scf_converged']} Hf {q['Hf']:.6f} stats {EM.STATS}",flush=True)
    (q0,t0,_,f0),(q1,t1,st,f1)=res["fp64"],res["auto"]
    dH=float(q1["Hf"]-q0["Hf"]); dF=float(np.abs(q1["forces"]-q0["forces"]).max())
    print(f"   dHf {dH:+.2e} kcal/mol  max dF {dF:.2e}  speedup x{t0/t1:.2f}",flush=True)
    out.append(dict(system=nm,cutoff=rc,t_fp64=t0,t_mixed=t1,fock_fp64=f0,fock_mixed=f1,dHf=dH,dF=dF,stats=st))
json.dump(out,open("scf_mixed.json","w"),indent=1)

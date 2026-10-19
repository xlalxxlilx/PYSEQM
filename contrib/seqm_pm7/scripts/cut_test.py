import sys,os,time,json,itertools,numpy as np,ase.io,torch; sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),".."))
import seqm_pm7.calculator as C
dev=sys.argv[1]
def load(nm): a=ase.io.read(os.path.join(os.environ.get("SEQM_PM7_MOLS",os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","validation","mols2")),nm,"xtbopt.xyz")); return a.get_chemical_symbols(), a.positions-a.positions.mean(0)
def grid(s,x,n,d): offs=[np.array(v)*d for v in itertools.product(range(n),repeat=3)]; return s*len(offs), np.vstack([x+o for o in offs])
R=json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","validation","mopac_ref.json"))); w=R["cl:HF_w20_start_neutral"]
s1,x1=load("TDMAZ"); s2,x2=load("Zr4O6OH4")
sets={"8x TDMAZ (296)":grid(s1,x1,2,11.0),"8x Zr4O6(OH)4 (144)":grid(s2,x2,2,9.0),"8x HF(H2O)20 (496)":grid(w["sym"],np.array(w["x"]),2,12.0)}
rows=[]
for nm,(s,x) in sets.items():
    if dev=="cuda": C.compute(s,x,"PM7",device=dev); torch.cuda.synchronize()
    t=time.time(); q0=C.compute(s,x,"PM7",device=dev); 
    if dev=="cuda": torch.cuda.synchronize()
    t0=time.time()-t
    print(f"{nm:22s} full    {t0:6.2f}s Hf {q0['Hf']:.4f}",flush=True)
    for rc in (8.0,10.0,12.0):
        t=time.time(); q=C.compute(s,x,"PM7",device=dev,cutoff=rc)
        if dev=="cuda": torch.cuda.synchronize()
        dt=time.time()-t; dH=q["Hf"]-q0["Hf"]; dF=np.abs(q["forces"]-q0["forces"]).max()
        rows.append(dict(system=nm,cutoff=rc,t_full=t0,t_cut=dt,dHf=dH,dF=dF))
        print(f"{nm:22s} Rc {rc:4.1f} {dt:6.2f}s dHf {dH:+.2e} dF {dF:.2e} conv {q['scf_converged']}",flush=True)
json.dump(rows,open(f"cut_test_{dev}.json","w"),indent=1)

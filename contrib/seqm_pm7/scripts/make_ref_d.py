import sys,os,re,json,glob,subprocess,tempfile,numpy as np,ase.io
from scipy.spatial.transform import Rotation as R
MOP=os.path.expanduser(os.environ.get("MOPAC_HOME", "~/opt/mopac-23.2.5-linux") + "/bin/mopac"); LIB=os.path.expanduser(os.environ.get("MOPAC_HOME", "~/opt/mopac-23.2.5-linux") + "/lib")
def mopac(s,x,kw):
    d=tempfile.mkdtemp(prefix="mopr_")
    open(f"{d}/m.mop","w").write(f"{kw} 1SCF GRAD PRECISE AUX(PRECISION=9)\n\n\n"+"".join(f"{a} {c[0]:.10f} 1 {c[1]:.10f} 1 {c[2]:.10f} 1\n" for a,c in zip(s,x)))
    subprocess.run([MOP,"m.mop"],cwd=d,capture_output=True,env={**os.environ,"LD_LIBRARY_PATH":LIB,"OMP_NUM_THREADS":"1"})
    aux=open(f"{d}/m.aux").read()
    H=float(re.search(r"HEAT_OF_FORMATION:KCAL/MOL=([-+0-9.EeD]+)",aux).group(1).replace("D","E"))
    G=np.array(re.search(r"GRADIENTS:KCAL/MOL/ANGSTROM\[\d+\]=\s*(.*?)\n\s*[A-Z]",aux,re.S).group(1).split(),float).reshape(-1,3)
    return H,G
ref={}
for d in ("mols","mols2"):
    for p in sorted(glob.glob(f"{d}/*/xtbopt.xyz")):
        nm=p.split("/")[1]; a=ase.io.read(p); s=a.get_chemical_symbols()
        x=R.from_euler("xyz",[17,29,41],degrees=True).apply(a.positions-a.positions.mean(0))
        e={"sym":s,"x":x.round(10).tolist(),"charge":0}
        for m in ("PM6","PM7-","PM7"):
            H,G=mopac(s,x,m); e[m]={"Hf":H,"grad":G.tolist()}
        ref[nm]=e; print(nm,len(s),{m:round(e[m]["Hf"],4) for m in ("PM6","PM7-","PM7")},flush=True)
json.dump(ref,open("../seqm_pm7/validation/mopac_ref_d.json","w"))
print("n",len(ref))

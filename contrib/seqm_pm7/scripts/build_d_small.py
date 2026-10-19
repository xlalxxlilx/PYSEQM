import numpy as np, os, subprocess, re
XTB=os.environ.get("XTB", "xtb")   # xtb の実行ファイル
def tet(c, r):
    v=np.array([[1,1,1],[-1,-1,1],[-1,1,-1],[1,-1,-1]])/np.sqrt(3); return c+r*v
M={}
M["SiH4"]=(["Si"]+["H"]*4, np.vstack([[0,0,0],tet(np.zeros(3),1.48)]))
M["AlH3"]=(["Al","H","H","H"], np.array([[0,0,0],[1.58,0,0],[-0.79,1.37,0],[-0.79,-1.37,0]]))
def M_OH4(m, r):
    O=tet(np.zeros(3), r); H=[o+0.96*np.cross(o,[0.3,0.5,0.81])/np.linalg.norm(np.cross(o,[0.3,0.5,0.81]))*0.6+0.96*0.8*o/np.linalg.norm(o) for o in O]
    return [m]+["O"]*4+["H"]*4, np.vstack([[0,0,0],O,H])
M["Si(OH)4"]=M_OH4("Si",1.63); M["Zr(OH)4"]=M_OH4("Zr",1.95); M["Hf(OH)4"]=M_OH4("Hf",1.93)
O=np.array([[1.70,0,0],[-0.85,1.47,0],[-0.85,-1.47,0]]); H=O*1.56
M["Al(OH)3"]=(["Al","O","O","O","H","H","H"], np.vstack([[0,0,0],O,O+np.array([[0.3,0.9,0],[0.6,-0.6,0],[-0.6,-0.6,0.3]])]))
M["H3SiOSiH3"]=(["Si","O","Si","H","H","H","H","H","H"], np.array([[-1.6,0,0],[0,0.3,0],[1.6,0,0],[-2.0,1.4,0],[-2.0,-0.7,1.2],[-2.0,-0.7,-1.2],[2.0,1.4,0],[2.0,-0.7,1.2],[2.0,-0.7,-1.2]]))
M["AlMe3"]=(["Al","C","C","C"]+["H"]*9, None)
M["ZrMe4"]=(["Zr"]+["C"]*4+["H"]*12, None)
M["HfMe4"]=(["Hf"]+["C"]*4+["H"]*12, None)
def methyls(m, centers):
    s=[m]; X=[np.zeros(3)]
    for c in centers:
        s.append("C"); X.append(c)
    for c in centers:
        u=c/np.linalg.norm(c); a=np.cross(u,[0.37,0.21,0.9]); a/=np.linalg.norm(a); b=np.cross(u,a)
        for k in range(3):
            ang=2*np.pi*k/3; s.append("H"); X.append(c+0.36*u+1.03*(np.cos(ang)*a+np.sin(ang)*b))
    return s, np.array(X)
M["AlMe3"]=methyls("Al", np.array([[1.96,0,0],[-0.98,1.70,0],[-0.98,-1.70,0]]))
M["ZrMe4"]=methyls("Zr", tet(np.zeros(3),2.25)); M["HfMe4"]=methyls("Hf", tet(np.zeros(3),2.22))
# Zr2O2(OH)4 型の小さな二核
M["Zr2O2(OH)4"]=(["Zr","Zr","O","O","O","O","O","O","H","H","H","H"], np.array([[-1.5,0,0],[1.5,0,0],[0,1.3,0],[0,-1.3,0],[-2.8,0,1.3],[-2.8,0,-1.3],[2.8,0,1.3],[2.8,0,-1.3],[-3.6,0.5,1.5],[-3.6,0.5,-1.5],[3.6,0.5,1.5],[3.6,0.5,-1.5]]))
M["Al2O3H2"]=(["Al","Al","O","O","O","H","H"], np.array([[-1.3,0,0],[1.3,0,0],[0,1.2,0],[0,-1.2,0],[-3.0,0,0],[3.0,0,0],[-3.9,0.3,0]]))
from ase import Atoms
import ase.io
for nm,(s,x) in M.items():
    d=f"mols/{nm.replace('(','').replace(')','')}"; os.makedirs(d,exist_ok=True)
    ase.io.write(f"{d}/in.xyz", Atoms(s,positions=x))
    chg=0
    r=subprocess.run([XTB,"in.xyz","--gfn","2","--opt"],cwd=d,capture_output=True,text=True,env={**os.environ,"OMP_NUM_THREADS":"4"})
    ok=os.path.exists(f"{d}/xtbopt.xyz"); print(nm, "ok" if ok else "FAIL", re.findall(r"TOTAL ENERGY\s+([-0-9.]+)",r.stdout)[-1:] )

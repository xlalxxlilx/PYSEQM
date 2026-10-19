"""MOPAC の Fortran ソースから補正項の表を抜き出して data_tables.json に保存する"""
import re, json, os
S=os.path.expanduser("~/src/mopac/src/")
def block(txt, start_pat, n):
    i=txt.index(start_pat); seg=txt[i:]
    seg=re.sub(r"!.*","",seg)          # コメント除去
    nums=re.findall(r"[-+]?\d+\.\d*[dD]?[-+]?\d*",seg)
    vals=[float(x.replace("d","e").replace("D","e").rstrip("e")) for x in nums[:n]]
    assert len(vals)==n, (start_pat,len(vals)); return vals
t=open(S+"chemistry/atomradii_C.F90").read()
cov=block(t,"data atom_radius_covalent",107)
h=open(S+"corrections/Hydrogen_bond_corrections.F90").read()
covrad=block(h,"data covrad",94)
d=open(S+"corrections/H_bond_correction_PM6_DH_Dispersion.F90").read()
C=block(d,"data C/",86); R=block(d,"data R/",86); N=block(d,"data N/",86)
out={"atom_radius_covalent":cov,"hb_covrad":covrad,"disp_C":C,"disp_R":R,"disp_N":N}
json.dump(out,open(os.path.join(os.path.dirname(__file__),"seqm_pm7","data_tables.json"),"w"))
print({k:(len(v),v[:9]) for k,v in out.items()})

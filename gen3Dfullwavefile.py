"""
Generate fullwave .dat files for ERMES in 3D from ne.dat and topfile.json
"""
import os, json, re
import pandas as pd
import numpy as np
from scipy.interpolate import UnivariateSpline, RectBivariateSpline
from math import *

cwd = os.getcwd()

ne_file = pd.read_csv(cwd + "/source_data/ne_189998_3000ms_quinn.dat", sep=' ', header=None, skiprows=1)
with open(cwd + "/source_data/topfile_189998_3000ms_quinn.json", 'r') as file: topfile_data = json.load(file)
msh_path = cwd + "/7_degree_3d-1.dat"

ne_data = ne_file.to_numpy(dtype=float).T  
ne_spline = UnivariateSpline(ne_data[0]**2, ne_data[1], s=0, ext=1)

# Load cross-section fields (R,Z grids and 2D arrays)
A = {k: np.array(v) for k, v in topfile_data.items()}

Rg = np.asarray(A['R'])
Zg = np.asarray(A['Z'])
PsiRZ = np.asarray(A['pol_flux'])
BrRZ = np.asarray(A['Br'])
BtRZ = np.asarray(A['Bt'])
BzRZ = np.asarray(A['Bz'])

# Build splines on the cross-section
pol_flux_spline = RectBivariateSpline(Rg, Zg, PsiRZ.T, kx=3, ky=3, s=0)
Br_spline = RectBivariateSpline(Rg, Zg, BrRZ.T,  kx=3, ky=3, s=0)
Bt_spline = RectBivariateSpline(Rg, Zg, BtRZ.T,  kx=3, ky=3, s=0)
Bz_spline = RectBivariateSpline(Rg, Zg, BzRZ.T,  kx=3, ky=3, s=0)

# Parse GiD mesh nodes:  No[ID] = p(x,y,z);
node_ids, xs, ys, zs = [], [], [], []
pat = re.compile(r"No\[(\d+)\]\s*=\s*p\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^)]+)\s*\)\s*;")

with open(msh_path, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        m = pat.search(line)
        if m:
            node_ids.append(int(m.group(1)))
            xs.append(float(m.group(2)))
            ys.append(float(m.group(3)))
            zs.append(float(m.group(4)))

node_ids = np.asarray(node_ids, dtype=np.int64)
x = np.asarray(xs, float)
y = np.asarray(ys, float)
z = np.asarray(zs, float)

if node_ids.size == 0:
    raise RuntimeError("No nodes parsed. Check the .dat/.msh format.")

# Map (x,y,z) -> (R, phi, Z) (cylindrical)
# x = R cosphi, y = Z, z = -R sinphi
Rnod = np.hypot(x, z)
Znod = y
phi = np.arctan2(-z, x) # minus to satisfy z = -R sinφ

# Evaluate Psi, Br, Bt, Bz at node (R,Z)
Psi = pol_flux_spline.ev(Rnod, Znod)
Br = Br_spline.ev(Rnod, Znod)
Bt = Bt_spline.ev(Rnod, Znod)
Bz = Bz_spline.ev(Rnod, Znod)
ne = ne_spline(Psi)

# Convert (Br,Bt,Bz) -> (Bx,By,Bz_cart) at local phi
# e_R=(cosphi,0,-sinphi), e_t=(-sinphi,0,-cosphi), e_Z=(0,1,0)
c = np.cos(phi)
s = np.sin(phi)
Bx = Br*c - Bt*s
By = Bz
Bz_cart = -Br*s - Bt*c

# Write outputs (sorted by NodeID)
order = np.argsort(node_ids)
nid_sorted = node_ids[order]

np.savetxt(
    "ne.dat",
    np.column_stack([nid_sorted, ne[order]*1e19]), # Scale here AFTER splining to minimize issues with the scale being too large
    fmt=["%d", "%.8e"]
)

np.savetxt(
    "mag.dat",
    np.column_stack([nid_sorted, Bx[order], By[order], Bz_cart[order]]),
    fmt=["%d", "%.8e", "%.8e", "%.8e"]
)

# (Optional) quick sanity prints
print("psi(grid) range:", float(np.nanmin(PsiRZ)), float(np.nanmax(PsiRZ)))
print("psi(nodes) range:", float(np.nanmin(Psi)), float(np.nanmax(Psi)))
print(f"Wrote {nid_sorted.size} nodes to ne.dat and mag.dat")
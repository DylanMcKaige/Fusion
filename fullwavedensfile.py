"""
Created on Sun Jan 14 16:27:25 2024

@author: matth

Updated on 19/5/2025
Updated by Dylan James Mc Kaige
"""

from scipy import interpolate
from math import *
import sys
import os
import pandas as pd

# GiD problem path
Folder = os.getcwd()

if( sys.platform.startswith( 'win' ) ): 
    Folder = Folder + '\\'
else: 
    Folder = Folder + '/'

print(Folder)

projname = 'test'
f = pd.read_csv("RZ_ERMES.csv", header=None)
n = pd.read_csv("ne_ERMES.csv", header=None)
g = open(projname + ".gid/" + projname + "-1.dat") # the first dat file is your mesh file, i.e projectname.gid/projectname-1.dat
next(g) # Skip the header

Rcoords = f[f.columns[0]]
Zcoords = f[f.columns[1]]

ninterp = interpolate.RectBivariateSpline(
    Rcoords,
    Zcoords,
    n,
    kx=3,
    ky=3,
    s=0
)

nodeid = []
count = 0
magfile = []
for lines in g:
    list0, list1 = lines.split("[")[0],lines.split("[")[1]
    nodalID, list2 = list1.split("]")[0],list1.split("]")[1]
    list3, list4 = list2.split("(")[0],list2.split("(")[1]
    x, y, list7 = float(list4.split(",")[0]),float(list4.split(",")[1]),list4.split(",")[2]

    nodeid.append(nodalID)
    ndat = ninterp(x,y)[0][0]
    magfile.append([ndat])  
    
with open("transposed_" + projname + "_ne.dat", "w") as test_file:
    for ii in range(len(nodeid)):
        if ii < 9:
            test_file.write(str(int(nodeid[ii])) + "   " + '{:0.15e}'.format((float(magfile[ii][0])))+"\n")
        elif ii < 99: test_file.write(str(int(nodeid[ii])) + "  " + '{:0.15e}'.format((float(magfile[ii][0])))+"\n")
        else: test_file.write(str(int(nodeid[ii])) + " " + '{:0.15e}'.format((float(magfile[ii][0])))+"\n")


    
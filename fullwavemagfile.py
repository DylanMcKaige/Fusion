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
Br=pd.read_csv("Br_ERMES.csv", header=None)
Bz=pd.read_csv("Bz_ERMES.csv", header=None)
Bt=pd.read_csv("Bt_ERMES.csv", header=None)
g = open(projname + ".gid/" + projname + "-1.dat")
next(g) # Skip the header

Rcoords = f[f.columns[0]]
Zcoords = f[f.columns[1]]

Brinterp = interpolate.RectBivariateSpline(
    Rcoords,
    Zcoords,
    Br,
    kx=1,
    ky=1,
    s=0
    )

Btinterp = interpolate.RectBivariateSpline(
    Rcoords,
    Zcoords,
    Bt,
    kx=1,
    ky=1,
    s=0
    )
Bzinterp = interpolate.RectBivariateSpline(
    Rcoords,
    Zcoords,
    Bz,
    kx=1,
    ky=1,
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
    z = float(list7.split(")")[0])
    nodeid.append(nodalID)

    #map to minor radius coordinates:
    Brdat = Brinterp(x,y)[0][0]
    Btdat = Btinterp(x,y)[0][0]
    Bzdat = Bzinterp(x,y)[0][0]
    Brdat = Brdat 
    Btdat = Btdat 
    Bzdat = Bzdat 
    magfile.append([Brdat,Bzdat,Btdat])

def formatNumber(n, digits):
    formatter = '{:.' + '{}'.format(digits) + 'f}'
    x = round(n, digits)
    return formatter.format(x)

with open("transposed_" + projname + "_ne.dat", "w") as test_file:
    for ii in range(len(nodeid)):
        if ii < 9:
            if (magfile[ii][0] >= 0) and (magfile[ii][1] >= 0) and (magfile[ii][2] >= 0):
                test_file.write(str(int(nodeid[ii])) + "    " + formatNumber((float(magfile[ii][0])),15) + "  " + formatNumber((float(magfile[ii][1])),15)+ "  " + formatNumber((float(magfile[ii][2])),15)+"\n")
            elif (magfile[ii][0] >= 0) and (magfile[ii][1] >= 0) and (magfile[ii][2] < 0):
                test_file.write(str(int(nodeid[ii])) + "    " + formatNumber((float(magfile[ii][0])),15) + "  " + formatNumber((float(magfile[ii][1])),15)+ " " + formatNumber((float(magfile[ii][2])),15)+"\n")
            elif (magfile[ii][0] >= 0) and (magfile[ii][1] < 0) and (magfile[ii][2] >= 0):
                test_file.write(str(int(nodeid[ii])) + "    " + formatNumber((float(magfile[ii][0])),15) + " " + formatNumber((float(magfile[ii][1])),15)+ "  " + formatNumber((float(magfile[ii][2])),15)+"\n")
            elif (magfile[ii][0] < 0) and (magfile[ii][1] >= 0) and (magfile[ii][2] >= 0):
                test_file.write(str(int(nodeid[ii])) + "   " + formatNumber((float(magfile[ii][0])),15) + "  " + formatNumber((float(magfile[ii][1])),15)+ "  " + formatNumber((float(magfile[ii][2])),15)+"\n")
            elif (magfile[ii][0] >= 0) and (magfile[ii][1] < 0) and (magfile[ii][2] < 0):
                test_file.write(str(int(nodeid[ii])) + "    " + formatNumber((float(magfile[ii][0])),15) + " " + formatNumber((float(magfile[ii][1])),15)+ " " + formatNumber((float(magfile[ii][2])),15)+"\n")
            elif (magfile[ii][0] < 0) and (magfile[ii][1] < 0) and (magfile[ii][2] >= 0):
                test_file.write(str(int(nodeid[ii])) + "   " + formatNumber((float(magfile[ii][0])),15) + " " + formatNumber((float(magfile[ii][1])),15)+ "  " + formatNumber((float(magfile[ii][2])),15)+"\n")
            elif (magfile[ii][0] < 0) and (magfile[ii][1] >= 0) and (magfile[ii][2] < 0):
                test_file.write(str(int(nodeid[ii])) + "   " + formatNumber((float(magfile[ii][0])),15) + "  " + formatNumber((float(magfile[ii][1])),15)+ " " + formatNumber((float(magfile[ii][2])),15)+"\n")
            else:
                test_file.write(str(int(nodeid[ii])) + "   " + formatNumber((float(magfile[ii][0])),15) + " " + formatNumber((float(magfile[ii][1])),15)+ " " + formatNumber((float(magfile[ii][2])),15)+"\n")
        elif ii < 99:
            if (magfile[ii][0] >= 0) and (magfile[ii][1] >= 0) and (magfile[ii][2] >= 0):
                test_file.write(str(int(nodeid[ii])) + "   " + formatNumber((float(magfile[ii][0])),15) + "  " + formatNumber((float(magfile[ii][1])),15)+ "  " + formatNumber((float(magfile[ii][2])),15)+"\n")
            elif (magfile[ii][0] >= 0) and (magfile[ii][1] >= 0) and (magfile[ii][2] < 0):
                test_file.write(str(int(nodeid[ii])) + "   " + formatNumber((float(magfile[ii][0])),15) + "  " + formatNumber((float(magfile[ii][1])),15)+ " " + formatNumber((float(magfile[ii][2])),15)+"\n")
            elif (magfile[ii][0] >= 0) and (magfile[ii][1] < 0) and (magfile[ii][2] >= 0):
                test_file.write(str(int(nodeid[ii])) + "   " + formatNumber((float(magfile[ii][0])),15) + " " + formatNumber((float(magfile[ii][1])),15)+ "  " + formatNumber((float(magfile[ii][2])),15)+"\n")
            elif (magfile[ii][0] < 0) and (magfile[ii][1] >= 0) and (magfile[ii][2] >= 0):
                test_file.write(str(int(nodeid[ii])) + "  " + formatNumber((float(magfile[ii][0])),15) + "  " + formatNumber((float(magfile[ii][1])),15)+ "  " + formatNumber((float(magfile[ii][2])),15)+"\n")
            elif (magfile[ii][0] >= 0) and (magfile[ii][1] < 0) and (magfile[ii][2] < 0):
                test_file.write(str(int(nodeid[ii])) + "   " + formatNumber((float(magfile[ii][0])),15) + " " + formatNumber((float(magfile[ii][1])),15)+ " " + formatNumber((float(magfile[ii][2])),15)+"\n")
            elif (magfile[ii][0] < 0) and (magfile[ii][1] < 0) and (magfile[ii][2] >= 0):
                test_file.write(str(int(nodeid[ii])) + "  " + formatNumber((float(magfile[ii][0])),15) + " " + formatNumber((float(magfile[ii][1])),15)+ "  " + formatNumber((float(magfile[ii][2])),15)+"\n")
            elif (magfile[ii][0] < 0) and (magfile[ii][1] >= 0) and (magfile[ii][2] < 0):
                test_file.write(str(int(nodeid[ii])) + "  " + formatNumber((float(magfile[ii][0])),15) + "  " + formatNumber((float(magfile[ii][1])),15)+ " " + formatNumber((float(magfile[ii][2])),15)+"\n")
            else:
                test_file.write(str(int(nodeid[ii])) + "  " + formatNumber((float(magfile[ii][0])),15) + " " + formatNumber((float(magfile[ii][1])),15)+ " " + formatNumber((float(magfile[ii][2])),15)+"\n")
        else:
            if (magfile[ii][0] >= 0) and (magfile[ii][1] >= 0) and (magfile[ii][2] >= 0):
                test_file.write(str(int(nodeid[ii])) + "  " + formatNumber((float(magfile[ii][0])),15) + "  " + formatNumber((float(magfile[ii][1])),15)+ "  " + formatNumber((float(magfile[ii][2])),15)+"\n")
            elif (magfile[ii][0] >= 0) and (magfile[ii][1] >= 0) and (magfile[ii][2] < 0):
                test_file.write(str(int(nodeid[ii])) + "  " + formatNumber((float(magfile[ii][0])),15) + "  " + formatNumber((float(magfile[ii][1])),15)+ " " + formatNumber((float(magfile[ii][2])),15)+"\n")
            elif (magfile[ii][0] >= 0) and (magfile[ii][1] < 0) and (magfile[ii][2] >= 0):
                test_file.write(str(int(nodeid[ii])) + "  " + formatNumber((float(magfile[ii][0])),15) + " " + formatNumber((float(magfile[ii][1])),15)+ "  " + formatNumber((float(magfile[ii][2])),15)+"\n")
            elif (magfile[ii][0] < 0) and (magfile[ii][1] >= 0) and (magfile[ii][2] >= 0):
                test_file.write(str(int(nodeid[ii])) + " " + formatNumber((float(magfile[ii][0])),15) + "  " + formatNumber((float(magfile[ii][1])),15)+ "  " + formatNumber((float(magfile[ii][2])),15)+"\n")
            elif (magfile[ii][0] >= 0) and (magfile[ii][1] < 0) and (magfile[ii][2] < 0):
                test_file.write(str(int(nodeid[ii])) + "  " + formatNumber((float(magfile[ii][0])),15) + " " + formatNumber((float(magfile[ii][1])),15)+ " " + formatNumber((float(magfile[ii][2])),15)+"\n")
            elif (magfile[ii][0] < 0) and (magfile[ii][1] < 0) and (magfile[ii][2] >= 0):
                test_file.write(str(int(nodeid[ii])) + " " + formatNumber((float(magfile[ii][0])),15) + " " + formatNumber((float(magfile[ii][1])),15)+ "  " + formatNumber((float(magfile[ii][2])),15)+"\n")
            elif (magfile[ii][0] < 0) and (magfile[ii][1] >= 0) and (magfile[ii][2] < 0):
                test_file.write(str(int(nodeid[ii])) + " " + formatNumber((float(magfile[ii][0])),15) + "  " + formatNumber((float(magfile[ii][1])),15)+ " " + formatNumber((float(magfile[ii][2])),15)+"\n")
            else:
                test_file.write(str(int(nodeid[ii])) + " " + formatNumber((float(magfile[ii][0])),15) + " " + formatNumber((float(magfile[ii][1])),15)+ " " + formatNumber((float(magfile[ii][2])),15)+"\n")

"""
Jank way of getting ERMES and PETSc to work if you can't have it automatically run. 

Taken from various PETSc and ERMES scripts.
Creates the X vector file for ERMES to read.


Written by Dylan James Mc Kaige
Created 14/5/2025
Updated 19/5/2025
"""

import numpy as np

# GiD problem folder path, MANUALLY CHANGE THIS
FolderPath = '/to/tour/.gid'
print( '------------------------------------------------------------' )
print( 'Creating X matrix' )
X_vector = np.array( np.fromfile(FolderPath + 'Vector_X_PETSc.dat', dtype=np.dtype( ( np.float64, 2 ) ) ) )
X_vector = np.array( X_vector ).byteswap()
NewFile = open( FolderPath + 'Vector_Xo.bin', 'wb' )
X_vector.tofile(NewFile)
NewFile.close()
print('Done, calculate ERMES in READ mode')
print( '---------------------------------------------------------' )


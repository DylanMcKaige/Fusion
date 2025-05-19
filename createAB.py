"""
Jank way of getting ERMES and PETSc to work. Taken from various PETSc and ERMES scripts.
Creates the A,B matrix and vector files for the PETSc Solver to act on

TODO
1. Take user input in terminal for path

Written by Dylan James Mc Kaige
Created 14/5/2025
Updated 19/5/2025
"""

# Required imports
import PETSc.Solver.PetscBinaryIO as PBIO
import numpy as np
import scipy.sparse
import sys
import subprocess as sp
import os

# GiD problem folder path, MANUALLY CHANGE THIS
FolderPath = '/home/chenvh/mckaigedj/test.gid/'

print( '------------------------------------------------------------' );
print( 'Creating A and B matrices, run ./PETSc_batch.sh after this, then run createVectorX.py' )

# Read system matrices and vector ( AX = B )
B_vector = np.array( np.fromfile( FolderPath + 'Vector_B.bin'      , dtype=np.dtype( ( np.float64, 2 ) ) ) )
A_matrix = np.array( np.fromfile( FolderPath + 'Matrix_A_cmplx.bin', dtype=np.dtype( ( np.float64, 2 ) ) ) )
A_indexs = np.array( np.fromfile( FolderPath + 'Matrix_A_int.bin'  , dtype=np.dtype( ( np.int32  , 2 ) ) ) )

# Join real and imaginary parts
B_vector = B_vector[:,0] + 1j * B_vector[:,1]
A_matrix = A_matrix[:,0] + 1j * A_matrix[:,1]

# Python sparse matrix format
A_matrix = scipy.sparse.csr_matrix( ( A_matrix, ( A_indexs[:,0]-1, A_indexs[:,1]-1 ) ) );

# Convert to PETSc format
io = PBIO.PetscBinaryIO()
io.writeBinaryFile( FolderPath + 'Vector_B_PETSc.dat', [ B_vector.view( PBIO.Vec ), ] )
io.writeBinaryFile( FolderPath + 'Matrix_A_PETSc.dat', [ A_matrix                 , ] )

# Cleaning used objects
del A_matrix
del A_indexs
del B_vector

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 15:00:21 2023

@author: kyso3185

Uses reciprocal relationship to get secondary view factors from the primary view factors

When binary view factors are pre-calculated, the lack of the normal nxn matrix
means that they view factors represent the primary, not the secondary 

These need to be re-fed through the reciprocity relationship to get secondary view factors

Thus, in addition to the large first VF array, a second of equal size will need to be generated 

These will be used in conjunction with the standard, single body nxn arrays for
each body   


View Factor F_ij :: the ratio of the radiation that Area A_j recieves from Area
A_i to the radiation emitted by area A_i. 

Thus view factors F_ji :: ratio of radiation recieved from A_i originating from
A_j to the total radiation emitted by A_j

Original view factor calculation pre-calculates all of the F_ij values. 

This script uses the areas of the mesh triangles, as well as that matrix of 
F_ij values, to generate F_ji values that will tell us about secondary --> 
primary heating. 

"""

# Imports
import numpy as np
import trimesh

###############################################################################
## Functions
###############################################################################

# Function to apply reciprocity relationship (vectorized)
def flipVF(vfSlice,priAreas, secAreas, n, m):
    # vfSlice: Pre Calculated Orientation of binary view factors
    # priAreas: areas of the facets on the primary body. Array
    # secAreas: areas of the facets on the secondary body. Array
    # n: number of facets on the primary body 
    # m: number of facets on the secondary body 
    
    # Reciprocity relationship 
    # A_i * F_ij = A_j * F_ji
    storage = np.zeros((n,m))
    for i in range(n):
        F_ji_row = (priAreas[i] * vfSlice[i,:]) / secAreas
        storage[i,:] = F_ji_row
        total = np.sum(F_ji_row)
        if total >= 1:
            print ("Error in Row {}".format(i))
            print ("     Row sums to more than 1: {}".format(total))
    return storage,total 


###############################################################################
## Execute 
###############################################################################

# Pre Calculated View Factors Path (F_ij)
preVF = np.load('PRE CALCULATED BINARY VIEW FACTORS FILE (.npy or .npz)')

# File to save flipped view factors (F_ji) to
flippedVFFile = 'FLIPPED VIEW FACTORS FILE (.npy)'


# Import shapes 
# Shape model paths 
priModelPath = 'PRIMARY SHAPE MODEL FILE (ex. .obj)'
secModelPath = 'SECONDARY SHAPE MODEL FILE (ex. .obj)'


priMesh = trimesh.load_mesh(priModelPath)
priAreas = priMesh.area_faces * 1e6 # convert km^2 to m^2
priFacetNum = np.shape(priAreas)[0] 

secMesh = trimesh.load_mesh(secModelPath)
secAreas = secMesh.area_faces * 1e6 # convert km^2 to m^2
secFacetNum = np.shape(secAreas)[0]


flippedVFArr = np.zeros(np.shape(preVF))
total = 0
for i in range(np.shape(preVF)[0]):
    flippedVFArr[i],totalTemp = flipVF(preVF[i],secAreas, priAreas, priFacetNum, secFacetNum)
    if totalTemp > total:
            total = totalTemp


np.save(flippedVFFile,flippedVFArr)



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 16:14:39 2021

@author: kyso3185

Description: Script for calculating view factors separately from thermal model.
Requires a mesh (ie. obj or wav format). Uses visibility.py to calculate. 

If sphere is chosen, normals need to be flipped (made negative) if validating 
since normals need to face inward for a closed sphere. Alternatively you can flip
them in the visibility module, but they muse be flipped back after. 
"""


import numpy as np
import math
import kdtree_3d_Kya as kdtree_3d
import visibility_feedback as vis
import time 
from scipy import sparse


# Read in Shape Model
import shapeModelMultiples as shape

#Multiprocessing
from multiprocessing import Manager
import multiprocessing
import pathos
mp = pathos.helpers.mp
from itertools import repeat


global epsilon

def viewFactorsHelper(tris,facetNum,tree,i = 0,vfStorage = {},local = False):
    # feed in two numbers
    # One is for the row (range of 0 to facet numbers)
	# other is for what col to start at (comparable to j) (range is 0 to facet num) Will iterate from this number to facet num

    rowArray,infoArray = vis.visibilityMulti_BF_Feedback(tris = tris,facetNum = facetNum,tree = tree, i = i,local = local) 
    vfStorage[i] = rowArray#return vfcount


def normalize(n):
    # After finding the normal vector for a face, normalize it to length 1
    length = math.sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2])
    normal = n / length
    return normal



# Option to save as sparse matrix (.npz)
sparseSave = False 

    
if __name__ == '__main__':
        
        
    ##############################################################################
    ## Visibility for bowl shaped crater
    ##############################################################################
        
    shapeModelPath = "SHAPE MODEL FILE PATH HERE"
    vfFile = "VIEW FACTOR STORAGE FILE NAME"
    
    local = False
    shape = shape.shapeModel(shapeModelPath,local)
    facets = len(shape.tris)
    
    # KD Tree
    KDTree = kdtree_3d.KDTree(shape.tris,leaf_size = 5, max_depth = 0)
    
    
    # Calculating view factors
    # Establish multiprocessing manager and thread safe storage
    manager = Manager()
    
    iVals = np.arange(0,facets)
    viewFactorStorage = manager.dict()
    infoStorage = manager.dict()
   
    visStartTime = time.perf_counter()
    
    p = mp.Pool(multiprocessing.cpu_count())
    viewFactors = np.zeros((facets,facets))
    viewFactorRun = p.starmap(viewFactorsHelper,zip(repeat(shape.tris),repeat(facets),repeat(KDTree),iVals,repeat(viewFactorStorage),repeat(local)))
    
    
    for i in iVals:
        viewFactors[i,:] = viewFactorStorage[i]
    
    print ("Visibility calculations took: "+str(time.perf_counter()-visStartTime)+" sec")
    print ("Number of final non zero view factors: "+str(np.count_nonzero(viewFactors)))
    
    if sparseSave:
        viewFactors = sparse.csr_matrix(viewFactors)
        sparse.save_npz(vfFile, viewFactors)
        
    
    else: 
        open(vfFile,'w').close()
        np.save(vfFile,viewFactors)

 
        
        
        
        
    



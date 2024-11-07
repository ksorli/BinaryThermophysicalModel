#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 12:45:00 2023

@author: kyso3185

Preceeded by: visibility.py

tldr: View factors for binary

This is a script that runs  the view factor calculation for binary systems. It 
iterates through many different orientations of the primary and secondary to 
create a large lookup table that can then be used by the model later. Current 
parameters to scan over are:

Rotation of primary about axis
Rotation of secondary about primary
To add: Rotation of secondary if not tidally locked 
To add: Consideration of center of mass and separation 

In the model, these values will be tracked and then lookup table can be used 
to find closest match for each facet at that given timestep.  
"""

import numpy as np
from Raytracing_3d import triangle
from Raytracing_3d import ray
import kdtree_3d_Kya
import shapeModelMultiples as shape
import planets 
import time 
import multiprocessing
from copy import deepcopy
from itertools import repeat

###############################################################################
## Functions
###############################################################################


def findVisibleSecondaryFacets(primary, secondary):
    # Vector from origin to secondary mesh 
    secVect = np.asarray(secondary.meshCentroid)
    secNorms = secondary.trimesh_normals
    # Find secondary facets that face the primary (tidally locked case) 
    # Dot of vector to secondary with each secondary facet. 
    dots = np.arccos(np.clip(np.dot(secNorms,secVect), -1, 1))
    whereValid = np.where(np.logical_and(dots >= np.pi / 2.0, dots <= ((3* np.pi) / 2)))

    return dots,whereValid 

def calcInterBodyViewFactors(secOrientation, primary,secondary,whereValid):
    # Given secondary facets that are visible for a tidally locked body figure out the
    # primary facets that can be seen 
    # Get to right orientation 
    priNorms, secNorms, priCentroids, secCentroids = AdvanceBinaryOrbit(primary,secondary,0,secOrientation) 
    
    priVisArr = np.zeros((np.shape(secNorms)[0],np.shape(secNorms)[0]))
    
    # Do a vectorized version of the dot product test to figure out which ones need to be calc'd 
    # NOTE now we want the normals facing each other. If angle between them is <90, they can't see each other 
    # Iterate through primary normals 
    j = 0
    for normal in priNorms:
        
        # Only check against geometrically visible secondary facets 
        points = np.subtract(priCentroids[j],secCentroids[whereValid])
        C = np.einsum('ij,ij->i',points,secNorms[whereValid])
        D = np.dot(points,normal)
        candd = np.where(np.logical_and(C > 0, D < 0))
        facetsSeeOther = np.array(np.ravel(candd))
        
        # Flag visible facets for view factor calculation
        access = whereValid[facetsSeeOther]
        priVisArr[j][access] = True
        j+=1
    
    return priVisArr 

def AdvanceBinaryOrbit(priModel: shape, secModel: shape, priTheta,secTheta):
    # Rotates primary about own axis by a given angle (deg.)
    # Rotates secondary about primary by a given angle (deg.)
    
    # Rotate primary and update vertice/face information 
    priModel.rotateMesh(priTheta)
    
    # Rotates secondary about primary (only for tidally locked secondary)
    secModel.rotateMesh(secTheta)
    
    return priModel.trimesh_normals, secModel.trimesh_normals, priModel.centroids, secModel.centroids

def calcViewFactors_SmallR(tri1: triangle, tri2: triangle,ray1: ray):
    # This is a method of calculating view factors from Rezac & Zhao (2020)
    # Given by Eq. 4 in the paper, or Method M2
    # This method is better suited to situations where the distance between 
    #     areas are small compared to the area itself than the traditional method
    # Version originally given in Abaqus Commercial package (Abaqus 2020)
    # Function returns the value of the two view factors between a triangle pair
    
    # initial ray goes from tri1 to tri2 
    direction = tri1.centroid - tri2.centroid
    ray2 = ray(tri2.centroid,direction)
    
    # Distance between centroids
    r12 = np.linalg.norm(ray1.d)
    
    # Solve for cos of angles 
    cos12 = (np.dot(ray1.d,tri1.normal) / (r12))    
    cos21 = (np.dot(ray2.d,tri2.normal) / (r12))
    if cos12 < 0:
        #print ("cos12 is negative")
        return 0.0#,0.0
    
    if cos21 < 0: 
        #print("cos21 is negative")
        return 0.0#,0.0
    
    # Calculate F12, or view factor from tri1 to tri2
    term1 = (4 * np.sqrt(tri1.area*tri2.area)) / (np.pi**2 * tri1.area)
    term2 = np.arctan((np.sqrt(np.pi * tri1.area) * cos12) / (2*r12))
    term3 = np.arctan((np.sqrt(np.pi * tri2.area) * cos21) / (2*r12))
    F12 = term1*term2*term3    
    
    
    # # Use Fij and reciprocity relationship to solve for Fji
    # F21 = (F12 * tri1.area) / tri2.area
    
    return F12#,F21

def Initial_Traversal(tris, ray, expTri,orgTri):
    # Alternative to KD tree. Checks all tris for intersection and returns the closest
    distExp = kdtree_3d_Kya.find_distance(tri = expTri,ray = ray)
    hitcount = 0
    
    #print ("Checking")
    for tri in tris: 
        if tri == orgTri:
            #print ("Origin tri")
            continue
        if tri == expTri:
            #print ("Expected tri")
            continue
        if tri.intersect_simple(ray) == 1:
            hitcount += 1
            #hit = True
            if hitcount <= 1:
                dist = kdtree_3d_Kya.find_distance(ray=ray,tri = tri)
                #closest_tri = tri
            
            else:
                dist2 = kdtree_3d_Kya.find_distance(ray=ray,tri=tri)
                if dist2 < dist:
                    dist = dist2
            if dist < distExp:
                    return False
    
    return True

def rayCheck_Binary_BruteForce(tris, tri1: triangle, tri2: triangle, rayBetween: ray):
    # cast ray from triangle 1 to triangle 2
    # If intersection occurs and no obstacle found, return visible = True
    expectedIntersect = Initial_Traversal(tris, ray = rayBetween, expTri = tri2, orgTri = tri1)
    return expectedIntersect


def visibilityBinary(priTris, secTris, priFacetNum, secFacetNum,priWhereValid):
    # Function returns array of dimension (no. of primary facets) x (No. of 
        # secondary facets) with the view factors between facet pairs. This 
        # function is only applicable to systems with a tidally locked secondary 
    # Dot products have already been checked so skip to raytracing 
    
    vfArray = np.zeros((priFacetNum,secFacetNum))
    # Iterate through primary facets 
    for i in range(priFacetNum):
        checkSecTris = np.ravel(priWhereValid[i].nonzero())
        if np.any(priWhereValid[i]):

            for j in checkSecTris:

                j = int(j)
                
                direction = secTris[j].centroid - priTris[i].centroid
                rayBetween = ray(priTris[i].centroid, direction)
                
                # Cast rays to see if obstacles between
                # Need to check both primary and secondary facets 
                vis = rayCheck_Binary_BruteForce(secTris, priTris[i], secTris[j], rayBetween)
    
                if vis: 
                    # If visible, calculate view factors
                    vfij = calcViewFactors_SmallR(priTris[i], secTris[j], rayBetween)
                    vfArray[i][j] = vfij

    return vfArray 


def vfBinaryMP(priShape,secShape,secOrientation,whereValid,index):
    # View factor calculation for binaries capable of multiprocessing 
    
    # Do interbody calc with predetermined setup 
    # Make deepcopies to ensure you're only getting the orientation you want
    priShapeCp = deepcopy(priShape)
    secShapeCp = deepcopy(secShape)
    
    # Figure out which secondary facets are visible from each primary facet 
    priVis = calcInterBodyViewFactors(secOrientation, priShapeCp, secShapeCp, whereValid)
    
    # Calc view factors 
    vfArraySlice = visibilityBinary(priShapeCp.tris, secShapeCp.tris, priShape.facetNum, secShape.facetNum, priVis)
    
    return vfArraySlice


# Function to apply reciprocity relationship for binary systems (vectorized)
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
###############################################################################
## Binary View Factor Calculation 
###############################################################################
############################################################################### 
  
    
###############################################################################
## Read in Shape Models and Define Parameters
###############################################################################

print ("Save version")

# Set parameters for a system 
planet = planets.FG31996            # Replace with primary body of interest 
primary = planet 
secondary = planets.FG31996_Second  # Replace with secondary body of interest
shiftVal= None                      # Translational shift to be applied to secondary (km)
shift = (shiftVal,0,0)              # Moves secondary shape model shiftVal km away in x direction

# Read in shape models 
# Shape model paths 
priModelPath = "PRIMARY SHAPE MODEL FILE (ex: .obj)"
secModelPath = "SECONDARY SHAPE MODEL FILE (ex: .obj)"


priShape = shape.shapeModel(priModelPath)
print ("Primary Model Loaded")
secShape = shape.shapeModel(secModelPath, shift = shift)
print ("Secondary Model Loaded with a shift of "+str(secondary.separation) + " km")

sysFacetNum = priShape.facetNum + secShape.facetNum

binaryVFFile = 'BINARY VIEW FACTORS FILE (.npy)'


###############################################################################
# Multiprocessed view factor calculation using scan through binary orientations 

    # Set the resolution of the scan 
    # Theta is rotation of primary
    # Phi is orbital angle of secondary about primary 
    # Psi is the rotation of secondary about local axis (for non tidally locked 
        # secondaries, not yet implemented)
###############################################################################
 
# Timer start
startTime = time.perf_counter()
    
# Define orientations to cycle through 
# Since secondary is tidally locked, rotate secondary around fixed primary 
# Will take the difference between secondary phi and primary theta as angle for lookup  
steps = 160
secStep = 360. / steps
secOrientations = np.linspace(0,360,steps,endpoint = False)# np.arange(steps) * secStep
print ("Steps per 360 deg: {}".format(steps))
print ("First sec orientations: {}".format(secOrientations[:5]))

# Initialize view factor array
vfLookup = np.zeros((steps,priShape.facetNum,secShape.facetNum))
                    
# Orientation identifier for multiprocessing  
iVals = np.arange(np.size(secOrientations))

# Find secondary facets that face the primary 
dots, whereValid = findVisibleSecondaryFacets(priShape, secShape)
whereValid = np.ravel(whereValid)


# Initialize pool 
pool = multiprocessing.Pool(multiprocessing.cpu_count())
print ("pool started")

# Do visibility calcs 
arg1 = priShape
arg2 = secShape
arg3 = secOrientations
arg4 = np.ravel(whereValid)
arg5 = np.arange(steps)
result = pool.starmap(vfBinaryMP,zip(repeat(arg1),repeat(arg2),arg3, repeat(arg4), arg5))
fijRes = np.asarray(result) # View factor results (F_ij)
print ("Size of result: {}".format(np.shape(fijRes)))

###############################################################################
## Save initial set of view factors (F_ij) to .npy file
##    This can be update to .npz if sparse matrices desired 
###############################################################################
np.save(binaryVFFile, fijRes)


###############################################################################
## Flip the view factor array (Go from F_ij to F_ji)
###############################################################################

# File to save flipped view factors (F_ji) to
flippedVFFile = 'FLIPPED VIEW FACTORS FILE (.npy)'


flippedVFArr = np.zeros(np.shape(fijRes))
total = 0
for i in range(np.shape(fijRes)[0]):
    flippedVFArr[i],totalTemp = flipVF(fijRes[i],secShape.mesh.areasSqMeters, priShape.mesh.areasSqMeters, priShape.facetNum, secShape.facetNum)
    if totalTemp > total:
            total = totalTemp


np.save(flippedVFFile,flippedVFArr)

# Timer end 
print ("Total time: "+str(time.perf_counter() - startTime) + " sec")
print ("Complete")

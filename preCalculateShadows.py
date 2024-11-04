#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 12:14:40 2023

@author: kyso3185

Module for calculating shadows through ray tracing. Can be applied to 
binary asteroids, single bodies or landforms with a DEM.  

Generates shadowing arrays (.npy)  
"""


import numpy as np
import rayTracing as raytracing 
import kdTree3d as kdtree

# Read in Shape Model
import shapeModule as shape

# Useful
import time

#Multiprocessing
import multiprocessing
from itertools import repeat



##############################################################################
## Pick your poison
##############################################################################
shadowDEM = False                   # Shadowing for standalone DEM 
shadowBinary = True                 # Shadowing for a binary asteroid 
singleShadow = False                # Shadowing for a single asteroid
dynamicShadows = False              # Shadowing for moving binary asteroid
rotateBodiesIndependently = False   # Shadoing for two bodies moving independently of each other 
dynamicMP = True                    # Multiprocessed version of shadowing 


##############################################################################
## Functions
##############################################################################

def ShadowCheck(tris,ray,expTri,baseDist):
        # Checks through all triangles to find the closest intersect. Least computationally effective method 
        hitcount = 0
        
        for tri in tris:
            result = tri.intersect_simple(ray)
            if result == 1: #tri.intersect_simple(ray) == 1:
                hitcount += 1
                if tri.num == expTri.num: # If origin triangle
                    continue
                dist = kdtree.find_distance(ray=ray,tri = tri)
                if dist < baseDist:
                    return True ,tri.num
                elif dist == baseDist:
                    if tri != expTri:
                        print ("diff tri, same distance")
                        continue
                else:
                    continue
            if result == 2:
                print ("Something's gone wrong: Ray starting point is in triangle")

        return False,expTri.num    # Light makes it to expected facet, no shadow 
    
def NormVisCheck(tri,ray):
    # first takes the dot product of solar vector with the facet normal
    # Make sure ray is of length 1 
    # Remove
    # Not necessarily remove but be wary of the negative you introduced to deal with standalone landforms (see ray direction line)
    # Make negative if doing single (due to flip of axes) 
    rayDirection = -np.asarray(ray.d / np.linalg.norm(ray.d))
    dotproduct = np.dot(rayDirection,tri.normal)
    i = np.arccos(dotproduct) # solar incidence angle 
    if i <= (np.pi / 2.0) or i >= ((3*np.pi) / 2.0): #if on day side
        # If less than 90 deg, cast ray to see if visible 
        return True, dotproduct # on day side
    return False, dotproduct # on night side


def FluxCalc(dot,dist,solar):
    flux = (solar / (dist**2)) * dot
    return flux

def AdvanceBinaryOrbit(priModel: shape, secModel: shape, priTheta,secTheta):
    # Rotates primary about own axis by a given angle (deg.)
    # Rotates secondary about primary by a given angle (deg.)
    
    # Rotate primary and update vertice/face information 
    priModel.rotateMesh(priTheta)
    
    # Rotates secondary about primary (only for tidally locked secondary)
    secModel.rotateMesh(secTheta)
    
    allTrisNew = []
    allTrisNew.append(priModel.tris)
    allTrisNew.append(secModel.tris)
    
    return allTrisNew

def clipShadows(priModel: shape, secModel: shape, solarLocation: np.array, maxBinaryRadius = None):
    # Given the shape models and angle of rotation (primary) and orbit (secondary)
    # determine if it is possible for inter-body shadowing to occur. If it is,
    # Continue with checking shadowing against facets on both bodies. If not, 
    # only check shadowing on same body 
    
    # Find angle between solar direction and placement of secondary 
    vecPriSun = np.asarray(solarLocation - priModel.meshCentroid) # Vector from solar location to primary centroid
    vecPriSec = np.asarray(secModel.meshCentroid - priModel.meshCentroid) # Vector between primary and secondary centroids 
    binarySep = np.linalg.norm(vecPriSec) #Spatial separation between primary and secondary 
    phi = np.arccos((np.dot(vecPriSun, vecPriSec)) / (np.linalg.norm(vecPriSun)* binarySep)) #Angle between 
    if maxBinaryRadius == None:
        maxBinaryRadius = shapes[0].maxDim + shapes[1].maxDim
    # Determine if shadowing is possible 
    if maxBinaryRadius >= binarySep * np.sin(phi):
        # Do shadows if true 
        clip = False
    else: 
        # Only check shadows on own body
        clip = True 
    return clip


def traceShadowsMP(iVal, priOrientation, steps, secStep, priModel: shape, secModel: shape):#,secondaryFluxes):
    print ("iVal: {}".format(iVal))
    # Rotate priamry 
    allTris = AdvanceBinaryOrbit(priModel,secModel,priOrientation,0)
    primaryFluxSingleIndex = np.zeros((steps,priModel.facetNum))
    secondaryFluxSingleIndex = np.zeros((steps,secModel.facetNum))
   
    # Sun
    # Solar location is different for full bodies. Need to place in xy plane
    # For local, right now solar position in on the z axis 
    sLoc = np.asarray([2.124e11,0,0])
    
    # For a given rotation of the primary, do 360 degree orbit of secondary about the primary 
    for k in range(steps):
            clip = clipShadows(priModel, secModel, sLoc,maxBinaryRadius)
            print ("     Clip: {}".format(clip))
            # Flux Storage
            fluxVals = []
            for i in range(len(allTris)):
                shadowCount = 0
                nonShadowCount = 0
                fluxValsTemp = []
                for tri in allTris[i]:
                    sVect = np.asarray(tri.centroid - sLoc)
                    baseDist = np.linalg.norm(sVect)
                    sVect = sVect * 1/baseDist
                    
                    # Vis check with sun angle
                    ray = raytracing.ray(sLoc,sVect)
                    vis = NormVisCheck(tri,ray)              
                    if vis[0]:
                        if not clip: # If clipping function is false and interbody shadowing could happen
                            shadow = ShadowCheck(sum(allTris,[]),ray,tri,baseDist)
                        if clip: # Clipping applied. Eclipses/interbody shadowing impossible. Only check own body
                            shadow = ShadowCheck(allTris[i],ray,tri,baseDist)
                        if shadow[0] == True: 
                            shadowCount += 1
                            fluxValsTemp.append(0)
                        if shadow[0] == False:
                            flux = vis[1]
                            fluxValsTemp.append(flux)
                            nonShadowCount += 1
                    else:
                        fluxValsTemp.append(0) 
                if i == 0:
                    print ("     Primary shadow count: "+str(shadowCount))
                else: 
                    print("     Secondary Shadow Count: "+str(shadowCount))
                fluxVals.append(fluxValsTemp)
                
            # Store fluxes
            primaryFluxSingleIndex[k] = np.asarray(fluxVals[0])
            secondaryFluxSingleIndex[k] = np.asarray(fluxVals[1]) #Manager array
            
            # Only rotate the secondary
            allTris = AdvanceBinaryOrbit(priModel,secModel,0,secStep)
            print ("Secondary to {} degrees".format(orientations[k]))
    
    return primaryFluxSingleIndex,secondaryFluxSingleIndex
    


##############################################################################
## Read in Shape Models
##############################################################################
##%%
paths = []
if shadowDEM:
    shapeModelPath = "/Users/kyso3185/Documents/3D_Model/ShapeModels/BowlCrater_Andrew.obj"
    paths.append(shapeModelPath)

if shadowBinary:
    # 1996 FG3
    primaryPath = "1996FG3_primary_sept20.obj"
    paths.append(primaryPath)
    
    # 1996 FG3 Secondary
    secondaryPath = "1996FG3_second_sept20.obj"
    paths.append(secondaryPath)
    
local = False
shapes = {}
facets = []
allTris = []
#shift = (2.46e3,0,0) #2.46 km semi major axis for 1996 FG3
#shift = (-2.46,0,0) #2.46 km semi major axis for 1996 FG3
shiftVal= 2.46
shift = (shiftVal,0,0) #2.46 km semi major axis for 1996 FG3
#shift = (10,0,0)
n = 7.5 # if incrementing through orientations manually
secRot = 0 #90 + 10*n
priRot = 0 # 0 + 20*n
for i in range(len(paths)):
    shapeModelPath = paths[i]
    if i == 1:
        print ("Secondary")
        shapes[i] = shape.shapeModel(filePath = shapeModelPath,local = local,shift = shift,rot = secRot) # The vector in 3d space to shift the secondary 
    else: 
        print ("Primary")
        shapes[i] = shape.shapeModel(filePath = shapeModelPath,local = local,rot = priRot) #primary 
    facets.append(len(shapes[i].tris))
    allTris.append(shapes[i].tris)

if len(paths) > 1:
    allTrisJoined = sum(allTris,[])
else:
    allTrisJoined = shapes[0].tris
    
originalShapes = shapes

##%%
##############################################################################
## shadowDEM
##############################################################################

if shadowDEM:
    print ("DEM")
    shadowFile = 'ShadowSave.txt'
    open(shadowFile,'w').close()
    incidenceAngles = np.radians(np.asarray([0.,30.,60.,90.]))

    
    sLoc = np.asarray([0,0,1.496e11]) # 1 AU
    solar = 3.826e26 / (4.0*np.pi)
    #sVect = np.asarray([1,0,0])
    for theta in incidenceAngles:
        # rotate to new solar position 
        newSLoc = np.asarray(shape.rotate(sLoc,theta))
        
        fluxVals = []
        
        bruteST = time.perf_counter()
        shadowCount = 0
        nonShadowCount = 0
        visFailCount = 0
        # For tri in all tris 
        for tri in allTrisJoined:
            # Base 0 incidence angle 
            baseSVect = np.asarray(tri.centroid-sLoc) 
            
            # find sVect with new rotated solar position 
            sVect = np.asarray(tri.centroid - newSLoc)
           
            baseDist = np.linalg.norm(sVect)
            sVect = sVect * 1/baseDist
            
            # Include Rotated version 
            ray = raytracing.ray(newSLoc,sVect)
            
            # Test if visible using dot products 
            vis = NormVisCheck(tri,ray)
            # Remove
            # Relace the if statement doing the vis check
            # Trying to figure out which is shadow and which is not
            if vis[0]:
                shadow = ShadowCheck(allTrisJoined, ray, tri,baseDist)
                if shadow[0] == True: 
                    shadowCount += 1
                    fluxVals.append(0)
                if shadow[0] == False:
                    flux = FluxCalc(vis[1], baseDist,solar) # remove
                    fluxVals.append(flux)
                    nonShadowCount += 1
            else:
                visFailCount += 1
                fluxVals.append(0) 
                
        print ("Incidence Angle: "+str(np.degrees(theta)))
        print ("Shadow Count: "+str(shadowCount))
        print ("Vis Fail Count: "+str(visFailCount))
        print("Non Shadow Count: "+str(nonShadowCount))
        print ("Brute force shadows took: "+str(time.perf_counter()-bruteST)+" sec")
        
        primaryFluxes = np.asarray(fluxVals[:shapes[0].facetNum])

        
        #Save this iteration of fluxes
        file = open(shadowFile,'ba')
        np.savetxt(file,[primaryFluxes])
        file.close()
        print ("Angle completed")
    




##############################################################################
## shadowBinary
##############################################################################
if singleShadow:
    print ("Binary")
    priFile = 'ShadowSave_pri_clip.npy'
    open(priFile,'w').close()
    
    secFile = 'ShadowSave_sec_clip.npy'
    open(secFile,'w').close()
    
    # Sun
    # Solar location is different for full bodies. Need to place in xy plane
    # For local, right now solar position in on the z axis 
    sLoc = np.asarray([2.124e11,0,0])
    solar = 3.826e26 / (4.0*np.pi)
    
    
    # Time it 
    bruteST = time.perf_counter()
    
    clip = clipShadows(shapes[0], shapes[1], sLoc)
        
        
    print ("Clip: {}".format(clip))
    
    
    # Flux Storage
    fluxVals = []
    
    #for orientation in orientations: 
    # Do orbital update here, including distance from sun and primary/secondary orbits 
    for i in range(len(allTris)):
        shadowCount = 0
        nonShadowCount = 0
        visFailCount = 0
        fluxValsTemp = []
        for tri in allTris[i]:
            sVect = np.asarray(tri.centroid - sLoc)
            baseDist = np.linalg.norm(sVect)
            sVect = sVect * 1/baseDist
            
            # Vis check with sun angle
            ray = raytracing.ray(sLoc,sVect)
            vis = NormVisCheck(tri,ray)
            
            if vis[0]:
                if not clip: # If clipping function is false and interbody shadowing could happen
                    shadow = ShadowCheck(allTrisJoined,ray,tri,baseDist)
                if clip: # Clipping applied. Eclipses/interbody shadowing impossible. Only check own body
                    shadow = ShadowCheck(allTris[i],ray,tri,baseDist)
                if shadow[0] == True: 
                    shadowCount += 1
                    fluxValsTemp.append(0)
                if shadow[0] == False:
                    flux = FluxCalc(vis[1], baseDist,solar) # remove
                    fluxValsTemp.append(flux)
                    nonShadowCount += 1
            else:
                visFailCount += 1
                fluxValsTemp.append(0) # remove
        fluxVals.append(fluxValsTemp)
        
        if i == 0: 
            print("Primary Stats-----------------------")
        else: 
            print ("Secondary Stats---------------------")
        print ("Shadow Count: "+str(shadowCount))
        print ("Vis Fail Count: "+str(visFailCount))
        print("Non Shadow Count: "+str(nonShadowCount))
        print ("Shadows took: "+str(time.perf_counter()-bruteST)+" sec")
        
    print ("Total brute force shadows took: "+str(time.perf_counter()-bruteST)+" sec")
    
    primaryFluxes = np.asarray(fluxVals[0])
    secondaryFluxes = np.asarray(fluxVals[1])
    
    np.save(priFile,primaryFluxes)
    np.save(secFile,secondaryFluxes)



##############################################################################
## dynamicShadows
##############################################################################
if dynamicShadows:
    print ("Entered dynamicshadows")
    
    priFile = 'ShadowSave_pri_120.npy'
    open(priFile,'w').close()
    
    secFile = 'ShadowSave_sec_120.npy'
    open(secFile,'w').close()
    
    # Sun
    # Solar location is different for full bodies. Need to place in xy plane
    # For local, right now solar position in on the z axis 
    sLoc = np.asarray([2.124e11,0,0])
    solar = 3.826e26 / (4.0*np.pi)
    
    # # Time it 
    # bruteST = time.perf_counter()
    startTime = time.perf_counter()

    
    # Orientations to cycle through 
    steps = 4
    secStep = 360 / steps
    print ("secStep: {}".format(secStep))
    priStep = 360 / steps #(steps/ 2)
    print ("priStep: {}".format(secStep))
    
    primaryFluxes = np.zeros((steps,shapes[0].facetNum)) 
    secondaryFluxes = np.zeros((steps,shapes[1].facetNum))
    
    orientations = np.linspace(secStep,360,steps)
    
    # Added binary radius for shadow clipping 
    maxBinaryRadius = shapes[0].maxDim + shapes[1].maxDim
    
    # for orientation in orientations: 
    # Do orbital update here, including distance from sun and primary/secondary orbits 
    for j in range(steps):
        # Time it 
        bruteST = time.perf_counter()
        print("------------------------------------------------------------")
        print ("Orientation "+str(j))
        
        clip = clipShadows(shapes[0], shapes[1], sLoc,maxBinaryRadius)
        print ("Clip: {}".format(clip))
        # Flux Storage
        fluxVals = []
        for i in range(len(allTris)):
            shadowCount = 0
            nonShadowCount = 0
            #visFailCount = 0
            fluxValsTemp = []
            for tri in allTris[i]:
                sVect = np.asarray(tri.centroid - sLoc)
                baseDist = np.linalg.norm(sVect)
                sVect = sVect * 1/baseDist
                
                # Vis check with sun angle
                ray = raytracing.ray(sLoc,sVect)
                vis = NormVisCheck(tri,ray)              
                if vis[0]:
                    if not clip: # If clipping function is false and interbody shadowing could happen
                        shadow = ShadowCheck(sum(allTris,[]),ray,tri,baseDist)
                    if clip: # Clipping applied. Eclipses/interbody shadowing impossible. Only check own body
                        shadow = ShadowCheck(allTris[i],ray,tri,baseDist)
                    if shadow[0] == True: 
                        shadowCount += 1
                        fluxValsTemp.append(0)
                    if shadow[0] == False:
                        flux = FluxCalc(vis[1], baseDist,solar) # remove
                        fluxValsTemp.append(flux)
                        nonShadowCount += 1
                else:
                    fluxValsTemp.append(0) # remove
            if i == 0:
                print ("Primary shadow count: "+str(shadowCount))
            else: 
                print("Secondary Shadow Count: "+str(shadowCount))
            fluxVals.append(fluxValsTemp)
        
        # Store current orientation 
        # Store mutiples
        primaryFluxes[j] = np.asarray(fluxVals[0])
        secondaryFluxes[j] = np.asarray(fluxVals[1])
        
        print ("Brute force shadows took: "+str(time.perf_counter()-bruteST)+" sec")
        
        
        
        # Update the orbit 
        allTris = AdvanceBinaryOrbit(shapes[0],shapes[1],priStep,secStep)


    print ("Total time: "+str(time.perf_counter() - startTime) + " sec")

        
    # Save to flux files 
    np.save(priFile,primaryFluxes)
    np.save(secFile,secondaryFluxes)
    
    
if rotateBodiesIndependently:
    print ("Entered rotateBodiesIndependently")
    
    priFile = 'ShadowSave_pri_clip.npy'
    open(priFile,'w').close()
    
    secFile = 'ShadowSave_sec_clip.npy'
    open(secFile,'w').close()
    
    # Sun
    # Solar location is different for full bodies. Need to place in xy plane
    # For local, right now solar position in on the z axis 
    sLoc = np.asarray([2.124e11,0,0])
    solar = 3.826e26 / (4.0*np.pi)
    
    # # Time it 
    # bruteST = time.perf_counter()
    startTime = time.perf_counter()

    
    # Orientations to cycle through 
    steps = 180
    secStep = 360 / steps
    print ("secStep: {}".format(secStep))
    priStep = 360 / (steps)
    print ("priStep: {}".format(secStep))
    
    primaryFluxes = np.zeros((steps,steps,shapes[0].facetNum)) 
    secondaryFluxes = np.zeros((steps,steps, shapes[1].facetNum))
    
    orientations = np.linspace(secStep,360,steps)
    priOrientations = np.linspace(priStep,360,steps)
    
    # Added binary radius for shadow clipping 
    maxBinaryRadius = shapes[0].maxDim + shapes[1].maxDim
    
    # Iterate through primary orientations (this will be the first layer of the fluxes array)
    print("------------------------------------------------------------")
    for j in range(steps):
        # Time it 
        bruteST = time.perf_counter()
        
        # Rotate secondary arond by 360 
        for k in range(steps):
            clip = clipShadows(shapes[0], shapes[1], sLoc,maxBinaryRadius)
            print ("     Clip: {}".format(clip))
            # Flux Storage
            fluxVals = []
            for i in range(len(allTris)):
                shadowCount = 0
                nonShadowCount = 0
                fluxValsTemp = []
                for tri in allTris[i]:
                    sVect = np.asarray(tri.centroid - sLoc)
                    baseDist = np.linalg.norm(sVect)
                    sVect = sVect * 1/baseDist
                    
                    # Vis check with sun angle
                    ray = raytracing.ray(sLoc,sVect)
                    vis = NormVisCheck(tri,ray)              
                    if vis[0]:
                        if not clip: # If clipping function is false and interbody shadowing could happen
                            shadow = ShadowCheck(sum(allTris,[]),ray,tri,baseDist)
                        if clip: # Clipping applied. Eclipses/interbody shadowing impossible. Only check own body
                            shadow = ShadowCheck(allTris[i],ray,tri,baseDist)
                        if shadow[0] == True: 
                            shadowCount += 1
                            fluxValsTemp.append(0)
                        if shadow[0] == False:
                            flux = vis[1]
                            fluxValsTemp.append(flux)
                            nonShadowCount += 1
                    else:
                        fluxValsTemp.append(0) # remove
                if i == 0:
                    print ("     Primary shadow count: "+str(shadowCount))
                else: 
                    print("     Secondary Shadow Count: "+str(shadowCount))
                fluxVals.append(fluxValsTemp)
            
            # Store current orientation 
            # Store mutiples
            primaryFluxes[j][k] = np.asarray(fluxVals[0])
            secondaryFluxes[j][k] = np.asarray(fluxVals[1])
            print ("     Brute force shadows took: "+str(time.perf_counter()-bruteST)+" sec")
            
            
            # Only rotate the secondary
            allTris = AdvanceBinaryOrbit(shapes[0],shapes[1],0,secStep)
            print ("Secondary to {} degrees".format(orientations[k]))
            
        print("------------------------------------------------------------")
        # Only advance the primary
        allTris = AdvanceBinaryOrbit(shapes[0], shapes[1], priStep, 0)
        print ("Primary rotated to {} degrees".format(priOrientations[j]))

    print ("Total time: "+str(time.perf_counter() - startTime) + " sec")

        
    # Save to flux files 
    np.save(priFile,primaryFluxes)
    np.save(secFile,secondaryFluxes)
    
    

    
if dynamicMP:
    
    # Set things up 
    print ("Entered dynamicsMP")
    
    priFile = 'ShadowSave_pri_mp.npy'
    open(priFile,'w').close()
    
    secFile = 'ShadowSave_sec_mp.npy'
    open(secFile,'w').close()
    
    # # Time it 
    startTime = time.perf_counter()

    
    # Orientations to cycle through 
    steps = 180
    secStep = 360 / steps
    priStep = 360 / steps
    priRotVals = np.arange(steps) * priStep
    secOrbVals = np.arange(steps) * secStep
    
    
    primaryFluxes = np.zeros((steps,steps,shapes[0].facetNum)) 
    secondaryFluxes = np.zeros((steps,steps, shapes[1].facetNum))
    
    orientations = np.linspace(secStep,360,steps)
    priOrientations = np.linspace(priStep,360,steps)
    
    # Added binary radius for shadow clipping 
    maxBinaryRadius = shapes[0].maxDim + shapes[1].maxDim
    
    # Orientation of primary 
    iVals = np.arange(np.size(priOrientations))
    
    # Initialize pool 
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    print ("pool started")
    
    arg1 = iVals
    arg2 = priOrientations
    arg3 = steps
    arg4 = secStep
    arg5 = shapes[0]
    arg6 = shapes[1]
    result = pool.starmap(traceShadowsMP,zip(arg1,arg2,repeat(arg3), repeat(arg4),repeat(arg5),repeat(arg6)))
    res = np.asarray(result)
    print ("Shape of result {}".format(np.shape(res)))
    
    primaryShadows = res[:,0,:,:]
    secondaryShadows = res[:,1,:,:]
    
    # Save to flux files 
    np.save(priFile,primaryShadows)
    np.save(secFile,secondaryShadows)

    print ("Complete")
    print ("Dot Products saved to files")
    


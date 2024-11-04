#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 14:27:25 2021

@author: kyso3185

Description: This code performs calculations upon setup and initialization 
of the model that determine topography, what other facets a single facet can
see and well it can see it.

For a single body, this uses three tests. Initially, normals of the facets are 
compared. If more than 90 degrees different, facets cannot see each other and 
visibility is returned as 0. If visibility is not 0, a ray is then traced 
between the two to check for obstructions. If no obstructions, the view factor
between the two triangular facets is calculated
"""


import numpy as np 
from Raytracing_3d import triangle
from Raytracing_3d import ray
from kdtree_3d_Kya import KDTree
import math



##############################################################################
## Normals check
## If normals are more than 90 degrees apart, facets are not visible to each other
##############################################################################

# DELETE 
def normalize(n):
    # After finding the normal vector for a face, normalize it to length 1
    length = math.sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2])
    normal = n / length
    return normal

# DELETE  
def visCheck_oldlatlongversion(tri1: triangle, tri2: triangle):
    # if sufficiently close in terms of latitude and longitude
    # Make an exception for the pole, where large longitude differences are close
    if tri1.lat >=75 and tri2.lat >= 75:
       return True
    
    elif tri1.lat <=-75 and tri2.lat <= -75:
        return True
            
    elif tri1.lat - tri2.lat <= 45 and np.abs(tri1.long - tri2.long) <= 45:
        dotproduct= np.dot(tri1.normal, tri2.normal)
        angle = math.acos(dotproduct)
        #print ("angle "+str(angle))
        #if angle >= np.radians(1): # Make sure normals aren't parallel
        if angle <= (math.pi / 2.0) or angle >= ((3*math.pi) / 2.0):
                # if not on opposite ("nighttime") side
                #return True
            return True
    return False


def visCheck(tri1: triangle, tri2: triangle,local = False):
    # if sufficiently close in terms of latitude and longitude
    # if a global body:
    if local == False:
        # Check this by calculating the dot product of the vectors from the origin to the centroids
        locationAngle = np.arccos(np.dot(tri1.rVect,tri2.rVect))
        if np.degrees(locationAngle) <= 45: 
            # if relatively close on the global sphere, check to make sure facet inclinations are visible to each other 
            dotproduct= np.dot(tri1.normal, tri2.normal)
            angle = math.acos(dotproduct) 
            if angle <= (math.pi / 2.0) or angle >= ((3*math.pi) / 2.0): # if not on opposite ("nighttime") side
                return True
        return False
    else:
        dotproduct= np.dot(tri1.normal, tri2.normal)
        angle = math.acos(dotproduct)
        if angle <= (math.pi / 2.0) or angle >= ((3*math.pi) / 2.0): # if not on opposite ("nighttime") side
            return True
        return False
        

# unaltered version
def rayCheck_KD(tri1: triangle, tri2: triangle, tree: KDTree, rayBetween: ray):
    # cast ray from triangle 1 to triangle 2
    # If intersection occurs and no obstacle found, return visible = True
    #expectedIntersect = tree.root.Initial_Traversal(ray = rayBetween, expTri = tri2,startTri = tri1)
    expectedIntersect = tree.root.Initial_Traversal_KD(ray = rayBetween,count = 0, expTri = tri2)
    return expectedIntersect


def rayCheck_BruteForce(tris, tri1: triangle, tri2: triangle, tree: KDTree, rayBetween: ray):
    # cast ray from triangle 1 to triangle 2
    # If intersection occurs and no obstacle found, return visible = True
    #print ("In raycheck")
    expectedIntersect = tree.root.Initial_Traversal(tris, ray = rayBetween, expTri = tri2, orgTri = tri1)
    return expectedIntersect



def calcViewFactors_NonSingularR(tri1: triangle, tri2: triangle,ray: ray):
    # Given the vertices of two different arbitrarily oriented planar triangles
    # this function uses the method described in 
    # https://abaqus-docs.mit.edu/2017/English/SIMACAETHERefMap/simathe-c-viewfactor.htm 
    # to calculate view factors
    # Function returns the value of the view factor for 1 triangle pair
    
    #
    # calcViewFactors is the default but since R is singular, this is a more
    # complex version for more general surfaces 
    # 
    #
    # if distance apart is large compared to the elemental areas
    #print ("Calculating View Factors for 1 facet pair")
    r12 = np.linalg.norm(ray.d)
    areaVF = (((tri1.area * tri2.area)**0.25) / \
              (math.pi)**2.)*math.atan((math.sqrt(math.pi*tri1.area)\
                                       *(np.dot(ray.d,tri1.normal) / r12)) / \
                                       (2.*r12))*math.atan((math.sqrt(math.pi*tri2.area)*\
                                                                             (np.dot(ray.d,tri2.normal) / r12)) / \
                                                                             (2.*r12))
    # Remove
    # Commented out the two lines below the next(if areaVF < 0 statement) and made the area VF the absolute value
    areaVF = abs(areaVF)
    if areaVF < 0:
        return 0.0,0.0
    else:
        vf12 = areaVF / tri1.area
        vf21 = areaVF / tri2.area
        return vf12,vf21
    
    

def viewFactors2(tri1: triangle, tri2: triangle, ray: ray):
    # See section 2.1 of Hancock et al. (2021): "A GPU Accelerated ray-tracing...."
    # https://www.sciencedirect.com/science/article/pii/S0360544221006873?via%3Dihub
    # Returns view factors. vf12 is the fraction of energy exiting
    # surface 1 that impinges on surface 2 (or vice versa for vf21). vf12 and 
    # vf21 are related by the reciprocity relationship, such that 
    # area(surface 1) * vf12 = area(surface 2) * vf21
    r12 = tri1.centroid - tri2.centroid
    r12Mag = np.linalg.norm(r12)
    theta1 = math.acos(np.dot(tri1.normal,r12) / r12Mag)
    theta2 = math.acos(np.dot(tri2.normal,r12) / r12Mag)
    areaVF = (np.cos(theta1)*np.cos(theta2)) / (math.pi * (r12Mag**2))
    if areaVF < 0:
        return 0.0,0.0
    else:
        vf12 = areaVF / tri1.area
        vf21 = areaVF / tri2.area
        return vf12,vf21
    
    
    

def calcViewFactors(tri1: triangle, tri2: triangle,ray: ray):
    # Given the vertices of two different arbitrarily oriented planar triangles
    # this function uses the method described by link below to calculate view factors
    # Function returns the value of the two view factors between a triangle pair
    # 
    # Using view factor approximation for two elemental areas a1 and a2 
    # from https://abaqus-docs.mit.edu/2017/English/SIMACAETHERefMap/simathe-c-viewfactor.htm
    # 
    
    # if a1 and a2 are small compared to distance between the two them 
    # triangle normals are already normalized to length 1, so division by 
    # magnitude of the vector between triangles is all that is required to get
    # the cos(theta) where theta(1,2) is the angle between the vector between
    # and the normals for triangles 1 and 2 
    r12 = np.linalg.norm(ray.d)
    areaVF = (tri1.area*(np.dot(ray.d,-tri1.normal) / r12)) * \
        (tri2.area* (np.dot(-ray.d,-tri2.normal) / r12)) / (np.pi * r12**2)
    #print (areaVF)
    if areaVF < 0:
        return 0.0,0.0
    else:
        vf12 = areaVF / tri1.area
        vf21 = areaVF / tri2.area
        return vf12,vf21
    
    
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
		return 0.0#,0.0
	if cos21 < 0: 
		return 0.0#,0.0
	
	# Calculate F12, or view factor from tri1 to tri2
	term1 = (4 * np.sqrt(tri1.area*tri2.area)) / (np.pi**2 * tri1.area)  
	term2 = np.arctan((np.sqrt(np.pi * tri1.area) * cos12) / (2*r12)) 
	term3 = np.arctan((np.sqrt(np.pi * tri2.area) * cos21) / (2*r12)) 
	F12 = term1*term2*term3    

	# # Use Fij and reciprocity relationship to solve for Fji
	# F21 = (F12 * tri2.area) / tri1.area #swap
	
	return F12#,F21
    

# Unaltered version
def visibility(tris, facetNum, tree: KDTree):
    # Function will return two things. 
    # 1) a boolean array of dimension (no. of facets) x (no. of facets). 
        #Each entry represents 1 facet pair. 
        # None implies facet with itself, False implies facets are not visible to 
        # each other and True implies facets are visible to each other. 
    # 2) An array of dimension (no. of facets) x (no. of facets) with the view 
        # factors between facet pairs. View factors are only calculated for 
        # facet pairs determined to be visible by the boolean array
    #facetNum = 100
    rows,cols = (facetNum,facetNum)

    viewFactors = np.asarray([[0.0]*cols]*rows)          # View Factor storage array
    #for i in range(len(tris)):
    for i in range(rows): #formerly rows
        print ("i: "+str(i))
        raytrace_count = 0
        vfcount = 0
        for j in range(i,cols): 
        #for j in range(100,200):
            #print ("j: "+str(j))
            if i != j: # view factor with itself should be 0 

                normVis = visCheck(tris[i],tris[j]) #Check dot products

                if normVis == True:
                    #print ("geometry check passed")
                    direction = tris[j].centroid - tris[i].centroid
                    rayBetween = ray(tris[i].centroid,direction)
                    raytrace_count += 1
                    vis = rayCheck_KD(tris[i],tris[j],tree,rayBetween)
                    
                    if vis:
                        #print ("raytrace: "+str(vis))
                        # If visible, calculate view factors
                        #print ("Entered view factor statement")
                        
                        vfij,vfji = calcViewFactors_NonSingularR(tris[i],tris[j], rayBetween)
                        #vfij,vfji = calcViewFactors(tris[i],tris[j], rayBetween)
                        #vfij,vfji = viewFactors2(tris[i],tris[j], rayBetween)
                        vfcount += 1
                        viewFactors[i][j] = vfij 
                        viewFactors[j][i] = vfji

        print (str(raytrace_count)+"rays traced with "+str(vfcount)+" nonzero view factors")
                                                                         
    return viewFactors


def visibilityMulti(tris, facetNum, tree: KDTree,i = 0,local = False):
    # Function will return two things. 
    # 1) a boolean array of dimension (no. of facets) x (no. of facets). 
        #Each entry represents 1 facet pair. 
        # None implies facet with itself, False implies facets are not visible to 
        # each other and True implies facets are visible to each other. 
    # 2) An array of dimension (no. of facets) x (no. of facets) with the view 
        # factors between facet pairs. View factors are only calculated for 
        # facet pairs determined to be visible by the boolean array
	rowArray = np.zeros(facetNum)
	colArray = np.zeros(facetNum)
	vfcount = 0  
	for j in range(i,facetNum):
		if i != j:  #If not the same facet
			normVis = True #visCheck(tris[i],tris[j],local) #Check dot products
			if normVis:
				# visible
				direction = tris[j].centroid - tris[i].centroid
				rayBetween = ray(tris[i].centroid,direction)
                
				# Cast rays between facets to see if there are obstacles between
				vis = rayCheck_BruteForce(tris, tris[i],tris[j],tree,rayBetween)

				if vis:
					# If visible, calculate view factors
					vfij,vfji = calcViewFactors_SmallR(tris[i],tris[j],rayBetween)
					vfcount += 1
					rowArray[j] = vfij 
					colArray[j] = vfji
	                    
		if i == j:
			vfcount = 0                                                                  
	return rowArray,colArray#,vfcount



def visibilityMulti_BF(tris, facetNum, tree: KDTree,i = 0,local = False):
    # Function will return two things. 
    # 1) a boolean array of dimension (no. of facets) x (no. of facets). 
        #Each entry represents 1 facet pair. 
        # None implies facet with itself, False implies facets are not visible to 
        # each other and True implies facets are visible to each other. 
    # 2) An array of dimension (no. of facets) x (no. of facets) with the view 
        # factors between facet pairs. View factors are only calculated for 
        # facet pairs determined to be visible by the boolean array
    rowArray = np.zeros(facetNum)
    for j in range(facetNum):
        if i != j:  #If not the same facet
            normVis = visCheck(tris[i],tris[j],local) #Check dot products
            if normVis:
                # visible
                direction = tris[j].centroid - tris[i].centroid
                rayBetween = ray(tris[i].centroid,direction)
                
                # Cast rays between facets to see if there are obstacles between                    
                vis = rayCheck_BruteForce(tris, tris[i],tris[j],tree,rayBetween)

                if vis:
                    # If visible, calculate view factors
                    vfij = calcViewFactors_SmallR(tris[i],tris[j], rayBetween)
                    rowArray[j] = vfij   
                                                                      
    return rowArray



def visibilityMulti_BF_Feedback(tris, facetNum, tree: KDTree,i = 0,local = False):
    # Function will return two things. 
    # 1) a boolean array of dimension (no. of facets) x (no. of facets). 
        #Each entry represents 1 facet pair. 
        # None implies facet with itself, False implies facets are not visible to 
        # each other and True implies facets are visible to each other. 
    # 2) An array of dimension (no. of facets) x (no. of facets) with the view 
        # factors between facet pairs. View factors are only calculated for 
        # facet pairs determined to be visible by the boolean array
    rowArray = np.zeros(facetNum)
    infoArray = np.zeros((facetNum,2))
    for j in range(facetNum):
        if i != j:  #If not the same facet
            normVis = True #visCheck(tris[i],tris[j],local) #Check dot products
            if normVis:
                # visible
                infoArray[j][0] = 1 # If passed dot product check, mark as 1. Else 0
                direction = tris[j].centroid - tris[i].centroid
                rayBetween = ray(tris[i].centroid,direction)
                
                # Cast rays between facets to see if there are obstacles between                    
                vis = rayCheck_BruteForce(tris, tris[i],tris[j],tree,rayBetween)

                if vis:
                    infoArray[j][1] = 1 # If passed raytrace check, mark as 1. Else 0
                    # If visible, calculate view factors
                    vfij = calcViewFactors_SmallR(tris[i],tris[j], rayBetween)
                    rowArray[j] = vfij   
                                                                      
    return rowArray,infoArray


def visibilityMulti_KDTree_Feedback(tris, facetNum, tree: KDTree,i = 0,local = False):
    # Function will return two things. 
    # 1) a boolean array of dimension (no. of facets) x (no. of facets). 
        #Each entry represents 1 facet pair. 
        # None implies facet with itself, False implies facets are not visible to 
        # each other and True implies facets are visible to each other. 
    # 2) An array of dimension (no. of facets) x (no. of facets) with the view 
        # factors between facet pairs. View factors are only calculated for 
        # facet pairs determined to be visible by the boolean array
    rowArray = np.zeros(facetNum)
    infoArray = np.zeros((facetNum,2))
    for j in range(facetNum):
        if i != j:  #If not the same facet
            normVis = True #visCheck(tris[i],tris[j],local) #Check dot products
            if normVis:
                # visible
                infoArray[j][0] = 1 # If passed dot product check, mark as 1. Else 0
                direction = tris[j].centroid - tris[i].centroid
                rayBetween = ray(tris[i].centroid,direction)
                
                # Cast rays between facets to see if there are obstacles between                    
                vis = rayCheck_KD(tris, tris[i],tree,rayBetween)

                if vis:
                    infoArray[j][1] = 1 # If passed raytrace check, mark as 1. Else 0
                    # If visible, calculate view factors
                    vfij = calcViewFactors_SmallR(tris[i],tris[j], rayBetween)
                    rowArray[j] = vfij    
                                                                      
    return rowArray,infoArray















#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 14:15:22 2023

@author: kyso3185

Complement for binary systems is heat3d_binary_shadowing_vectorized.py

Full single body thermal model, using shadowing, view factors and obliquity 
built off of vectorized thermal model. 

Shadows and view factors are precalculated and fed in. 


"""


# Physical constants:
sigma = 5.67051196e-8 # Stefan-Boltzmann Constant
#S0 = 1361.0 # Solar constant at 1 AU [W.m-2]
chi = 2.7 # Radiative conductivity parameter [Mitchell and de Pater, 1994]
R350 = chi/350**3 # Useful form of the radiative conductivity
TWOPI = 6.283185307

# Numerical parameters:
F = 0.1 # Fourier Mesh Number, must be <= 0.5 for stability
m = 10 # Number of layers in upper skin depth [default: 10]
n = 5 # Layer increase with depth: dz[i] = dz[i-1]*(1+1/n) [default: 5]
b = 20 # Number of skin depths to bottom layer [default: 20]

# Accuracy of temperature calculations
# The model will run until the change in temperature of the bottom layer
# is less than DTBOT over one diurnal cycle
DTSURF = 0.1 # surface temperature accuracy [K]
DTBOT = DTSURF # bottom layer temperature accuracy [K]
NYEARSEQ = 1.0 # equilibration time [orbits]
NPERDAY = 24 # minimum number of time steps per diurnal cycle

# NumPy is needed for various math operations
import numpy as np
#import math

# MatPlotLib and Pyplot are used for plotting
import matplotlib.pyplot as plt
import matplotlib as mpl

# Methods for calculating solar angles from orbits
import orbits

# Planets database
import planets

# Read in Shape Model
#import shapeModel as shape
import shapeModelMultiples as shape

# Visibility Calculations
#import visibility as vis

# Reflection Calculations
#import Reflection as reflect

# For interpolation functions
import scipy.interpolate as interp
import scipy.sparse as sparse

# For interpreting facets 
#import Raytracing_3d as raytracing 
import kdtree_3d_Kya as kdtree_3d

#from copy import deepcopy

# #Multiprocessing
# from multiprocessing import Manager
import multiprocessing
# #from pathos.multiprocessing import ProcessingPool as Pool
# import pathos
# mp = pathos.helpers.mp
from itertools import repeat
# import functools

from numba import jit
#import mkl; mkl.set_num_threads(4)


'''
#####
EDITS
#####
'''
# Facets Class
# Going to start with 1 facet 
#import facets
# fluxVals = np.genfromtxt('/Users/kyso3185/Documents/3D_Model/1996FG3_FluxFile.txt')
# facetNum = np.shape(fluxVals)[0]
# hours = np.shape(fluxVals)[1]
# testFlux = fluxVals[0,:] #24 'hour' worth of fluxes for 1 facet 

#Set up solar values and physical constants 
sigma = 5.67051196e-8  # Stefan-Boltzmann Constant (W m^-2 K^-4)
solarLum = (3.826e26 / (4.0*np.pi))
solar = 3.826e26 / (4.0*np.pi)
#sVect = [1,0,0] #unit vector in x direction for sun direction 

#sVect,fluxFile = vectorSelect(0)
s0 = solarLum  #[W.m-2] Intensity of sun at distance of asteroid 
#print (s0)

# version2 = True
# num = 1000
# print ("Num: {}".format(num))
#print ("jit advance")



        



##############################################################################
## Single Body Model Class
##############################################################################
class SingleModel(object):
    def __init__(self, shapeModelPath, planet = planets.Moon, ndays=1,nyears = 1.0,local = False,shadowLookupPath = None, vfPath = None):
    
        # Change Equilibration Time
        NYEARSEQ = nyears   
        print ("Equilibration time: {} years".format(NYEARSEQ))
    
        # Initialize
        self.planet = planet
        #self.planet = priPlanet # For bulk orbital values 
        print ("Selected Planet: {}".format(planet.name))
        self.Sabs = self.planet.S * (1.0 - self.planet.albedo)
        self.r = self.planet.rAU # solar distance [AU]
        self.nu = np.float() # orbital true anomaly [rad]
        self.nudot = np.float() # rate of change of true anomaly [rad/s]
        self.dec = np.float() # solar declination [rad]
        self.solarFlux = np.float() # Solar flux (W/m^2) at the body given the distance to the sun
        #self.sPos = np.array([0,0,0])
       
        # Read in Shape Model & initialize corresponding values
        self.local = local
        self.shape = shape.shapeModel(shapeModelPath,local)
        self.facetNum = self.shape.facetNum
        # List of ints from 0 to the number of facets. Used to keep track of which facet you're working on
        iVals = np.arange(0,self.facetNum)
        print ("Shape Model Loaded")
        
        # Initialize arrays for latitudes, fluxes, shadows 
        self.lats = self.shape.lats#np.zeros(self.facetNum) # np array of latitudes for each facet
        
        # Rotation of the primary about own axis 
        # Used for shadowing and view factor setup  
        self.theta = np.float() # Radians, starts at 0.
        
        # # Rotation of the secondary about primary
        # # Used for shadowing setup 
        # self.phi = np.float() # Radians, starts at 0
        
        # Load shadow array
        shadowArr = np.load(shadowLookupPath)
        shadowArr[shadowArr <= 0] =  0.
        self.shadows = shadowArr

        print ("Shadow array loaded")
        
        
        # Initialize profile(s)
        self.profile = profile(planet = self.planet, facetNum = self.facetNum, lat = self.lats, emis = self.planet.emissivity)

        
        # # Attempting to shorten the aphelion run by feeding in temps from another run. 
        # if startPriT != None and startSecT != None:
        #     priStartT = np.load(startPriT)[0]
        #     secStartT = np.load(startSecT)[0]
            
        #     self.priProfile.T = priStartT
        #     self.secProfile.T = secStartT
    
    
        # Model run times
        # Equilibration time -- TODO: change to convergence check
        self.equiltime = NYEARSEQ*planet.year - \
                        (NYEARSEQ*planet.year)%planet.day
        # Runtime for output
        self.endtime = self.equiltime + ndays*planet.day
        self.t = 0.
        
        # Get conductivity associated with max temperatures to set timestep 
        maxTemp = getPerihelionSolarIntensity(planet)
        maxTempK = thermCond(self.profile.kc, maxTemp)
        
        # Get Timestep 
        # Uses mean temperature
        self.dt = getTimeStep(self.profile, planet.day)
        # Uses max temperature at perihelion 
        #self.dt = getHighTempTimeStep(self.profile, maxTempK, planet.day)
        self.dtout = self.dt
        
        # Check for maximum time step
        dtmax = self.planet.day/NPERDAY
        if self.dt > dtmax:
            self.dtout = dtmax
        
        # Array for output temperatures and local times
        N_steps = np.int((ndays*planet.day)/self.dtout)
        self.N_steps = N_steps
        print ("Timesteps: {}".format(self.N_steps) )
        
        #  Layers
        N_z = np.shape(self.profile.z)[0]
        self.N_z = N_z
        

        # Temperature and local time arrays 
        self.T = np.zeros([N_steps, N_z,self.facetNum])
        self.lt = np.zeros((N_steps,3)) #[N_steps, 3]) # Time, theta, nu
        self.readoutScat = np.zeros([N_steps, self.facetNum])


        # Resolution for view factors and shadowing 
        # count how many facets around equator (ie have lat of ~0?) 
        # This needs to be split into two for a binary 
        if shadowLookupPath == None:
            print ("Calculating facet resolution around equator")
            lats = np.zeros(self.facetNum)
            longs = np.zeros(self.facetNum)
            equatorial_facets = 0
            for tri in self.shape.tris:
                lats[tri.num] = tri.lat
                longs[tri.num] = tri.long
                if tri.lat < 5 and tri.lat >= 0:
                    equatorial_facets += 1
                    
            
            # Find how frequently you need to update shadowing and view factors 
            self.deg_per_facet = 360 / equatorial_facets # degrees per facet
            print ("Equatorial facets: "+str(equatorial_facets))
            print ("Degrees per facet: "+str(self.deg_per_facet))
            
        else: 
            print ("Shadowing Lookup Table provided. Skipping resolution check")
        
        
        
        # View Factors
        if vfPath != None:
            if vfPath.endswith('.npy'):
                self.viewFactors = np.load(vfPath) 
                #self.IncludeReflections = True
            elif vfPath.endswith('.npz'):
                vf = sparse.load_npz(vfPath)
                self.viewFactors = np.asarray(vf.toarray())
                print (np.shape(self.viewFactors))
                #self.IncludeReflections = True 
            
        else: 
            print ("WARNING: AUTOMATIC VIEW FACTORS NOT ENABLED! \n Proceeding with null view factors")
            #self.IncludeReflections = False
            
        
        # # View factors 
        # if vfFile != None: # If you're reading in a precalculated set of view factors 
        #     viewFactors = np.genfromtxt(vfFile)
        #     viewFactors = np.transpose(viewFactors)
        #     #viewFactors = np.zeros(np.shape(viewFactors)) # Remove. This is for debugging shadowing 
        # else:
        #     # Calc view factors with multiprocessing 
        #     # Initialize view factor array 
        #     print ("ERROR ERROR ERROR")
        #     print ("Custom view factors not enabled yet. Need to feed in pre calculated view factors")

        #     viewFactors = np.zeros((self.facetNum,self.facetNum))
        #     # viewFactorRun = p.starmap(self.viewFactorsHelper,zip(iVals,repeat(viewFactorStorage),repeat(local)))
            
        #     # for i in iVals:
        #     #     viewFactors[:,i] = viewFactorStorage[i][1]
                
        #     # for i in iVals:
        #     #     viewFactors[i,i:self.facetNum] = viewFactorStorage[i][0][i:self.facetNum]
                
        #     # np.savetxt(vfFile,viewFactors)
        # print ("View factors saved")
        
            # Test case: 4D (aka not including non tidally locked secondary)
            # theta of 0, pi/2, pi. Would eventaully have int(equatorial_facets) or ~120
            # phi of 0, pi/4, pi/2. Would eventually have int(equatorial_facets) or ~120
            # Run with each combination 
            
        # steps = int(equatorial_facets)
        
                
        # Sun location initialization 
        self.sLoc = np.asarray([0,0,self.planet.rsm])#np.asarray([0,0,1.496e11]) # 1 AU # make this self.sLoc
        self.newSLoc = self.sLoc
        self.baseDist = np.float()
        
        print ("Completed Visibility Setup")
        
        self.vfCount = 0
        
        
        
    def run(self,endTrueAnomaly = 0., calcShadows = False, vfFreq = 10):
        # print ("Entered run")
        # # Precompute view factors 
        # # ** Need to put view factors in 
        
        # Precompute shadows 
        if calcShadows: # Shadows have not been precalculated 
            print ("Shadows Needed. Initiating shadow calculations")
        
        # Might need to update shadows in a regular, but infrequent basis to make sure you're accomodating siderial vs synodic days 
        shadowSteps = np.shape(self.shadows)[0]
        
        # Equilibrate 
        print ("-------------Beginning Equilibration---------------")
        i = 0
        vfUpdate = True
        vfCount = 0
        while self.t < self.equiltime or self.nu < np.radians(endTrueAnomaly): #optional input of TA to stop at (works if you're doing complete orbits for equilibration)
            # Update orbit & binary angles 
            self.updateOrbit(equilibration = True)
            # Get slice of shadow and view factor arrays that correspond to current position 
            shadowVals = self.sliceShadows(shadowSteps) # Slice the sections of lookup arrays you need  
            # If enough time has passed that you need to do a sidereal vs synotic update do that 
            self.advance(shadowVals, vfUpdate)
            
            # if i > 100: 
            #     break
            
            # Updating every 5 
            if vfCount == 0:
                vfUpdate = True
            else: vfUpdate = False

            i += 1
            vfCount += 1
            
            if vfCount > vfFreq:
                vfCount = 0
                
        if endTrueAnomaly != 0.0:       
            print ("Time at end of equilibration: {} s".format(self.t))
            print ("Target True Anomaly/Distance was: {}".format(endTrueAnomaly))
            print ("     Actual TA was: {}".format(np.degrees(self.nu)))
            print ("     Actual r was: {}".format(self.r))
                
        print ("Equilibration reached. Saving temperatures")
        # Run through end of model and store output
        self.dt = self.dtout
        self.t = 0.0  # reset simulation time
         
        for i in range(0, self.N_steps):
            self.updateOrbit()
            shadowVals = self.sliceShadows(shadowSteps)
            self.advance(shadowVals,vfUpdate = True)
            self.T[i,:,:] = self.profile.T # temperature [K]
            self.lt[i] = np.asarray([self.t / self.planet.day * 24.0, self.theta, self.nu])  # local time [hr], theta (rad), true anomaly (rad)
            #vfUpdate = not vfUpdate
            
            # #***
            # self.OutFlux[i] = self.Qs
            # self.OutIndex[i] = self.indices
        print ("VF Update called {} times".format(self.vfCount))
 
    def advance(self,shadowVals,vfUpdate = False):
        # Assuming geometry is handled separately 
        #   Aka view factors, shadowing 
        
        # Finding Qs for each facet 
        reflectedFlux = self.surfModelFlux_Vectorized(shadowVals,self.profile)
        
        if vfUpdate: 
            self.vfCount += 1
            # Reflections/scattering 
            # Output array of Qs values including scattering and reflection 
            # If a view factor matrix was provided and 
            # if self.IncludeReflections:
            #     self.priProfile.Qs = reflect.totalScatteredFluxVectorized(self.priVF, self.priProfile.Qs, self.priProfile.T[0,:], self.priPlanet.albedo)
            #     self.secProfile.Qs = reflect.totalScatteredFluxVectorized(self.secVF, self.secProfile.Qs, self.secProfile.T[0,:], self.secPlanet.albedo)
    
                
            # priQs1 = totalScatteredFluxVectorized(self.priVF, self.priProfile.Qs, self.priProfile.T[0,:], priReflectedFlux, self.priPlanet.albedo)
            # secQs1 = totalScatteredFluxVectorized(self.secVF, self.secProfile.Qs, self.secProfile.T[0,:], secReflectedFlux, self.secPlanet.albedo)
               
            # Only do T^4 once since it's a bit slow computationally
            temps4 = np.power(self.profile.T[0,:],4)
            
            # Scattered Flux/reflections 
            # Single body so no interbody 
            self.totalScattered = totalScatteredFluxVectorized2(self.viewFactors, self.profile.Qs, temps4, reflectedFlux, self.planet.albedo)
            #self.totalScattered = scatteredFlux
                  
        # Add scattered flux to Qs
        self.profile.Qs = self.profile.Qs + self.totalScattered

        # Primary
        self.profile.update_T(self.dt, self.profile.Qs, self.planet.Qb)
        self.profile.update_cp()
        self.profile.update_k()
        
        # Increment time
        self.t += self.dt 
        
        
    
    def updateOrbit(self,equilibration = False):
        # New update orbit needs to change orbital values, increase theta and phi 
        orbits.orbitParams(self)
        orbits.updateAngles(self, self.dt, equilibration) # Update theta and phi based on the last time step 
        self.nu += self.nudot * self.dt
        if self.nu >= 2*np.pi:
            self.nu = self.nu - 2*np.pi
        
    def interpolateFluxes(self):
        divisions = len(self.fluxes) 
        hourAngles = np.linspace(0,TWOPI,divisions)
        fluxInterp = interp.interp1d(hourAngles,self.fluxes)
        return fluxInterp
        
    def NormVisCheck(self,tri,ray,local = True):
        # first takes the dot product of solar vector with the facet normal
        # Remove
        # Not necessarily remove but be wary of the negative you introduced to deal with standalone landforms (see ray direction line)
        # Make negative if doing single (due to flip of axes) 
        if local: 
            rayDirection = -np.asarray(ray.d / np.linalg.norm(ray.d))
        else: 
            rayDirection = np.asarray(ray.d / np.linalg.norm(ray.d))
        dotproduct = np.dot(rayDirection,tri.normal)
        i = np.arccos(dotproduct) # solar incidence angle 
        if i <= (np.pi / 2.0) or i >= ((3*np.pi) / 2.0): #if on day side
            # If less than 90 deg, cast ray to see if visible 
            return True, dotproduct # on day side
        return False, dotproduct # on night side
    

  
    def surfModelFlux_Vectorized(self,shadows, profile):
        # Using shadow arrays, calc flux on each facet
        #     given current orbital position 
        
        # Intensity of sun at current orbital distance times dot products. Accounts for albedo
        insolation = self.solarFlux * shadows
        profile.Qs = (1.0 - self.planet.albedo) * insolation 
        # print (shadows[:10])
        # print (profile.Qs[:10])
        self.Qs = profile.Qs
        
        return self.planet.albedo * insolation
        
       
    # def surfModelFlux_Old(self):
    #     #Get hour angle 
    #     h = orbits.hourAngle(self.t,self.planet.day)
    #     #Use hour angle/flux interpolation function to find corresponding flux
    #     self.Qs = self.fluxInterp(h)
        
    
    # def updateFluxInterp(self,normals,solarVect):
    #     # this might only require scaling the flux by the solar power at the distance
    #     # and making a new interpolated function
    #     # rather than taking the dot product with the solar vector all over again
    #     return
            
    def fluxIntensity(self,solarDist,rDist,sVect,normal):
        # Called from FluxCalc
        flux = (solarDist / (rDist**2)) * np.dot(sVect,normal)
        return flux
    
    def findClosestIndexInLookupArrays(self,orientations: np.array):
        # Given angle and the parameters
        #     of a lookup array, find the index of the array that is closest to 
        #     the current orientation
        # ArrayOrientations is an array of the rotational orientations that a 
        #     lookup array covers. It has the same number of indices as the lookup
        #     array has entris 
        index = diff(orientations, np.degrees(self.theta)).argmin()
        
        return index 
    
    def sliceShadows(self,steps):
        # Given the current value of theta, returns the sections of the 
        #     shadowing lookup arrays that most closely corresponds to current 
        #     position (model.theta and model.phi) for primary and secondary 
        
        stepSize = 360. / steps
        orientations = np.arange(0,360,stepSize) # Includes 0, stops before 360
        
        index = self.findClosestIndexInLookupArrays(orientations)
        
        # Slice array
        shadowSlice = self.shadows[index]
        
        return shadowSlice
    
    def sliceViewFactors(self,steps = 70):
        secStep = 360. / steps
        secOrientations = np.arange(steps) * secStep
        
        if self.phi > self.theta:
            separation = self.phi - self.theta + 2*np.pi
        else: 
            separation = self.phi - self.theta
        
        vfIndex = diff(secOrientations, np.degrees(separation)).argmin()
        #return vfIndex 
        return self.binaryVF[vfIndex]
             
    
    
    
    
    
    
    
##############################################################################
## Profile Class
##############################################################################
class profile(object):
    """
    Profiles are objects that contain the model layers
    
    In this implementation, there will be 1 profile with information about
    all facets stored in arrays with at least one dimension the same length
    as the number of facets 
    
    The profile class defines methods for initializing and updating fields
    contained in the model layers, such as temperature and conductivity.
    
    """
    
    #def __init__(self, planet=planets.Moon, facetNum = np.float, lat = np.array, fluxInt = np.array, shadows = np.array,emis = 0.95):
    
    def __init__(self, planet=planets.Moon, facetNum = np.float, lat = np.array,emis = 0.95):

        # planet         
        self.planet = planet
    
        # Number of facets
        self.facetNum = facetNum
                
        # latitude: array?
        self.lat = lat
        
        # # Intensity of sun on facet (related to solar vector, normal of facet) 
        # # array? 
        # self.fluxIntensities = fluxInt
        
        # Array of booleans for whether a facet is shadowed or not 
        #self.shadows = shadows
        
        # emissivity 
        self.emissivity = emis
        

        ######################################################################
        ######################################################################
        
        # to figure out: geometry, lat, long, dec 
        # arrays for facet dependent properties like albedo can go here 
        
        # Initialize surface flux 
        self.Qs = np.zeros(facetNum) #np.float() # surface flux
        
        ######################################################################
        ######################################################################
        
        # The spatial grid
        ks = planet.ks
        kd = planet.kd

        rhos = planet.rhos
        rhod = planet.rhod
        H = planet.H
        cp0 = planet.cp0
        kappa = ks/(rhos*cp0)
        
        #self.z = spatialGrid(skinDepth(planet.day, kappa), m, n, b)
        self.z = spatialGrid(skinDepth(planet.day, kappa), m, n, b,self.facetNum)
        self.nlayers = np.shape(self.z)[0] # number of model layers (grid is same for all facets, so use [0])        
        self.dz = np.diff(self.z[:,0]) # difference along a given axis. Same across all columns (use 1st col)
        self.d3z = self.dz[1:]*self.dz[0:-1]*(self.dz[1:] + self.dz[0:-1])
        self.g1 = 2*self.dz[1:]/self.d3z[0:] # A.K.A. "p" in the Appendix
        self.g2 = 2*self.dz[0:-1]/self.d3z[0:] # A.K.A. "q" in the Appendix
        
        # Thermophysical properties
        self.kc = kd - (kd-ks)*np.exp(-self.z/H)
        self.rho = rhod - (rhod-rhos)*np.exp(-self.z/H)
        
        # Initialize temperature profile
        self.init_T(planet)
        
        # Initialize conductivity profile
        self.update_k()
        
        # Initialize heat capacity profile
        self.update_cp()
        

        

    
    # Temperature initialization
    def init_T(self, planet=planets.Moon, lat = 0):
        self.T = np.zeros([self.nlayers, self.facetNum]) \
                 + T_eq(planet, lat)#self.lat)
    
    
    # Heat capacity initialization
    def update_cp(self):
        self.cp = heatCapacity(self.planet, self.T)
        #self.cp = heatCapacity_ice(self.T)
    
    
    # Thermal conductivity initialization (temperature-dependent)
    def update_k(self):
        self.k = thermCond(self.kc, self.T)
        
        
    # # niu 
    # def rotate_normal(self,angle):
    #     theta = np.radians(angle)
    #     newN = np.asarray(rotate(self.startingNormal,theta))
    #     self.normal = newN
    
    ##########################################################################
    # Core thermal computation                                               #
    # dt -- time step [s]                                                    #
    # Qs -- surface heating rate [W.m-2]                                     #
    # Qb -- bottom heating rate (interior heat flow) [W.m-2]                 #
    ##########################################################################
                         
    def update_T(self, dt, Qs = np.array, Qb = 0):#0, Qb = 0):
        # Coefficients for temperature-derivative terms
        #alpha = self.g1*self.k[0:-2]
        #beta = self.g2*self.k[1:-1]
        #print ("Qs: {}".format(Qs[:20]))
        alpha = np.transpose(self.g1*self.k[0:-2].T)
        beta = np.transpose(self.g2*self.k[1:-1].T)
        # Temperature of first layer is determined by energy balance
        # at the surface
        surfTemp(self, Qs)
        # Temperature of the last layer is determined by the interior
        # heat flux
        botTemp(self, Qb)
        # This is an efficient vectorized form of the temperature
        # formula, which is much faster than a for-loop over the layers
        self.T[1:-1,:] = self.T[1:-1,:] + dt/(self.rho[1:-1,:]*self.cp[1:-1,:]) * \
                     (alpha*self.T[0:-2,:] - \
                       (alpha+beta)*self.T[1:-1,:] + \
                       beta*self.T[2:,:] )
                    
        # # print ("In Update_T")
        # print (Qs[100])            
        # print (self.T[:2,0])
                         

     ##########################################################################   
    
    # Simple plot of temperature profile
    def plot(self):
        ax = plt.axes(xlim=(0,400),ylim=(np.min(self.z),np.max(self.z)))
        plt.plot(self.T, self.z)
        ax.set_ylim(1.0,0)
        plt.xlabel('Temperature, $T$ (K)')
        plt.ylabel('Depth, $z$ (m)')
        mpl.rcParams['font.size'] = 14
    
    # Initialize arrays for temperature, lt 
    def defineReadoutArrays(self,N_steps, N_z,facetNum):
        self.readOutT = np.zeros([N_steps, N_z,facetNum])
        self.readOutlT = np.zeros([N_steps])

#---------------------------------------------------------------------------
"""

The functions defined below are used by the thermal code.

"""
#---------------------------------------------------------------------------

# Thermal skin depth [m]
# P = period (e.g., diurnal, seasonal)
# kappa = thermal diffusivity = k/(rho*cp) [m2.s-1]
def skinDepth(P, kappa):
    return np.sqrt(kappa*P/np.pi)

# # The spatial grid is non-uniform, with layer thickness increasing downward
# def spatialGrid(zs, m, n, b):
#     dz = np.zeros(1) + zs/m # thickness of uppermost model layer
#     z = np.zeros(1) # initialize depth array at zero
#     zmax = zs*b # depth of deepest model layer

#     i = 0
#     while (z[i] < zmax):
#         i += 1
#         h = dz[i-1]*(1+1/n) # geometrically increasing thickness
#         dz = np.append(dz, h) # thickness of layer i
#         z = np.append(z, z[i-1] + dz[i]) # depth of layer i
    
#     return z

# The spatial grid is non-uniform, with layer thickness increasing downward
def spatialGrid(zs, m, n, b,facetNum):
    # Each column represents a new facet 
    dz = np.zeros([1,facetNum]) + zs/m # thickness of uppermost model layer
    z = np.zeros([1,facetNum]) # initialize depth array at zero
    zmax = zs*b # depth of deepest model layer

    i = 0
    while (np.any(z[i,:]) < zmax):
        i += 1
        h = dz[i-1,:]*(1+1/n) # geometrically increasing thickness
        dz = np.append(dz, [h],axis = 0) # thickness of layer i (axis = 0 --> across rows ie columns)
        z = np.append(z, [z[i-1,:] + dz[i,:]],axis = 0) # depth of layer i (axis = 0 --> across rows ie columns)

    return z

# The spatial grid is non-uniform, with layer thickness increasing downward
def spatialGrid(zs, m, n, b, num_facet):
    dz = np.zeros([1, num_facet]) + zs/m # thickness of uppermost model layer
    z = np.zeros([1, num_facet]) # initialize depth array at zero
    zmax = zs*b # depth of deepest model layer

    i = 0
    while (np.any(z[i, :] < zmax)):
        i += 1
        h = dz[i-1, :]*(1+1/n) # geometrically increasing thickness
        dz = np.append(dz, [h], axis=0) # thickness of layer i
        z = np.append(z, [z[i-1, :] + dz[i, :]], axis=0) # depth of layer i
    
    return z

# Solar incidence angle-dependent albedo model
# A0 = albedo at zero solar incidence angle
# a, b = coefficients
# i = solar incidence angle
def albedoVar(A0, a, b, i):
    return A0 + a*(i/(np.pi/4))**3 + b*(i/(np.pi/2))**8

# Radiative equilibrium temperature at local noontime
def T_radeq(planet, lat):
    return ((1-planet.albedo)/(sigma*planet.emissivity) * planet.S * np.cos(lat))**0.25

# Equilibrium mean temperature for rapidly rotating bodies
def T_eq(planet, lat = 0):
    return T_radeq(planet, lat)/np.sqrt(2)

# Heat capacity of regolith (temperature-dependent)
# This polynomial fit is based on data from Ledlow et al. (1992) and
# Hemingway et al. (1981), and is valid for T > ~10 K
# The formula yields *negative* (i.e. non-physical) values for T < 1.3 K
def heatCapacity(planet, T):
    c = planet.cpCoeff
    return np.polyval(c, T)

# Temperature-dependent thermal conductivity
# Based on Mitchell and de Pater (1994) and Vasavada et al. (2012)
def thermCond(kc, T):
    return kc*(1 + R350*T**3)

# Surface temperature calculation using Newton's root-finding method
# p -- profile object
# Qs -- heating rate [W.m-2] (e.g., insolation and infared heating)
    # Array same length as number of facets with Qs for each 
# def surfTemp(p, Qs):
#     Ts = p.T[0]
#     deltaT = Ts
    
#     while (np.abs(deltaT) > DTSURF):
#         x = p.emissivity*sigma*Ts**3
#         y = 0.5*thermCond(p.kc[0], Ts)/p.dz[0]
    
#         # f is the function whose zeros we seek
#         f = x*Ts - Qs - y*(-3*Ts+4*p.T[1]-p.T[2])
#         # fp is the first derivative w.r.t. temperature        
#         fp = 4*x - \
#              3*p.kc[0]*R350*Ts**2 * \
#                 0.5*(4*p.T[1]-3*Ts-p.T[2])/p.dz[0] + 3*y
        
#         # Estimate of the temperature increment
#         deltaT = -f/fp
#         Ts += deltaT
#     # Update surface temperature
#     p.T[0] = Ts
    
def surfTemp(p, Qs):
    Ts = p.T[0,:]
    deltaT = Ts
    
    while (np.any(np.abs(deltaT) > DTSURF)):
        x = p.emissivity*sigma*Ts**3
        y = 0.5*thermCond(p.kc[0,:], Ts)/p.dz[0]
    
        # f is the function whose zeros we seek
        f = x*Ts - Qs - y*(-3*Ts+4*p.T[1,:]-p.T[2,:])
        # fp is the first derivative w.r.t. temperature        
        fp = 4*x - \
             3*p.kc[0,:]*R350*Ts**2 * \
                0.5*(4*p.T[1,:]-3*Ts-p.T[2,:])/p.dz[0] + 3*y
        
        # Estimate of the temperature increment
        deltaT = -f/fp
        Ts += deltaT
    # Update surface temperature
    p.T[0,:] = Ts

# Bottom layer temperature is calculated from the interior heat
# flux and the temperature of the layer above
def botTemp(p, Qb):
    #p.T[-1] = p.T[-2] + (Qb/p.k[-2])*p.dz[-1]
    p.T[-1,:] = p.T[-2,:] + (Qb / p.k[-2,:])*p.dz[-1]

def getTimeStep(p, day):
    dt_min = np.min( F * p.rho[:-1,0] * p.cp[:-1,0] * p.dz**2 / p.k[:-1,0] )
    return dt_min

# Adjusted for highest temperature conductivity 
def getHighTempTimeStep(p, highTk, day):
    dt_min = np.min( F * p.rho[:-1,0] * p.cp[:-1,0] * p.dz**2 / highTk[:-1,0] )
    return dt_min

 
# Used to determine closest index for lookup data
# Takes into account the fact the circular nature of degrees
#    ie. 359 is closer to 0 than to 336 
def diff(a, b, turn=360): 
    return np.minimum((np.remainder(a - b,turn)),np.remainder(b-a,turn))

# Returns the max theoretical temperature at the equator during perihelion
# Used to determine a stable timestep 
def getPerihelionSolarIntensity(planet):
    x = planet.rAU   * (1 - planet.eccentricity**2)
    peDist = x / (1 + planet.eccentricity * np.cos(0)) # Cos 0 corresponds to true anomaly at perihelion
    peIntensity = solarLum / (peDist*1.496e11)**2 #[W.m-2] Intensity of sun at perihelion. Convert AU to m
    peMaxTemp = ((1-planet.albedo)/(sigma*planet.emissivity) * peIntensity * np.cos(0))**0.25 # Max temperature, at equator, at perihelion 
    return peMaxTemp 



#@jit(nopython = True)
def totalScatteredFluxVectorized(viewFactors: np.array, fluxes: np.array, temps: np.array,reflectedFlux: np.array,albedo, emis = 0.95, readOut = False):
    #emis2sigma = np.float64(emis**2 * sigma)
    # if readOut:
    #     print (viewFactors.dtype)
    #     print (temps.dtype)
    #     print (reflectedFlux.dtype)
    #     print (isinstance( emis2sigma, np.float64 ))
        
    
    # Amount of flux reflected from each facet (available to other facets as reflection)
    visReflected = reflectedFlux#np.multiply(albedo,fluxes)
    
    q_Vis = np.asarray((1 - albedo) * np.multiply(viewFactors,visReflected[:,np.newaxis])) # Need to take into account how much is absorbed into recieving facet

    # Infrared emission from other facets (absorbed)
    q_IR = emis**2 * sigma * np.multiply(viewFactors,np.power(temps,4))
    
    # Add solar, visible reflected and absorbed scattered infrared light         
    # Axis = 0 sums columns 
    visSum = np.sum(q_Vis, axis=0)
    irSum = np.sum(q_IR, axis = 0)
    #fluxTotal = fluxes + np.sum(visSum) + np.sum(irSum)
    fluxTotal = fluxes + visSum + irSum
    #fluxTotal = fluxes + visSum
    

    return fluxTotal

@jit(nopython = True)#, nogil = True)#,parallel = True, nogil = True)
def totalScatteredFluxVectorized2(viewFactors: np.array, fluxes: np.array, temps: np.array, reflectedFlux: np.array,albedo, emis = 0.95):    
    # temps4 = np.power(temps,4)
    visReflected = reflectedFlux#np.multiply(albedo,fluxes)
    facets = np.shape(fluxes)[0]
    fluxTotals = np.zeros(facets)
    for i in range(np.shape(fluxes)[0]):
        Vis = np.asarray((1-albedo) * np.multiply(viewFactors[:,i],visReflected))
        # NOTE!! If not feeding in T^4 already need to uncomment temps4 above and submit temps4 instead of temps below
        IR = emis**2 * sigma * np.multiply(viewFactors[:,i],temps) 
        visSum = np.sum(Vis)
        irSum = np.sum(IR)
        
        # # Output total
        # fluxTotal = fluxes[i] + visSum + irSum
        # fluxTotals[i] = fluxTotal
        
        # Output only additional scattered/reflected flux
        fluxTotals[i] = visSum + irSum

    return fluxTotals

@jit(nopython = True)
def interBodyScatteredFlux(binaryViewFactors: np.array, fluxes: np.array, temps: np.array, reflectedFlux: np.array, body, facetNum, albedo, emis = 0.95):
    # These will use the flux/temps of the opposite body (ie primary will use secondary values to calculate scattering)
    fluxTotals = np.zeros(facetNum) # Storage array. Size is number of facets of body you're calculating it for 
    #temps4 = np.power(temps,4)
    for i in range(facetNum):
        if body == "primary":
            Vis = np.asarray((1-albedo) * np.multiply(binaryViewFactors[i,:],reflectedFlux)) # Use rows for primary
            # NOTE!! If not feeding in T^4 already need to uncomment temps4 above and submit temps4 instead of temps below
            IR = emis**2 * sigma * np.multiply(binaryViewFactors[i,:],temps)
        elif body == "secondary":
            Vis = np.asarray((1-albedo) * np.multiply(binaryViewFactors[:,i],reflectedFlux)) # Use rows for primary
            # NOTE!! If not feeding in T^4 already need to uncomment temps4 above and submit temps4 instead of temps below
            IR = emis**2 * sigma * np.multiply(binaryViewFactors[:,i],temps)
        else: print ("Neither primary or secondary selected for interbody scattering")
        visSum = np.sum(Vis)
        irSum = np.sum(IR)
        
        # # Output total
        # fluxTotal = fluxes[i] + visSum + irSum
        # fluxTotals[i] = fluxTotal
        
        # Output only additional scattered/reflected flux
        fluxTotals[i] = visSum + irSum
        
    return fluxTotals

@jit(nopython = True)
def singleScatteredMP(viewFactors: np.array, fluxes: np.array, temps: np.array, reflectedFlux: np.array,albedo):
    emis = 0.95
    Vis = np.asarray((1-albedo) * np.multiply(viewFactors,reflectedFlux))
    # NOTE!! If not feeding in T^4 already need to uncomment temps4 above and submit temps4 instead of temps below
    IR = emis**2 * sigma * np.multiply(viewFactors,temps) 
    visSum = np.sum(Vis)
    irSum = np.sum(IR)
            
    return visSum + irSum

@jit(nopython = True)
def binaryScatteredMP(binaryViewFactors: np.array, fluxes: np.array, temps: np.array, reflectedFlux: np.array, albedo):
    emis = 0.95
    Vis = np.asarray((1-albedo) * np.multiply(binaryViewFactors,reflectedFlux)) # Use rows for primary
    # NOTE!! If not feeding in T^4 already need to uncomment temps4 above and submit temps4 instead of temps below
    IR = emis**2 * sigma * np.multiply(binaryViewFactors,temps)
    visSum = np.sum(Vis)
    irSum = np.sum(IR)
    return visSum + irSum


@jit(nopython = True)
def viewFactorsProcess(index, viewFactors: np.array,binaryViewFactors: np.array, secFluxes: np.array, priFluxes: np.array, \
                       secTemps: np.array, priTemps: np.array, secReflectedFlux: np.array, priReflectedFlux: np.array, priFacetNum, priAlbedo, secAlbedo):
    # Each process will represent one facet so i will be priFacetNum + secFacetNum
    if index < priFacetNum: 
        i = index
        # Individual body
        # Each of these processes will only get pertinent view factors 
        single = singleScatteredMP(viewFactors, priFluxes, priTemps, priReflectedFlux, priAlbedo)
        # Binary
        binary = binaryScatteredMP(binaryViewFactors[i,:], secFluxes, secTemps, secReflectedFlux, priAlbedo)
    else: 
        i = index - priFacetNum
        single = singleScatteredMP(viewFactors, secFluxes, secTemps, secReflectedFlux, secAlbedo)
        binary = binaryScatteredMP(binaryViewFactors[:,i], priFluxes, priTemps, priReflectedFlux, secAlbedo)
        
    return single + binary 









# # Models contain the profiles and model results
# class binaryModel(object):
#     def __init__(self, priShapeModelPath, secShapeModelPath, priPlanet = planets.Moon,secPlanet = planets.Moon, ndays=1,nyears = 1.0,local = False,priShadowLookupPath = None, secShadowLookupPath = None,priVF = None,secVF = None,binaryVF = None):
    
#         # Change Equilibration Time
#         NYEARSEQ = nyears   
#         print ("Equilibration time: {} years".format(NYEARSEQ))
    
#         # Initialize
#         self.priPlanet = priPlanet
#         self.secPlanet = secPlanet
#         #self.planet = priPlanet # For bulk orbital values 
#         #self.lat = lat
#         self.Sabs = self.priPlanet.S * (1.0 - self.priPlanet.albedo)
#         self.r = self.priPlanet.rAU # solar distance [AU]
#         self.nu = np.float() # orbital true anomaly [rad]
#         self.nudot = np.float() # rate of change of true anomaly [rad/s]
#         self.dec = np.float() # solar declination [rad]
#         self.solarFlux = np.float() # Solar flux (W/m^2) at the body given the distance to the sun
#         #self.sPos = np.array([0,0,0])
       
#         # Read in Shape Model & initialize corresponding values
#         self.local = local
#         self.priShape = shape.shapeModel(priShapeModelPath,local)
#         self.secShape = shape.shapeModel(priShapeModelPath,local)
#         self.priFacetNum = self.priShape.facetNum
#         self.secFacetNum = self.priShape.facetNum
#         self.facetNum = self.priFacetNum + self.secFacetNum
#         # List of ints from 0 to the number of facets. Used to keep track of which facet you're working on
#         iVals = np.arange(0,self.facetNum)
#         print ("Shape Model Loaded")
        
#         # Initialize arrays for latitudes, fluxes, shadows 
#         #self.lats = self.shape.lats#np.zeros(self.facetNum) # np array of latitudes for each facet
#         self.allLats = np.concatenate([self.priShape.lats,self.secShape.lats]) # np array of latitudes for each facet for both bodies
#         # Rotation of the primary about own axis 
#         # Used for shadowing and view factor setup  
#         self.theta = np.float() # Radians, starts at 0.
        
#         # Rotation of the secondary about primary
#         # Used for shadowing setup 
#         self.phi = np.float() # Radians, starts at 0
        
#         # Load shadow arrays 
#         self.secShadows = np.load(secShadowLookupPath)
#         self.priShadows = np.load(priShadowLookupPath)
#         print ("Shadow arrays loaded")
        
        
#         # Initialize profile(s)
#         # Remove
#         #self.profile = profile(planet = self.planet,facetNum = self.facetNum,lat = self.lats,fluxInt = self.QsVals,shadows = self.shadows) 
#         #self.profile = profile(planet = self.secPlanet,facetNum = self.facetNum,lat = self.allLats)
        
#         # Two profiles
#         self.priProfile = profile(planet = self.priPlanet, facetNum = self.priFacetNum, lat = self.priShape.lats)
#         self.secProfile = profile(planet = self.secPlanet, facetNum = self.secFacetNum, lat = self.secShape.lats)
        
#         # # Attempting to shorten the aphelion run by feeding in temps from another run. 
#         # if startPriT != None and startSecT != None:
#         #     priStartT = np.load(startPriT)[0]
#         #     secStartT = np.load(startSecT)[0]
            
#         #     self.priProfile.T = priStartT
#         #     self.secProfile.T = secStartT
    
    
#         # Model run times
#         # # Equilibration time -- TODO: change to convergence check
#         # self.equiltime = NYEARSEQ*planet.year - \
#         #                 (NYEARSEQ*planet.year)%planet.day
                        
#         # Binary model run times.
#         # Equilibration time 
#         # **Most of this will be done on basis of secondary day. Primary will spin multiple times during this period 
#         self.equiltime = NYEARSEQ * secPlanet.year - (NYEARSEQ*secPlanet.year)%secPlanet.day
#         # Run time for output
#         # # Singl body 
#         # self.endtime = self.equiltime + ndays*planet.day
#         # **binary
#         self.endtime = self.equiltime + ndays*secPlanet.day
#         self.t = 0.
        
#         # Find min of two timesteps for primary and secondary profiles
#         priTimeStep = getTimeStep(self.priProfile, self.priPlanet.day)
#         secTimeStep = getTimeStep(self.secProfile, self.secPlanet.day)
#         self.dt = min([priTimeStep,secTimeStep])

#         #self.dt = getTimeStep(self.profile, self.secPlanet.day)  
#         # Check for maximum time step
#         self.dtout = self.dt
        
#         print ("timesteps to equiltime: {}".format(self.equiltime / self.dt))
#         # # Single
#         # dtmax = self.planet.day/NPERDAY
#         # **Binary 
#         dtmax = secPlanet.day/NPERDAY
#         if self.dt > dtmax:
#             self.dtout = dtmax
        
#         # Array for output temperatures and local times
#         # # single
#         # N_steps = np.int((ndays*planet.day)/self.dtout )
#         # **binary 
#             # Done on the basis of secondary day in length but using primary's timestep...will be huge readout arrays 
#         N_steps = np.int((ndays*secPlanet.day) / self.dtout)
#         self.N_steps = N_steps
        
#         # Old: 1 profile
#         # N_z = np.shape(self.profile.z)[0]
#         # self.N_z = N_z
        
#         # New: binary profile
#         if np.shape(self.priProfile.z)[0] != np.shape(self.secProfile.z)[0]:
#             print ("Depth layers for two bodies not equivalent. Smaller value recommended")
#         N_z = np.shape(self.priProfile.z)[0]
#         self.N_z = N_z
        
#         # Temperature and local time arrays 
#         #self.T = np.zeros([N_steps, N_z,self.facetNum])
#         #***
#         self.priT = np.zeros([N_steps, N_z,self.priFacetNum])
#         self.secT = np.zeros([N_steps, N_z,self.secFacetNum])
#         self.lt = np.zeros((N_steps,3)) #[N_steps, 3]) # Time, theta, phi
#         self.priReadoutScat = np.zeros([N_steps, self.priFacetNum])
#         self.secReadoutScat = np.zeros([N_steps, self.secFacetNum])
        
#         # ***
#         # ***
#         self.PriOutIndex = np.zeros((N_steps,2))
#         self.PriOutFlux = np.zeros((N_steps,self.priFacetNum))

#         # Resolution for view factors and shadowing 
#         # count how many facets around equator (ie have lat of ~0?) 
#         # This needs to be split into two for a binary 
#         if priShadowLookupPath == None or secShadowLookupPath == None:
#             print ("Calculating facet resolution around primary equator")
#             lats = np.zeros(self.priFacetNum)
#             longs = np.zeros(self.priFacetNum)
#             equatorial_facets = 0
#             for tri in self.priShape.tris:
#                 lats[tri.num] = tri.lat
#                 longs[tri.num] = tri.long
#                 if tri.lat < 5 and tri.lat >= 0:
#                     equatorial_facets += 1
                    
            
#             # Find how frequently you need to update shadowing and view factors 
#             self.deg_per_facet = 360 / equatorial_facets # degrees per facet
#             print ("Equatorial facets: "+str(equatorial_facets))
#             print ("Degrees per facet: "+str(self.deg_per_facet))
            
#         else: 
#             print ("Shadowing Lookup Table provided. Skipping resolution check")
        
        
        
#         # View Factors
#         #*** Currently just for individual bodies
#         if priVF != None and secVF != None:
#             # self.priVF = np.transpose(np.load(priVF)) # Primary
#             # self.secVF = np.transpose(np.load(secVF)) # Secondary 
#             self.priVF = np.load(priVF) # Primary
#             self.secVF = np.load(secVF) # Secondary 
#             self.IncludeReflections = True
#             self.bothVF = np.concatenate((self.priVF,self.secVF), axis = 0)
            
#         else: 
#             print ("WARNING: AUTOMATIC VIEW FACTORS NOT ENABLED! \n Proceeding with null view factors")
#             self.IncludeReflections = False
            
#         # Binary View factors: inter body
#         if binaryVF != None:
#             self.binaryVF = np.load(binaryVF)
#         else: print ("NO BINARY VIEW FACTORS PROVIDED")
            
        
#         # # View factors 
#         # if vfFile != None: # If you're reading in a precalculated set of view factors 
#         #     viewFactors = np.genfromtxt(vfFile)
#         #     viewFactors = np.transpose(viewFactors)
#         #     #viewFactors = np.zeros(np.shape(viewFactors)) # Remove. This is for debugging shadowing 
#         # else:
#         #     # Calc view factors with multiprocessing 
#         #     # Initialize view factor array 
#         #     print ("ERROR ERROR ERROR")
#         #     print ("Custom view factors not enabled yet. Need to feed in pre calculated view factors")

#         #     viewFactors = np.zeros((self.facetNum,self.facetNum))
#         #     # viewFactorRun = p.starmap(self.viewFactorsHelper,zip(iVals,repeat(viewFactorStorage),repeat(local)))
            
#         #     # for i in iVals:
#         #     #     viewFactors[:,i] = viewFactorStorage[i][1]
                
#         #     # for i in iVals:
#         #     #     viewFactors[i,i:self.facetNum] = viewFactorStorage[i][0][i:self.facetNum]
                
#         #     # np.savetxt(vfFile,viewFactors)
#         # print ("View factors saved")
        
#             # Test case: 4D (aka not including non tidally locked secondary)
#             # theta of 0, pi/2, pi. Would eventaully have int(equatorial_facets) or ~120
#             # phi of 0, pi/4, pi/2. Would eventually have int(equatorial_facets) or ~120
#             # Run with each combination 
            
#         # steps = int(equatorial_facets)
        
                
#         # Sun location initialization 
#         self.sLoc = np.asarray([0,0,1.496e11]) # 1 AU # make this self.sLoc
#         self.newSLoc = self.sLoc
#         self.baseDist = np.float()
        
#         print ("Completed Visibility Setup")
        
        
        
#     def run(self,calcShadows = False):
        
#         #mkl.set_num_threads(4)
#         # Initialize pool 
#         # pool = multiprocessing.Pool(multiprocessing.cpu_count())
#         # print ("pool started")

#         # print ("Res size: {}".format(np.shape(res)))
#         # print ("Entered run")
#         # # Precompute view factors 
#         # # ** Need to put view factors in 
        
#         # # Precompute shadows 
#         # if calcShadows: # Shadows have not been precalculated 
#         #     print ("Shadows Needed. Initiating shadow calculations")
        
#         # Might need to update shadows in a regular, but infrequent basis to make sure you're accomodating siderial vs synodic days 
#         priSteps = np.shape(self.priShadows)[0]
#         secSteps = np.shape(self.priShadows)[1]
#         vfSteps = np.shape(self.binaryVF)[0]
#         # Equilibrate 
#         print ("-------------Beginning Equilibration---------------")
#         i = 0
#         vfUpdate = True
#         vfCount = 0
#         while self.t < self.equiltime:

#             # Update orbit & binary angles 
#             self.updateOrbit(equilibration = True)
            
#             # Get slice of shadow and view factor arrays that correspond to current position 
#             priShadowVals, secShadowVals = self.sliceShadows(priSteps,secSteps) # Slice the sections of lookup arrays you need 
            
#             # Slice view factors for current position 
#             #***
#             binaryVFSlice = self.sliceViewFactors(vfSteps)
#             # Update solar location (used for sideal vs synotic)
#             # Currently sun isnt moving 
            
#             # If enough time has passed that you need to do a sidereal vs synotic update do that 
            

#             # Updating every 5 
#             if vfCount == 0:
#                 vfUpdate = True
#             else: vfUpdate = False
#             #self.advance(priShadowVals, secShadowVals,binaryVFSlice,pool,vfUpdate)
#             self.advance(priShadowVals, secShadowVals,binaryVFSlice,vfUpdate)
#             #vfUpdate = not vfUpdate 
#             i += 1
#             vfCount += 1
#             # if i > num:
#             #     print ("{} interations completed. Break".format(num))
#             #     break
        
#             if i%1000 == 0:
#                 print (i)
            
#             if vfCount > 5:
#                 vfCount = 0
            
#         print ("Equilibration reached. Saving temperatures")
#         # Run through end of model and store output
#         self.dt = self.dtout
#         self.t = 0.0  # reset simulation time
        

#         for i in range(0, self.N_steps):
#             self.updateOrbit()
#             priShadowVals, secShadowVals = self.sliceShadows(priSteps,secSteps)
#             binaryVFSlice = self.sliceViewFactors(vfSteps)
#             #*** Need View factor slice 

#             self.advance(priShadowVals, secShadowVals,binaryVFSlice,vfUpdate)
            
#             # self.priReadoutScat[i] = self.savePrimaryScat
#             # self.secReadoutScat[i] = self.secTotalScattered
#             self.priReadoutScat[i], self.secReadoutScat[i] = self.savePrimaryScat,self.secTotalScattered
            
#             # *** Output T arrays with two profiles 
#             #self.T[i, :,:] = self.profile.T  # temperature [K]
#             self.priT[i,:,:] = self.priProfile.T # primary temperature [K]
#             self.secT[i,:,:] = self.secProfile.T # secondary temperature [K]
#             self.lt[i] = np.asarray([self.t / self.secPlanet.day * 24.0, self.theta, self.phi])  # local time [hr], theta (rad), phi (rad) 
            
#             #***
#             self.PriOutFlux[i] = self.priQs
#             self.PriOutIndex[i] = self.indices
            
#             vfUpdate = not vfUpdate # Do view factors every 2 
#             # if i > 10:
#             #     break

#     #@jit(nopython=True)    
#     def advance(self,priShadowVals, secShadowVals,  binaryVFSlice, vfUpdate = True):
#         # Non multiprocessed version of advance that uses one profile object and adds a dimension to each array 
        
#         # Assuming geometry is handled separately 
#         #   Aka view factors, shadowing 
        
#         # Finding Qs for each facet 
#         #    Needs to be vectorized
#         #    Feed in boolean array (or 0s and 1s) for shadowing state of each facet 
#         #self.surfModelFlux(self.shadows)
#         priReflectedFlux, secReflectedFlux = self.surfModelFlux_Vectorized_Binary(priShadowVals,secShadowVals,self.priProfile,self.secProfile)
        
#         # Reflections/scattering 
#         # Output array of Qs values including scattering and reflection 
#         # If a view factor matrix was provided and 
#         # if self.IncludeReflections:
#         #     self.priProfile.Qs = reflect.totalScatteredFluxVectorized(self.priVF, self.priProfile.Qs, self.priProfile.T[0,:], self.priPlanet.albedo)
#         #     self.secProfile.Qs = reflect.totalScatteredFluxVectorized(self.secVF, self.secProfile.Qs, self.secProfile.T[0,:], self.secPlanet.albedo)

            
#         # priQs1 = totalScatteredFluxVectorized(self.priVF, self.priProfile.Qs, self.priProfile.T[0,:], priReflectedFlux, self.priPlanet.albedo)
#         # secQs1 = totalScatteredFluxVectorized(self.secVF, self.secProfile.Qs, self.secProfile.T[0,:], secReflectedFlux, self.secPlanet.albedo)
#         if vfUpdate:
#             # Only do T^4 once since it's a bit slow computationally
#             priTemps4 = np.power(self.priProfile.T[0,:],4)
#             secTemps4 = np.power(self.secProfile.T[0,:],4)
            
#             # # Do visibility calcs 
#             # arg1 = np.arange(self.facetNum)
#             # arg2 = self.bothVF
#             # arg3 = binaryVFSlice
#             # arg4 = self.secProfile.Qs
#             # arg5 = self.priProfile.Qs
#             # arg6 = secTemps4
#             # arg7 = priTemps4
#             # arg8 = secReflectedFlux
#             # arg9 = priReflectedFlux
#             # arg10 = self.priFacetNum
#             # arg11 = self.priPlanet.albedo
#             # arg12 = self.secPlanet.albedo
#             # result = pool.starmap_async(viewFactorsProcess,zip(arg1, arg2,repeat(arg3),repeat(arg4),repeat(arg5), repeat(arg6), repeat(arg7),repeat(arg8), repeat(arg9), repeat(arg10), repeat(arg11),repeat(arg12)))
#             # output = result.get()#np.array([m.get() for m in result])
            
            
#             # pool = mp.Pool(processes=pro_num_VF)
#             # results = [pool.apply_async(viewFactors_3D_5, args=(self.tri_vert[:,:,:], self.tri_cent[i,:],\
#             #                              self.tri_area[:], self.tri_norm[:,:], self.tri_rays[i,:,:], i)) for i in pro_idx_VF]
            
            
            
#             # self.priTotalScattered = res[:self.priFacetNum]
#             # self.secTotalScattered = res[self.secFacetNum:]
            
#             # primaryScatteredFlux = totalScatteredFluxVectorized2(self.priVF, self.priProfile.Qs, self.priProfile.T[0,:], priReflectedFlux, self.priPlanet.albedo)
#             # secondaryScatteredFlux = totalScatteredFluxVectorized2(self.secVF, self.secProfile.Qs, self.secProfile.T[0,:], secReflectedFlux, self.secPlanet.albedo)
            
#             # primaryBinaryScatteredFlux = interBodyScatteredFlux(binaryVFSlice, self.secProfile.Qs, self.secProfile.T[0,:], secReflectedFlux, "primary", self.priProfile.facetNum, self.priPlanet.albedo)
#             # secondaryBinaryScatteredFlux = interBodyScatteredFlux(binaryVFSlice, self.priProfile.Qs, self.priProfile.T[0,:], priReflectedFlux, "secondary", self.secProfile.facetNum, self.secPlanet.albedo)
            
#             primaryScatteredFlux = totalScatteredFluxVectorized2(self.priVF, self.priProfile.Qs, priTemps4, priReflectedFlux, self.priPlanet.albedo)
#             #secondaryScatteredFlux = totalScatteredFluxVectorized2(self.secVF, self.secProfile.Qs, secTemps4, secReflectedFlux, self.secPlanet.albedo)
            
#             primaryBinaryScatteredFlux = interBodyScatteredFlux(binaryVFSlice, self.secProfile.Qs, secTemps4, secReflectedFlux, "primary", self.priProfile.facetNum, self.priPlanet.albedo)
#             secondaryBinaryScatteredFlux = interBodyScatteredFlux(binaryVFSlice, self.priProfile.Qs, priTemps4, priReflectedFlux, "secondary", self.secProfile.facetNum, self.secPlanet.albedo)
            
#             self.priTotalScattered = primaryScatteredFlux + primaryBinaryScatteredFlux
#             self.secTotalScattered = secondaryBinaryScatteredFlux#secondaryScatteredFlux + secondaryBinaryScatteredFlux
            
#             self.savePrimaryScat = primaryBinaryScatteredFlux
                  
#         # Add scattered flux (if not doing view factor calculations each time, this will allow for last time step's values to be maintained)
#         self.priProfile.Qs = self.priProfile.Qs + self.priTotalScattered
#         self.secProfile.Qs = self.secProfile.Qs + self.secTotalScattered
        
#         # Primary
#         self.priProfile.update_T(self.dt, self.priProfile.Qs, self.priPlanet.Qb)
#         self.priProfile.update_cp()
#         self.priProfile.update_k()
        
#         # Secondary
#         self.secProfile.update_T(self.dt, self.secProfile.Qs, self.secPlanet.Qb)
#         self.secProfile.update_cp()
#         self.secProfile.update_k()
        
#         self.t += self.dt # increment time 
        
        
    
#     def updateOrbit(self,equilibration = False):
#         # New update orbit needs to change orbital values, increase theta and phi 
#         orbits.orbitParams(self)
#         orbits.updateBinaryAngles(self, self.dt,equilibration) # Update theta and phi based on the last time step 
#         self.nu += self.nudot * self.dt
        
#     def interpolateFluxes(self):
#         divisions = len(self.fluxes) 
#         hourAngles = np.linspace(0,TWOPI,divisions)
#         fluxInterp = interp.interp1d(hourAngles,self.fluxes)
#         return fluxInterp
        
#     def NormVisCheck(self,tri,ray,local = True):
#         # first takes the dot product of solar vector with the facet normal
#         # Remove
#         # Not necessarily remove but be wary of the negative you introduced to deal with standalone landforms (see ray direction line)
#         # Make negative if doing single (due to flip of axes) 
#         if local: 
#             rayDirection = -np.asarray(ray.d / np.linalg.norm(ray.d))
#         else: 
#             rayDirection = np.asarray(ray.d / np.linalg.norm(ray.d))
#         dotproduct = np.dot(rayDirection,tri.normal)
#         i = np.arccos(dotproduct) # solar incidence angle 
#         if i <= (np.pi / 2.0) or i >= ((3*np.pi) / 2.0): #if on day side
#             # If less than 90 deg, cast ray to see if visible 
#             return True, dotproduct # on day side
#         return False, dotproduct # on night side
    

#     # def ShadowCheck(self,tris,ray,expTri,baseDist):
#     #     # Checks through all triangles to find the closest intersect. Least computationally effective method 
#     #     #distExp = find_distance(tri = expTri,ray = ray)
#     #     hitcount = 0
#     #     closest_tri = expTri
        
#     #     for tri in tris:
#     #         result = tri.intersect_simple(ray)
#     #         if result == 1: #tri.intersect_simple(ray) == 1:
#     #             hitcount += 1
#     #             if tri.num == expTri.num: # If origin triangle
#     #                 continue
#     #             dist = kdtree_3d.find_distance(ray=ray,tri = tri)
#     #             if dist < baseDist:
#     #                 # closer intersect found. Light is stopped before reaching facet
#     #                 # shadowed
#     #                 closest_tri = tri
#     #                 print ("Shadowed")
#     #                 return True
#     #             elif dist == baseDist:
#     #                 if tri != expTri:
#     #                     print ("diff tri, same distance")
#     #                     continue
#     #             else:
#     #                 # dist is either equal or greater than dist to tri in question. Skip to next
#     #                 # print ("Farther away")
#     #                 #print ("Tri "+str(tri.num)+" is Farther away than expected No. "+str(expTri.num))
#     #                 continue
#     #         if result == 2:
#     #             print ("Something's gone wrong: Ray starting point is in triangle")

#     #     return False    # Light makes it to expected facet, no shadow 
    


            
#     # def surfModelFlux(self, profile,ray, index, theta,baseDist):
#     #     tri = self.shape.tris[profile.num]
#     #     vis, dotproduct = self.NormVisCheck(tri,ray)
        
#     #     if vis:
#     #         shadow = self.ShadowCheck(self.shape.tris, ray, tri,baseDist) #allTrisJoined
#     #         # Remove
#     #         # if profile.shadow == None:
#     #         #     shadow = self.ShadowCheck(self.shape.tris,ray,tri, profile.baseDist)
#     #         #     profile.shadow = shadow
#     #         # else:
#     #         #     shadow = profile.shadow
#     #         #inLight = kdtree_3d.Shadow_Trace(profile.tri,sVect)
#     #         if shadow == True: 
#     #             profile.Qs = 0.
#     #             profile.Qs_reflected = 0.
#     #             intensity = 0.
#     #             #print ("Facet in shadow: "+str(tri.num))
#     #         if shadow == False:
#     #             intensity = self.solarFlux * dotproduct
#     #             # a = self.planet.albedoCoef[0]
#     #             # b = self.planet.albedoCoef[1]
#     #             # f = (1.0 - albedoVar(self.planet.albedo, a, b, theta))/(1.0 - self.planet.albedo)
#     #             #profile.Qs = f * intensity * (1.0-self.planet.albedo)
#     #             profile.Qs = intensity * (1.0-self.planet.albedo)
#     #             profile.Qs_reflected = intensity * self.planet.albedo
                
#     #     else: 
#     #         # If night side (angle between > 90 deg) in shadow and flux is 0
#     #         profile.Qs, profile.Qs_reflected = 0.,0.
#     #         #fluxDict[n] = 0
#     #         #fluxMaster.get(n).append(0)

#     # #@jit (nopython = True)
#     # def surfModelFlux_Vectorized(self,priShadows, secShadows, profile):
#     #     # Using shadow and view factor (eventually) arrays, calc flux on each facet
#     #     #     given current orbital position 
#     #     # Single profile

#     #     # NOTE!!!!If shadow arrays have fluxes in W/m^2, use next two lines
#     #     #allShadows = np.concatenate([priShadows,secShadows])
#     #     #profile.Qs = (1.0 - self.priPlanet.albedo) * allShadows
        
#     #     # NOTE!!! If shadow arrays have dot products, use next 3 lines 
#     #     allShadows = np.concatenate([priShadows,secShadows])
#     #     intensity = self.solarFlux * allShadows # Intensity of sun at current orbital distance times dot products 
#     #     profile.Qs = (1.0 - self.priPlanet.albedo) * intensity  # Account fo albedo 
        
#     def surfModelFlux_Vectorized_Binary(self,priShadows, secShadows, priProfile, secProfile):
#         # Using shadow and view factor (eventually) arrays, calc flux on each facet
#         #     given current orbital position 
#         # Explicitly requires two profiles 
        
#         # Two Profiles 
#         # Intensity of sun at current orbital distance times dot products. Accounts for albedo
#         priInsolation = self.solarFlux * priShadows
#         secInsolation = self.solarFlux * secShadows
#         priProfile.Qs = (1.0 - self.priPlanet.albedo) * priInsolation 
#         secProfile.Qs = (1.0 - self.secPlanet.albedo) * secInsolation
        
#         # ***
#         self.priQs = priProfile.Qs
        
#         return self.priPlanet.albedo * priInsolation, self.secPlanet.albedo * secInsolation
        
       
#     # def surfModelFlux_Old(self):
#     #     #Get hour angle 
#     #     h = orbits.hourAngle(self.t,self.planet.day)
#     #     #Use hour angle/flux interpolation function to find corresponding flux
#     #     self.Qs = self.fluxInterp(h)
        
    
#     # def updateFluxInterp(self,normals,solarVect):
#     #     # this might only require scaling the flux by the solar power at the distance
#     #     # and making a new interpolated function
#     #     # rather than taking the dot product with the solar vector all over again
#     #     return
            
#     def fluxIntensity(self,solarDist,rDist,sVect,normal):
#         # Called from FluxCalc
#         flux = (solarDist / (rDist**2)) * np.dot(sVect,normal)
#         return flux
    
#     def findClosestIndexInLookupArrays(self,priOrientations: np.array, secOrientations: np.array):
#         # Given two angles (primary theta and secondary phi) and the parameters
#         #     of a lookup array, find the index of the array that is closest to 
#         #     the current orientation
#         # ArrayOrientations is an array of the rotational orientations that a 
#         #     lookup array covers. It has the same number of indices as the lookup
#         #     array has entris 
#         priIndex = diff(priOrientations, np.degrees(self.theta)).argmin()
#         secIndex = diff(secOrientations, np.degrees(self.phi)).argmin()
        
#         return priIndex, secIndex 
    
#     def sliceShadows(self,priSteps,secSteps):
#         # Given the current values of theta and phi, returns the sections of the 
#         #     shadowing lookup arrays that most closely corresponds to current 
#         #     position (model.theta and model.phi) for primary and secondary 
#         if priSteps != secSteps: print ("Primary and secondary shadowing lookup arrays have different number of orientations")
        
#         priStepSize = 360. / priSteps
#         secStepSize = 360. / secSteps
#         priOrientations = np.arange(0,360,priStepSize) # Includes 0, stops before 360
#         secOrientations = np.arange(0,360,secStepSize)
        
#         priIndex, secIndex = self.findClosestIndexInLookupArrays(priOrientations,secOrientations)
#         # ***
#         self.indices = np.asarray([priIndex,secIndex])

#         # print ("Indices")
#         # print (priIndex, secIndex)
#         # Select arrays
#         priSlice = self.priShadows[priIndex][secIndex]
#         secSlice = self.secShadows[priIndex][secIndex]
        
#         # print ("Slice: {}".format(priSlice[:10]))
#         # print ("Slice size: {}".format(np.shape(priSlice)))
#         return priSlice, secSlice
    
#     def sliceViewFactors(self,steps = 70):
#         secStep = 360. / steps
#         secOrientations = np.arange(steps) * secStep
        
#         if self.phi > self.theta:
#             separation = self.phi - self.theta + 2*np.pi
#         else: 
#             separation = self.phi - self.theta
        
#         vfIndex = diff(secOrientations, np.degrees(separation)).argmin()
#         #return vfIndex 
#         return self.binaryVF[vfIndex]




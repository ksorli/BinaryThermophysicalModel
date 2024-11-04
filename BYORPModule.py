#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 14:08:20 2024

@author: kyso3185

Module developed as part of the Binary Thermophysical Model (BTM) to utilize
calculated temperatures to estimate the magnitude of the BYORP effect. 

Last updated: March 2024 by Kya Sorli 
"""

import numpy as np
from heat3d_Binary import binaryModel
from scipy import integrate

# Constants
c = 3e8 #[m s-1]
radPresFactor = 1./c
sigma = 5.67051196e-8 # Stefan-Boltzmann Constant
TWOPI = 6.283185307

###############################################################################
## Functions
###############################################################################

def RadiationForces(model: binaryModel, secModel, includeAbsorption = False): 
    # Initially based off of radiation pressure equation given in Hesar et al. (2017)
    # Arguments
    #     model: instance of model, containing information about fluxes, temperatures and thermophysical properties
    #     secModel: rotated deepcopy of secondary shape model corresponding to current state of system 
        
    reflectedForceVectors = CalculateDiffusedReflectionForce(model, secModel)
    
    emittedForceVectors = CalculateDiffusedEmissionForce(model, secModel)
    
    if includeAbsorption: 
        absorbedForceVectors = CalculateAbsorptionForce(model, secModel)
    else: 
        absorbedForceVectors = np.zeros(np.shape(reflectedForceVectors))
    
    # Sum different forces 
    forceVectors = absorbedForceVectors + reflectedForceVectors + emittedForceVectors
    
    netForceVector = np.sum(forceVectors, axis = 0)

    return netForceVector 

def CalculateAbsorptionForce(model: binaryModel, secModel): #, radiance):
    # Force from incoming insolation
    # Strength of incident flux is affected by starting insolation, albedo, angle 
    #    facet area and shadowing
    # Absorption force is directed along the wave direction of insolation aka vector s 
    # NOTE: BYORP is only concerned with secondary 
    
    # Get fluxes from model 
    secFlux = model.secProfile.Qs # Array # of facets long. Incident radiation on facet, including visibility, incidence angle, view factors, and albedo
    
    # Multiple by radiation pressure factor and areas 
    # Shape after should be 1 x number of facets (scalar values)
    secPressureTimesArea = radPresFactor * np.multiply(secFlux, secModel.areasSqMeters) # Get the radiation pressure, multiply by areas 
    
    # Get sun to facet vectors 
    secForceVectors = (secModel.centroids - model.sLoc)  # vectors from sun to each centroid
    magnitudes = np.linalg.norm(secForceVectors, axis = 1) # axis = 1 if vectors shape is nx3 
    secForceVectorsNormalized = secForceVectors / magnitudes[:,np.newaxis] 
    
    
    # Multiply by vector component 
    secAbsorbedForce = secForceVectorsNormalized*secPressureTimesArea[:,np.newaxis] # Radiation pressure x area x vector from sun to triangle 
    
    
    return secAbsorbedForce


def CalculateDiffusedReflectionForce(model: binaryModel, secModel):
    # Force from reflection (recoil)
    # Assumes lambertian (diffuse reflector)
    # Strength of reflection is affected by starting insolation, albedo, angle 
    #    facet area and shadowing
    # Tangential components cancel, leaving net force vector along - normal vector 
    # Utilizes, in part, methodology from Canuto, Enrico, et al. Spacecraft dynamics and control: the embedded model control approach. Butterworth-Heinemann, 2018. 
    
    
    # Don't use Qs. Need insolation x albedo 
    secRefFlux = model.secProfile.reflectedQs 
    
    # Assume diffuse reflection integrated over hemisphere with solid angle 2pi
    # Symmetric tangential components cancel. Normal components conserved (2/3 factor)
    secPressureTimesArea = radPresFactor * (2 /3) * np.multiply(secRefFlux,secModel.areasSqMeters) # radiation pressure constant, reflected flux, areas
    
    # Vectors. Recoil force is directed in direction of negative normal vectors 
    # Normals are already normalized to length of 1 
    secRefForceVectors = - secModel.trimesh_normals 
    
    # Force [N]
    # PressureTimesArea * secRefForceVectors
    secRefForce = secRefForceVectors*secPressureTimesArea[:,np.newaxis]
    
    return secRefForce 


def CalculateDiffusedEmissionForce(model: binaryModel, secModel): #, emis):
    # Force from diffusion of thermal infrared energy  
    # Assumes lambertian (diffuse emitter)
    # Strength of emission is affected by temperature, emissivity, and facet area 
    # Tangential components cancel, leaving net force vector along - normal vector 
    # Utilizes, in part, methodology from Canuto, Enrico, et al. Spacecraft dynamics and control: the embedded model control approach. Butterworth-Heinemann, 2018. 
    
    # Emission from temperature 
    # Surface temperatures 
    secEmisFlux = model.secPlanet.emissivity * sigma * np.power(model.secProfile.T[0,:],4)
    #secEmisFlux = model.secPlanet.emissivity * sigma * model.secProfile.T[0,:]
    
    # Pressure 
    # Diffuse emission (symmetric tangential components cancel and normal component conserved) --> 2/3 factor 
    secEmisPressure = radPresFactor * (2/3) * np.multiply(secEmisFlux, secModel.areasSqMeters)
    
    # Vectors: recoil force in the negative normal vector direction 
    secEmisForceVectors = - secModel.trimesh_normals
    
    # Force [N]
    secEmisForce = secEmisForceVectors*secEmisPressure[:,np.newaxis]
    
    
    return secEmisForce


def NormalizeCoefficients(model,netForces, rotationVals):
    # Using either Fourier decomposition (McMahon & Scheeres (2010) or numerical thermophysical modeling (this model))
    #     Get forces for each facet, divided by P(R). These are comparable to the coefficients from Fourier decomposition
    #     Coefficients have units Length scale squared
    #     Divide coefficients by normalizing radius squared (convention to use radius of volume equivalent sphere)
    
    # Inputs
    #     model: instance of the binary model 
    #     coefficients: 0th order coefficients along track direction (y component of forces in fixed frame of the secondary )
    
    p_r = model.solarFlux * radPresFactor # P(R) from Scheeres paper: solar pressure based on distance from sun 
    
    F_y = netForces[:,1]
    
    # Divide forces by p_r to get rid of heliocentric dependence 
    F_div_p_r = F_y / p_r
    
    # Integrate under y curve over 1 mutual orbit (2pi)
    xVals = rotationVals
    integrateCoeff = integrate.simpson(F_div_p_r, xVals)
    
    
    normalizedCoeff = integrateCoeff / model.secPlanet.radiusSphere**2 # Unitless, normalized coefficient for total body along track direction for secondary
    normCoeff_div_twopi = normalizedCoeff / (2 * np.pi)
    
    print ("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
    print ("Normalized Coefficient B: {}".format(normCoeff_div_twopi))
    print ("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
    
    return normalizedCoeff, normCoeff_div_twopi





############################################################
# Planetary Database                                       #
# Modified by Paul Hayne from the "planets.py" library by  #
# Raymond T. Pierrehumbert                                 #
#                                                          #
# Last modified: October, 2024                             #
#                                                          #
# Sources:                                                 #
#     1. http://nssdc.gsfc.nasa.gov/planetary/factsheet/   #
#     2. Lang, K. (2012). Astrophysical data: planets and  #
#        stars. Springer Science & Business Media.         #
#     3. JPL Small Bodies Database                         #
#                                                          #
############################################################

# All units M.K.S. unless otherwise stated


# Dependencies
import numpy as np

# Constants
AU = 149.60e9 # Astronomical Unit [m]
sigma = 5.67e-8 # Stefan-Boltzmann constant [W.m-2.K-4]
G = 6.67428e-11 # Gravitational constant
SOLARLUM = (3.826e26 / (4.0*np.pi)) # Solar Luminosity divided by 4pi steradians


class Planet:
    '''
    A Planet object contains basic planetary data.
    If P is a Planet object, the data are:
           P.name = Name of the planet
           P.R = Mean radius of planet [m]
           P.g = Surface gravitational acceleration [m.s-2]
           P.S = Annual mean solar constant (current) [W.m-2]
           P.psurf = Average atmospheric pressure at the surface [Pa]
           P.albedo = Bond albedo [fraction]
           P.emissivity = IR emissivity [fraction]
           P.Qb = Crustal/basal heat flow (average) [W.m-2]
           P.gamma = Surface layer thermal inertia [J.m-2.K-1.s-1/2]
           
           P.rsm = Semi-major axis of orbit about Sun [m]
           P.rAU = Semi-major axis of orbit about Sun [AU]
           P.year = Sidereal length of year [s]
           P.eccentricity =  Orbital eccentricity [unitless]
           P.day = Mean length of solar day [s]
           P.obliquity = Obliquity to orbit [radian]
           P.Lequinox = Longitude of equinox [radian]
           P.Lp = Longitude of perihelion [radian]

           P.Tsavg = Mean surface temperature [K]
           P.Tsmax = Maximum surface temperature [K]

    For gas giants, "surface" quantities are given at the 1 bar level
    '''

    #__repr__ object prints out a help string when help is
    #invoked on the planet object or the planet name is typed
    def __repr__(self):
        line1 =\
        'This planet object contains information on %s\n'%self.name
        line2 = 'Type \"help(Planet)\" for more information\n'
        return line1+line2
    def __init__(self):
        self.name = None #Name of the planet
        self.R = None #Mean radius of planet
        self.g = None #Surface gravitational acceleration
        self.S = None #Annual mean solar constant (current)
        self.psurf = None # Surface pressure [Pa]
        self.albedo = None #Bond albedo
        self.albedoCoef = [0.0, 0.0] # Coefficients in variable albedo model
        self.emissivity = None #IR emissivity [fraction]
        self.Qb = None #Crustal heat flow (average) [W.m-2]
        self.Gamma = None #Thermal inertia [J.m-2.K-1.s-1/2]
        self.ks = None # Solid (phonon) conductivity at surface [W.m-1.K-1]
        self.kd = None # Solid (phonon) conductivity at depth z>>H [W.m.K-1]
        self.rhos = None # Density at surface [kg.m-3]
        self.rhod = None # Density at depth z>>H [kg.m-3]
        self.H = None # e-folding scale of conductivity and density [m]
        self.cp0 = None # heat capacity at average surface temp. [J.kg.K-1]
        self.cpCoeff = None# Heat capacity polynomial coefficients
        self.rsm = None #Semi-major axis
        self.rAU = None #Semi-major axis [AU]
        self.year = None #Sidereal length of year
        self.eccentricity = None # Eccentricity
        self.day = None #Mean length of solar day
        self.obliquity = None #Obliquity to orbit
        self.Lequinox = None #Longitude of equinox
        self.Lp = None #Longitude of perihelion

        self.Tsavg = None #Mean surface temperature
        self.Tsmax = None #Maximum surface temperature
        
        # binary 
        self.separation = 0 #Separation between primary and secondary (from origin to origin of shape models) [km]
        self.secondaryMass = None # Mass for secondary if known. Used for BYORP calculations 
        self.secondaryVolume = None # Estimated volume of secondary if known. Used for BYORP calculations
        self.radiusSphere = None # Radius of volume equivalent sphere. Used for BYORP 
        
        
    def Teq(self, latitude=0):
        F = self.S
        A = self.albedo
        e = self.emissivity
        return ( (1-A)*F*np.cos(latitude*np.pi/180)/(4*e*sigma) )**0.25
    
    def calcObliquity(self, lat, long, inc, Omega):
        # Lat, long: ecliptic latitude and longitude of the spin pole
        # inc: orbital inclination
        # Omega: longitude of the ascending node 
        lat = np.radians(lat)
        long = np.radians(long)
        inc = np.radians(inc)
        Omega = np.radians(Omega)
        
        # Calculate the obl
        obliq = np.arcsin(np.sin(lat) * np.sin(inc) * np.cos(Omega) + np.cos(lat) * np.cos(inc))
        
        # Adjust for the hemisphere of the spin pole
        if lat < 0: 
            obliq = np.pi - obliq
            
        return np.degrees(obliq)
    
    
    def calcSolarFlux(self): # Solar flux at semi major axis distance 
        flux = SOLARLUM / (self.rAU*1.496e11)**2 # Flux in [W m-2]
        return flux
    
    def calcMaxT(self): # Maximum surface temperature based on perihelion distance 
        q = self.rsm * (1 - self.eccentricity)
        maxFlux = SOLARLUM / (q**2)
        maxT = (maxFlux / (self.emissivity*sigma))**.25
        return maxT

#----------------------------------------------------       
Mercury = Planet()        
Mercury.name = 'Mercury' #Name of the planet
Mercury.R = 2.4397e6 #Mean radius of planet
Mercury.g = 3.70 #Surface gravitational acceleration
Mercury.albedo = .119 #Bond albedo
Mercury.emissivity = 0.95 # Infrared emissivity
Mercury.S = 9126.6 #Annual mean solar constant (current)
Mercury.psurf = 1e-9 # Surface pressure [Pa]
#
Mercury.rsm = 57.91e9 #Semi-major axis
Mercury.rAU = Mercury.rsm/AU #Semi-major axis [AU]
Mercury.year = 87.969*24.*3600. #Sidereal length of year [s]
Mercury.eccentricity = .2056 # Eccentricity
Mercury.day = 4222.6*3600. #Mean length of solar day [s]
Mercury.obliquity = .01 #Obliquity to orbit [deg]
Mercury.Lequinox = None #Longitude of equinox [deg]
#
Mercury.Tsavg = 440. #Mean surface temperature
Mercury.Tsmax = 725. #Maximum surface temperature

#----------------------------------------------------        
Venus = Planet()
Venus.name = 'Venus' #Name of the planet
Venus.R = 6.0518e6 #Mean radius of planet
Venus.g = 8.87 #Surface gravitational acceleration
Venus.albedo = .750 #Bond albedo
Venus.emissivity = 0.95 # Infrared emissivity
Venus.S = 2613.9 #Annual mean solar constant (current)
Venus.psurf = 9.3e4 # Surface pressure [Pa]
#
Venus.rsm = 108.21e9 #Semi-major axis
Venus.rAU = Venus.rsm/AU #Semi-major axis [AU]
Venus.year = 224.701*24.*3600. #Sidereal length of year [s]
Venus.eccentricity = .0067 # Eccentricity
Venus.day = 2802.*3600. #Mean length of solar day [s]
Venus.obliquity = 177.36 #Obliquity to orbit [deg]
Venus.Lequinox = None #Longitude of equinox [deg]
#
Venus.Tsavg = 737. #Mean surface temperature [K]
Venus.Tsmax = 737. #Maximum surface temperature [K]

#----------------------------------------------------        
Earth = Planet()
Earth.name = 'Earth' #Name of the planet
Earth.R = 6.371e6 #Mean radius of planet
Earth.g = 9.798 #Surface gravitational acceleration
Earth.S = 1361 #Annual mean solar constant (current)
Earth.albedo = .306 #Bond albedo
Earth.emissivity = 0.95 # Infrared emissivity
Earth.psurf = 1.013e5 # Surface pressure [Pa]
#
Earth.rsm = 149.60e9 #Semi-major axis
Earth.rAU = Earth.rsm/AU #Semi-major axis [AU]
Earth.year = 365.256*24.*3600. #Sidereal length of year [s]
Earth.eccentricity = .0167 # Eccentricity
Earth.day = 24.000*3600. #Mean length of solar day [s]
Earth.obliquity = 23.45 #Obliquity to orbit [deg]
Earth.Lequinox = None #Longitude of equinox [deg]
#
Earth.Tsavg = 288. #Mean surface temperature [K]
Earth.Tsmax = 320. #Maximum surface temperature [K]

#----------------------------------------------------        
Mars = Planet()
Mars.name = 'Mars' #Name of the planet
Mars.R = 3.390e6 #Mean radius of planet
Mars.g = 3.71 #Surface gravitational acceleration
Mars.albedo = .250 #Bond albedo
Mars.emissivity = 0.95 # Infrared emissivity
Mars.S = 589.2 #Annual mean solar constant (current)
Mars.psurf = 632 # Average surface pressure [Pa]
#
Mars.rsm = 227.92e9 #Semi-major axis
Mars.rAU = Mars.rsm/AU #Semi-major axis [AU]
Mars.year = 686.98*24.*3600. #Sidereal length of year [s]
Mars.eccentricity = .0935 # Eccentricity
Mars.day = 24.6597*3600. #Mean length of solar day [s]
Mars.obliquity = 25.19 #Obliquity to orbit [deg]
Mars.Lequinox = None #Longitude of equinox [deg]
#
Mars.Tsavg = 210. #Mean surface temperature [K]
Mars.Tsmax = 295. #Maximum surface temperature [K]

#----------------------------------------------------        
Jupiter = Planet()
Jupiter.name = 'Jupiter' #Name of the planet
Jupiter.R = 69.911e6 #Mean radius of planet
Jupiter.g = 24.79 #Surface gravitational acceleration
Jupiter.albedo = .343 #Bond albedo
Jupiter.S = 50.5 #Annual mean solar constant (current)
#
Jupiter.rsm = 778.57e9 #Semi-major axis
Jupiter.rAU = Jupiter.rsm/AU #Semi-major axis [AU]
Jupiter.year = 4332.*24.*3600. #Sidereal length of year [s]
#Jupiter.eccentricity = .0489 # Eccentricity
Jupiter.eccentricity = 0. # Eccentricity
Jupiter.day = 9.9259*3600. #Mean length of solar day [s]
Jupiter.obliquity = 0.0546288 #Obliquity to orbit [radians]
Jupiter.Lequinox = None #Longitude of equinox [radians]
#
Jupiter.Tsavg = 165. #Mean surface temperature [K]
Jupiter.Tsmax = None #Maximum surface temperature [K]

#----------------------------------------------------        
Saturn = Planet()
Saturn.name = 'Saturn' #Name of the planet
Saturn.R = 58.232e6 #Mean radius of planet
Saturn.g = 10.44 #Surface gravitational acceleration
Saturn.albedo = .342 #Bond albedo
Saturn.S = 14.90 #Annual mean solar constant (current)
#
Saturn.rsm = 1433.53e9 #Semi-major axis
Saturn.rAU = Saturn.rsm/AU #Semi-major axis [AU]
Saturn.year = 10759.*24.*3600. #Sidereal length of year [s]
Saturn.eccentricity = .0565 # Eccentricity
Saturn.day = 10.656*3600. #Mean length of solar day [s]
Saturn.obliquity = 26.73 #Obliquity to orbit [deg]
Saturn.Lequinox = None #Longitude of equinox [deg]
#
Saturn.Tsavg = 134. #Mean surface temperature [K]
Saturn.Tsmax = None #Maximum surface temperature [K]

#----------------------------------------------------        
Uranus = Planet()
Uranus.name = 'Uranus' #Name of the planet
Uranus.R = 25.362e6 #Mean radius of planet
Uranus.g = 8.87 #Surface gravitational acceleration
Uranus.albedo = .300 #Bond albedo
Uranus.S = 3.71 #Annual mean solar constant (current)
#
Uranus.rsm = 2872.46e9 #Semi-major axis
Uranus.rAU = Uranus.rsm/AU #Semi-major axis [AU]
Uranus.year = 30685.4*24.*3600. #Sidereal length of year [s]
Uranus.eccentricity = .0457 # Eccentricity
Uranus.day = 17.24*3600. #Mean length of solar day [s]
Uranus.obliquity = 97.77 #Obliquity to orbit [deg]
Uranus.Lequinox = None #Longitude of equinox [deg]
#
Uranus.Tsavg = 76. #Mean surface temperature [K]
Uranus.Tsmax = None #Maximum surface temperature [K]


#----------------------------------------------------        
Neptune = Planet()
Neptune.name = 'Neptune' #Name of the planet
Neptune.R = 26.624e6 #Mean radius of planet
Neptune.g = 11.15 #Surface gravitational acceleration
Neptune.albedo = .290 #Bond albedo
Neptune.S = 1.51 #Annual mean solar constant (current)
#
Neptune.rsm = 4495.06e9 #Semi-major axis
Neptune.rAU = Neptune.rsm/AU #Semi-major axis [AU]
Neptune.year = 60189.0*24.*3600. #Sidereal length of year [s]
Neptune.eccentricity = .0113 # Eccentricity
Neptune.day = 16.11*3600. #Mean length of solar day [s]
Neptune.obliquity = 28.32 #Obliquity to orbit [deg]
Neptune.Lequinox = None #Longitude of equinox [deg]
#
Neptune.Tsavg = 72. #Mean surface temperature [K]
Neptune.Tsmax = None #Maximum surface temperature [K]

#----------------------------------------------------        
Pluto = Planet()
Pluto.name = 'Pluto' #Name of the planet
Pluto.R = 1.195e6 #Mean radius of planet
Pluto.g = .58 #Surface gravitational acceleration
Pluto.albedo = .5 #Bond albedo
Pluto.emissivity = 0.95 # Infrared emissivity
Pluto.S = .89 #Annual mean solar constant (current)
Pluto.psurf = 1.0 # Average surface pressure [Pa]
#
Pluto.rsm = 5906.e9 #Semi-major axis
Pluto.rAU = Pluto.rsm/AU #Semi-major axis [AU]
Pluto.year = 90465.*24.*3600. #Sidereal length of year [s]
Pluto.eccentricity = .2488 # Eccentricity
Pluto.day = 153.2820*3600. #Mean length of solar day [s]
Pluto.obliquity = 122.53 #Obliquity to orbit [deg]
Pluto.Lequinox = None #Longitude of equinox [deg]
#
Pluto.Tsavg = 50. #Mean surface temperature [K]
Pluto.Tsmax = None #Maximum surface temperature [K]


#Selected moons

#----------------------------------------------------  
############
# Earth Moon
############      
Moon = Planet()
Moon.name = 'Moon' #Name of the planet
Moon.R = 1.7374e6 #Mean radius of planet [m]
Moon.g = 1.62 #Surface gravitational acceleration [m.s-2]
Moon.S = 1361. #Annual mean solar constant [W.m-2]
Moon.psurf = 3.0e-10 # Surface pressure [Pa]

Moon.albedo = 0.12 #Bond albedo
Moon.albedoCoef = [0.06, 0.25] # Coefficients in variable albedo model
Moon.emissivity = .95 #IR emissivity
Moon.Qb = 0.018 #Heat flow [W.m-2]
# Thermophysical properties:
Moon.Gamma = 55. #Thermal inertia [J.m-2.K-1.s-1/2]
Moon.ks = 7.4e-4 # Solid (phonon) conductivity at surface [W.m-1.K-1]
Moon.kd = 3.4e-3 # Solid (phonon) conductivity at depth z>>H [W.m.K-1]
Moon.rhos = 1100. # Density at surface [kg.m-3]
Moon.rhod = 1800. # Density at depth z>>H [kg.m-3]
Moon.H = 0.07 # e-folding scale of conductivity and density [m]
Moon.cp0 = 600. # heat capacity at average surface temp. [J.kg.K-1]
Moon.cpCoeff = [8.9093e-9,-1.234e-5,\
                2.3616e-3,2.7431,-3.6125] # Heat capacity polynomial coefficients
#
Moon.rsm = Earth.rsm #Semi-major axis
Moon.rAU = Moon.rsm/AU #Semi-major axis [AU]
Moon.year = Earth.year #Sidereal length of year
Moon.eccentricity = Earth.eccentricity # Eccentricity
Moon.day = 29.53059*24.*3600. #Mean length of SYNODIC day [s]
Moon.obliquity = 0.026878 #Obliquity to orbit [radian]
Moon.Lequinox = None #Longitude of equinox [radian]
Moon.Lp = 0. # Longitude of perihelion [radian]
#
Moon.Tsavg = 250. #Mean surface temperature [K]
Moon.Tsmax = 400. #Maximum surface temperature [K]
Moon.Tsmin = 95. #Minimum surface temperature [K]


############
# Titan
############
Titan = Planet()
Titan.name = 'Titan' #Name of the planet
Titan.R = 2.575e6 #Mean radius of planet
Titan.g = 1.35 #Surface gravitational acceleration
Titan.S = Saturn.S #Annual mean solar constant (current)
Titan.albedo = .22 #Bond albedo (Not yet updated from Cassini)
Titan.emissivity = 0.95 # Infrared emissivity
Titan.psurf = 1.5e5 # Average surface pressure [Pa]
#        
Titan.rsm = Saturn.rsm #Semi-major axis [m]
Titan.rAU = Titan.rsm/AU #Semi-major axis [AU]
Titan.year = Saturn.year #Sidereal length of year [s]
Titan.eccentricity = Saturn.eccentricity # Eccentricity ABOUT SUN
Titan.day = 15.9452*24.*3600. #Mean length of solar day [s]
Titan.obliquity = Saturn.obliquity #Obliquity to plane of Ecliptic
                                   #(Titan's rotation axis approx parallel
                                   # to Saturn's
Titan.Lequinox = Saturn.Lequinox #Longitude of equinox
#
Titan.Tsavg = 92. #Mean surface temperature [K]
Titan.Tsmax = 94. #Maximum surface temperature [K]


############
# Europa
############
Europa = Planet()
Europa.name = 'Europa' #Name of the planet
Europa.R = 1.560e6 #Mean radius of planet
Europa.g = 1.31 #Surface gravitational acceleration
Europa.psurf = 1.0e-7 # Average surface pressure [Pa]

Europa.S = Jupiter.S #Annual mean solar constant (current)
Europa.albedo = 0.6 #Bond albedo

Europa.emissivity = 0.90 #IR emissivity
Europa.Qb = 0.030 #basal heat flow [W.m-2]
#Thermophysical properties:
Europa.ks = 2e-3 # Solid (phonon) conductivity at surface [W.m-1.K-1]
Europa.kd = 1e-2 # Solid (phonon) conductivity at depth z>>H [W.m.K-1]
Europa.rhos = 100. # Density at surface [kg.m-3]
Europa.rhod = 450. # Density at depth z>>H [kg.m-3]
Europa.H = 0.07 # e-folding scale of conductivity and density [m]
Europa.cp0 = 900 # heat capacity at average surface temp. [J.kg.K-1]
Europa.cpCoeff = [90.0,7.49] # Heat capacity polynomial coefficients
#        
Europa.rsm = Jupiter.rsm #Semi-major axis [m]
Europa.rAU = Europa.rsm/AU #Semi-major axis [AU]
Europa.year = Jupiter.year #Sidereal length of year [s]
Europa.eccentricity = Jupiter.eccentricity # Eccentricity
Europa.day = 3.06822e5 #Mean length of solar day [s]
Europa.obliquity = Jupiter.obliquity #Obliquity to plane of ecliptic
Europa.Lequinox = None #Longitude of equinox
Europa.Lp = 0. #Longitude of perihelion [radians]
#
Europa.Tsavg = 103. #Mean surface temperature [K]
Europa.Tsmax = 130. #Maximum surface temperature [K]

############
# Ganymede
############

Ganymede = Planet()
Ganymede.name = 'Ganymede' #Name of the planet
Ganymede.R = 2.631e6 #Mean radius of planet [m]
Ganymede.g = 1.43 #Surface gravitational acceleration
Ganymede.psurf = 1.0e-6 # Surface pressure [Pa]

Ganymede.S = Jupiter.S #Annual mean solar constant (current)
Ganymede.albedo = 0.4 #Bond albedo
Ganymede.emissivity = 0.90 #IR emissivity
Ganymede.Qb = 0.030 #basal heat flow [W.m-2]
#Thermophysical properties:
Ganymede.ks = 2e-3 # Solid (phonon) conductivity at surface [W.m-1.K-1]
Ganymede.kd = 1e-2 # Solid (phonon) conductivity at depth z>>H [W.m.K-1]
Ganymede.rhos = 100. # Density at surface [kg.m-3]
Ganymede.rhod = 450. # Density at depth z>>H [kg.m-3]
Ganymede.H = 0.07 # e-folding scale of conductivity and density [m]
Ganymede.cp0 = 900 # heat capacity at average surface temp. [J.kg.K-1]
Ganymede.cpCoeff = [90.0,7.49] # Heat capacity polynomial coefficients
#        
Ganymede.rsm = Jupiter.rsm #Semi-major axis [m]
Ganymede.rAU = Ganymede.rsm/AU #Semi-major axis [AU]
Ganymede.year = Jupiter.year #Sidereal length of year [s]
Ganymede.eccentricity = Jupiter.eccentricity # Eccentricity
Ganymede.day = 6.18192e5 #Mean length of solar day [s]
Ganymede.obliquity = Jupiter.obliquity #Obliquity to plane of ecliptic
Ganymede.Lequinox = None #Longitude of equinox
Ganymede.Lp = 0. #Longitude of perihelion [radians]
#
Ganymede.Tsavg = 110. #Mean surface temperature [K]
Ganymede.Tsmax = 140. #Maximum surface temperature [K]

############
# Triton
############

Triton = Planet()
Triton.name = 'Triton' #Name of the planet
Triton.R = 2.7068e6/2. #Mean radius of planet
Triton.g = .78 #Surface gravitational acceleration
Triton.psurf = 2e-5 # Average surface pressure [Pa]

Triton.S = Neptune.S #Annual mean solar constant (current)
Triton.albedo = .76 #Bond albedo
Triton.emissivity = 0.95 # Infrared emissivity
#        
Triton.rsm = Neptune.rsm #Semi-major axis [m]
Triton.rAU = Triton.rsm/AU #Semi-major axis [AU]
Triton.year = Neptune.year #Sidereal length of year
Triton.eccentricity = Neptune.eccentricity # Eccentricity about Sun
Triton.day = 5.877*24.*3600. #Mean length of solar day [s]
                             #Triton's rotation is retrograde
Triton.obliquity = 156. #Obliquity to ecliptic **ToDo: Check this.
                        #Note: Seasons are influenced by the inclination
                        #of Triton's orbit? (About 20 degrees to
                        #Neptune's equator
Triton.Lequinox = None #Longitude of equinox
#
Triton.Tsavg = 34.5 #Mean surface temperature [K]
                    #This is probably a computed blackbody
                    #temperature, rather than an observation
Triton.Tsmax = None #Maximum surface temperature [K]


############
# Hyperion
############
Hyperion = Planet()
Hyperion.name = 'Hyperion' #Name of the planet
Hyperion.R = 135e3 #Mean radius of planet [m]                                 # Thomas 2010
Hyperion.g = 0.02 #Surface gravitational acceleration [m.s-2]                 # Thomas 2010
Hyperion.S = Saturn.S #Annual mean solar constant [W.m-2]                     
Hyperion.albedo = 0.38 #Bond albedo                                           # Assuming a phase integral of 1 and geometric of 0.38 
                                                                              # Midpoint of 0.3 (from Cruikshank 2007), 0.38 (Blackburn 2011) and 
                                                                              # Verbiscer chapter of enceladus book
                                                                              # This will change with each facet. High is ~0.3 and low ~0.5? for geometric
Hyperion.emissivity = 0.99 #IR emissivity
Hyperion.Qb = 0.0 #basal heat flow [W.m-2]
#Thermophysical properties:
Hyperion.ks = 0.00091912 # Solid (phonon) conductivity at surface [W.m-1.K-1] # Based on thermal inertia of 20 like other Saturnian moons from Howett 2010
Hyperion.kd = 0.00091912 # Solid (phonon) conductivity at depth z>>H [W.m.K-1]
Hyperion.rhos = 544 # Density at surface [kg.m-3]                             # Thomas 2010
Hyperion.rhod = 544 # Density at depth z>>H [kg.m-3]                          # Thomas 2010
#Hyperion.ks = 1.49
#Hyperion.kd = 1.49
#Hyperion.rhos = 2940.
#Hyperion.rhod = 2940.
Hyperion.H = Moon.H # e-folding scale of conductivity and density [m]
Hyperion.cp0 = 800 # heat capacity at average surface temp. [J.kg.K-1]        # Taken from value for Rhea from Spencer and Moore 1992
Hyperion.cpCoeff = Moon.cpCoeff # Heat capacity polynomial coefficients
#        
Hyperion.rsm = Saturn.rsm #Semi-major axis [m]                                # Using saturn presets for now. Rotation is chaotic so year/day is relative          
Hyperion.rAU = Hyperion.rsm/AU #Semi-major axis [AU]                          # 
Hyperion.year = Saturn.year #Sidereal length of year [s]
Hyperion.eccentricity = Saturn.eccentricity # Eccentricity
Hyperion.day = 13*24*3600 #Mean length of solar day [s]                       # Orbital period from Thomas et al 1984
                                                                              # Rotation period is chaotic
Hyperion.obliquity = 0 #Obliquity to plane of ecliptic [radians]              # Chaotic orbit
Hyperion.Lequinox = None #Longitude of equinox
Hyperion.Lp = 0.  #Longitude of perihelion [radians]
#
Hyperion.Tsavg = 100 #Mean surface temperature [K]                   # Use Saturn Presets 
Hyperion.Tsmax = 250 #Maximum surface temperature [K]

# Small bodies

#--------------------------------------------------------------------------

# # Bennu
# Bennu = Planet()
# Bennu.name = 'Bennu' #Name of the planet
# Bennu.R = 246. #Mean radius of planet [m]
# Bennu.g = 1.0e-5 #Surface gravitational acceleration [m.s-2]
# Bennu.S = 1072.7 #Annual mean solar constant [W.m-2]
# Bennu.albedo = 0.021 #Bond albedo (Li et al., 2021) 
# Bennu.emissivity = 0.9 #IR emissivity (Typical of silicate materials at Orex wavelengths according to DellaGiustina et al., 2019)
# Bennu.Qb = 0.0 #basal heat flow [W.m-2]
# #Thermophysical properties:
# Bennu.ks = Moon.ks # Solid (phonon) conductivity at surface [W.m-1.K-1]
# Bennu.kd = Moon.kd # Solid (phonon) conductivity at depth z>>H [W.m.K-1]
# Bennu.rhos = 1190. # Density at surface [kg.m-3] (Scheeres et al., 2019)
# Bennu.rhod = 1190. # Density at depth z>>H [kg.m-3] (Scheeres et al., 2019)
# #Bennu.ks = 1.49
# #Bennu.kd = 1.49
# #Bennu.rhos = 2940.
# #Bennu.rhod = 2940.
# Bennu.H = Moon.H # e-folding scale of conductivity and density [m]
# Bennu.cp0 = Moon.cp0 # heat capacity at average surface temp. [J.kg.K-1]
# Bennu.cpCoeff = Moon.cpCoeff # Heat capacity polynomial coefficients
# #        
# Bennu.rsm = 1.685e11 #Semi-major axis [m]
# Bennu.rAU = Bennu.rsm/AU #Semi-major axis [AU]
# Bennu.year = Earth.year #Sidereal length of year [s]
# Bennu.eccentricity = 0#0.204 # Eccentricity
# Bennu.day = 15469.2 #Mean length of solar day [s]
# Bennu.obliquity = 3.106686 #Obliquity to plane of ecliptic [radians]
# Bennu.Lequinox = None #Longitude of equinox
# Bennu.Lp = 0. #Longitude of perihelion [radians]
# #
# Bennu.Tsavg = 270. #Mean surface temperature [K]
# Bennu.Tsmax = 400. #Maximum surface temperature [K]

# Bennu
Bennu = Planet()
Bennu.name = 'Bennu' #Name of the planet
Bennu.R = 246. #Mean radius of planet [m]
Bennu.g = 1.0e-5 #Surface gravitational acceleration [m.s-2]
Bennu.S = 1072.7 #Annual mean solar constant [W.m-2]
Bennu.albedo = 0.016 #Bond albedo (Li et al., 2021)
Bennu.emissivity = 0.9 #IR emissivity (Typical of silicate materials at Orex wavelengths according to DellaGiustina et al., 2019)
Bennu.Qb = 0.0 #basal heat flow [W.m-2]
#Thermophysical properties:
Bennu.ks = 0.137254  #0.050560 # Solid (phonon) conductivity at surface [W.m-1.K-1]
Bennu.kd = 0.137254  #0.050560 # Solid (phonon) conductivity at depth z>>H [W.m.K-1]
Bennu.rhos = 1190. # Density at surface [kg.m-3] (Scheeres et al., 2019)
Bennu.rhod = 1190. # Density at depth z>>H [kg.m-3] (Scheeres et al., 2019)
#
Bennu.H = Moon.H # e-folding scale of conductivity and density [m]
Bennu.cp0 = 750. #Moon.cp0 # heat capacity at average surface temp. [J.kg.K-1]
Bennu.cpCoeff = Moon.cpCoeff # Heat capacity polynomial coefficients
#
Bennu.rsm = 1.34e11 #1.685e11 #Semi-major axis [m]
Bennu.rAU = Bennu.rsm/AU #Semi-major axis [AU]
Bennu.year = 1.196 * Earth.year  #Sidereal length of year [s]
Bennu.eccentricity = 0. #0.204 # Eccentricity
Bennu.day = 15469.2 #Mean length of solar day [s]
Bennu.obliquity = 3.106686 #Obliquity to plane of ecliptic [radians]
Bennu.Lequinox = None #Longitude of equinox
Bennu.Lp = 0. #Longitude of perihelion [radians]
#
Bennu.Tsavg = 270. #Mean surface temperature [K]
Bennu.Tsmax = 400. #Maximum surface temperature [K]



# 1991 VH
VH1991 = Planet()
VH1991.name = 'VH1991' #Name of the planet
VH1991.R = 575. #Mean radius of planet [m]
VH1991.g = 1.0e-5 #Surface gravitational acceleration [m.s-2]
VH1991.S = 1072.7 #Annual mean solar constant [W.m-2]
VH1991.albedo = 0.0475 #Bond albedo
VH1991.emissivity = 0.95 #IR emissivity
VH1991.Qb = 0.0 #basal heat flow [W.m-2]
#Thermophysical properties:
# Currently using Itokawa like (Muller et al 2014, thermal inertia 700+/-200) 700: 0.68627, 500: 0.35014, 900: 1.1344
VH1991.ks = 0.002241 # Solid (phonon) conductivity at surface [W.m-1.K-1]
VH1991.kd = 0.002241 # Solid (phonon) conductivity at depth z>>H [W.m.K-1]
VH1991.rhos = 1190 # Density at surface [kg.m-3]
VH1991.rhod = 1190 # Density at depth z>>H [kg.m-3]
#VH1991.ks = 1.49
#VH1991.kd = 1.49
#VH1991.rhos = 2940.
#VH1991.rhod = 2940.
VH1991.H = Moon.H # e-folding scale of conductivity and density [m]
VH1991.cp0 = Moon.cp0 # heat capacity at average surface temp. [J.kg.K-1]
VH1991.cpCoeff = Moon.cpCoeff # Heat capacity polynomial coefficients
#        
VH1991.rsm = 1.9149e11  #Semi-major axis [m] 
                        #1.4561e11 standard, 1.4561e+11 perihelion, 1.9471e+11 aphelion, 1.9149e11 intersect
VH1991.rAU = 1.28    #Semi-major axis [AU] 
                        #1.137 standard, 0.97337 perihelion, 1.30154 aphelion, 1.28 intersect
VH1991.year = 4.57e7  #Sidereal length of year [s]
VH1991.eccentricity = 0 #0.144  # Eccentricity
VH1991.day = 9445.32  #Mean length of solar day [s]
VH1991.obliquity = 0  #Obliquity to plane of ecliptic [radians]
VH1991.Lequinox = None  #Longitude of equinox
VH1991.Lp = 0.  #Longitude of perihelion [radians]
#
VH1991.Tsavg = 270. #Mean surface temperature [K]
VH1991.Tsmax = 400. #Maximum surface temperature [K]


# 1996 FG3
FG31996 = Planet()
FG31996.name = 'FG31996' #Name of the planet
FG31996.R = 845. #Mean radius of planet [m]
FG31996.g = 1.0e-5 #Surface gravitational acceleration [m.s-2]
FG31996.S = 1072.7 #Annual mean solar constant [W.m-2]
FG31996.albedo = 0.011 #Bond albedo (one fourth of geometric albedo = 0.044)
FG31996.emissivity = 0.95 #IR emissivity
FG31996.Qb = 0.0 #basal heat flow [W.m-2]
#Thermophysical properties:
# Currently Yu (2014) thermal inertia of 80 +/- 40 (0.008965), 40: (0.002241), 120: 0.020168
FG31996.ks = 0.020168 # Solid (phonon) conductivity at surface [W.m-1.K-1]
                    #0.1286  #Bennu-like (Assuming Thermal inertia of ~303.3 from Emery et al 2019, Rozitis et al 2019)
FG31996.kd = 0.020168 # Solid (phonon) conductivity at depth z>>H [W.m.K-1]
FG31996.rhos = 1190 # Density at surface [kg.m-3]
FG31996.rhod = 1190 # Density at depth z>>H [kg.m-3]
#FG31996.ks = 1.49
#FG31996.kd = 1.49
#FG31996.rhos = 2940.
#FG31996.rhod = 2940.
FG31996.H = Moon.H # e-folding scale of conductivity and density [m]
FG31996.cp0 = Moon.cp0 # heat capacity at average surface temp. [J.kg.K-1]
FG31996.cpCoeff = Moon.cpCoeff # Heat capacity polynomial coefficients
#        
FG31996.rsm = 1.685e11  #Semi-major axis [m] 
                        #1.685e11 standard, 1.025e+11 perihelion, 2.128e+11 aphelion, 2.124e11 intersect
FG31996.rAU = 1.054    #Semi-major axis [AU] 
                        #1.054 standard, 0.6853 at perihelion, 1.4223 at aphelion, 1.42 intersect 
FG31996.year = 3.41412e7#5.34e7 #Sidereal length of year [s]
FG31996.eccentricity = 0.349 # Eccentricity
FG31996.day = 12942.7 #Mean length of solar day [s]
FG31996.obliquity = -3.0542#-1.44862328 #Obliquity to plane of ecliptic [radians]
FG31996.Lequinox = None #Longitude of equinox
FG31996.Lp = 0. #Longitude of perihelion [radians]
#
FG31996.Tsavg = 270. #Mean surface temperature [K]
FG31996.Tsmax = 400. #Maximum surface temperature [K]

# Apophis
Apophis = Planet()
Apophis.name = 'Apophis' #Name of the planet
Apophis.R = 370. #Mean radius of planet [m]
Apophis.g = 1.0e-5 #Surface gravitational acceleration [m.s-2]
Apophis.S = 1072.7 #Annual mean solar constant [W.m-2]
Apophis.albedo = 0.14 #Bond albedo
Apophis.emissivity = 0.95 #IR emissivity
Apophis.Qb = 0.0 #basal heat flow [W.m-2]
#Thermophysical properties:
#Sb type 
# Currently using Itokawa like (Muller et al 2014, thermal inertia 700+/-200) 700: 0.68627, 500: 0.35014, 900: 1.1344
Apophis.ks = 0.5042 # Solid (phonon) conductivity at surface [W.m-1.K-1]
Apophis.kd = 0.5042 # Solid (phonon) conductivity at depth z>>H [W.m.K-1]
Apophis.rhos = 1190 # Density at surface [kg.m-3]
Apophis.rhod = 1190 # Density at depth z>>H [kg.m-3]
#Apophis.ks = 1.49
#Apophis.kd = 1.49
#Apophis.rhos = 2940.
#Apophis.rhod = 2940.
Apophis.H = Moon.H # e-folding scale of conductivity and density [m]
Apophis.cp0 = Moon.cp0 # heat capacity at average surface temp. [J.kg.K-1]
Apophis.cpCoeff = Moon.cpCoeff # Heat capacity polynomial coefficients
#        
Apophis.rsm = 1.37995e11  #Semi-major axis [m] 
Apophis.rAU = .922438   #Semi-major axis [AU] 
Apophis.year = 2.795e7  #Sidereal length of year [s]
Apophis.eccentricity = .191 #0.144  # Eccentricity
Apophis.day = 110020  #Mean length of solar day [s]
Apophis.obliquity = 2.88  #Obliquity to plane of ecliptic [radians]
Apophis.Lequinox = None  #Longitude of equinox
Apophis.Lp = 0.  #Longitude of perihelion [radians]
#
Apophis.Tsavg = 270. #Mean surface temperature [K]
Apophis.Tsmax = 400. #Maximum surface temperature [K]


# Didymos A
Dida = Planet()
Dida.name = 'Didymos A' #Name of the planet
Dida.R = 730. #Mean radius of planet [m]
Dida.g = 1.0e-5 #Surface gravitational acceleration [m.s-2]
Dida.S = 1072.7 #Annual mean solar constant [W.m-2]
Dida.albedo = 0.08 #Bond albedo #0.07
Dida.emissivity = 0.95 #IR emissivity
Dida.Qb = 0.0 #basal heat flow [W.m-2]
#Thermophysical properties:
#Sb type 
# Currently using Itokawa like (Muller et al 2014, thermal inertia 700+/-200) 700: 0.68627, 500: 0.35014, 900: 1.1344
Dida.ks = 0.1920 # Solid (phonon) conductivity at surface [W.m-1.K-1]
Dida.kd = 0.1920 # Solid (phonon) conductivity at depth z>>H [W.m.K-1]
Dida.rhos = 2790 # Density at surface [kg.m-3]
Dida.rhod = 2790 # Density at depth z>>H [kg.m-3]
#Dida.ks = 1.49
#Dida.kd = 1.49
#Dida.rhos = 2940.
#Dida.rhod = 2940.
Dida.H = Moon.H # e-folding scale of conductivity and density [m]
Dida.cp0 = Moon.cp0 # heat capacity at average surface temp. [J.kg.K-1]
Dida.cpCoeff = Moon.cpCoeff # Heat capacity polynomial coefficients
#        
Dida.rsm = 2.45739195e11  #Semi-major axis [m] 
Dida.rAU = 1.642665059     #Semi-major axis [AU] 
Dida.year = 6.653e7   #Sidereal length of year [s]
Dida.eccentricity = 0.383264789 #0.144  # Eccentricity
Dida.day = 8136  #Mean length of solar day [s]
Dida.obliquity = 3.054  #Obliquity to plane of ecliptic [radians]
Dida.Lequinox = None  #Longitude of equinox
Dida.Lp = 0.  #Longitude of perihelion [radians]
#
Dida.Tsavg = 270. #Mean surface temperature [K]
Dida.Tsmax = 400. #Maximum surface temperature [K]


# 24 Themis
Themis = Planet()
Themis.name = 'Themis' #Name of the planet
Themis.R = 104e3 #Mean radius of planet [m] # 
Themis.g = 0.15 #Surface gravitational acceleration [m.s-2]
Themis.S = 138.128 #Annual mean solar constant [W.m-2]
Themis.albedo = 0.027 #Bond albedo #O'rourke (2020), switch to 0.067 from Jiang & Ji (2021) 
Themis.emissivity = 0.95 #IR emissivity
Themis.Qb = 0.0 #basal heat flow [W.m-2]
#Thermophysical properties:
#Sb type 
# Currently using Th.I. from Jiang & Ji(2021) of 29.5 +/- 26
Themis.ks = 0.0019851 # Solid (phonon) conductivity at surface [W.m-1.K-1] 
Themis.kd = 0.0019851 # Solid (phonon) conductivity at depth z>>H [W.m.K-1]
Themis.rhos = 1310 # Density at surface [kg.m-3] # C type 
Themis.rhod = 1310 # Density at depth z>>H [kg.m-3]
#
Themis.H = Moon.H # e-folding scale of conductivity and density [m]
Themis.cp0 = Moon.cp0 # heat capacity at average surface temp. [J.kg.K-1]
Themis.cpCoeff = Moon.cpCoeff # Heat capacity polynomial coefficients
#        
Themis.rsm = 4.6949e11  #Semi-major axis [m] 
Themis.rAU = 3.1383    #Semi-major axis [AU] 
Themis.year = 1.7545e8  #Sidereal length of year [s]
Themis.eccentricity = 0.1310 #0.144  # Eccentricity
Themis.day = 30147  #Mean length of solar day [s]
Themis.obliquity = 0  #Obliquity to plane of ecliptic [radians]
Themis.Lequinox = None  #Longitude of equinox
Themis.Lp = 0.  #Longitude of perihelion [radians]
#
Themis.Tsavg = 110. #Mean surface temperature [K]
Themis.Tsmax = 400. #Maximum surface temperature [K]



# Justitia 
Justitia = Planet()
Justitia.name = 'Themis' #Name of the planet
Justitia.R = 287e2 #Mean radius of planet [m] # 
Justitia.g = 0.15 #Surface gravitational acceleration [m.s-2]
Justitia.S = 138.128 #Annual mean solar constant [W.m-2]
Justitia.albedo = 0.027 #Bond albedo #O'rourke (2020), switch to 0.067 from Jiang & Ji (2021) 
Justitia.emissivity = 0.95 #IR emissivity
Justitia.Qb = 0.0 #basal heat flow [W.m-2]
#Thermophysical properties:
#Sb type 
# Currently using Th.I. from Jiang & Ji(2021) of 29.5 +/- 26
Justitia.ks = 0.0019851 # Solid (phonon) conductivity at surface [W.m-1.K-1] 
Justitia.kd = 0.0019851 # Solid (phonon) conductivity at depth z>>H [W.m.K-1]
Justitia.rhos = 1310 # Density at surface [kg.m-3] # C type 
Justitia.rhod = 1310 # Density at depth z>>H [kg.m-3]
#
Justitia.H = Moon.H # e-folding scale of conductivity and density [m]
Justitia.cp0 = Moon.cp0 # heat capacity at average surface temp. [J.kg.K-1]
Justitia.cpCoeff = Moon.cpCoeff # Heat capacity polynomial coefficients
#        
Justitia.rsm = 3.9122e11  #Semi-major axis [m] 
Justitia.rAU = 	2.61515    #Semi-major axis [AU] 
Justitia.year = 1.7545e8  #Sidereal length of year [s]
Justitia.eccentricity = 0.1310 #0.144  # Eccentricity
Justitia.day = 30147  #Mean length of solar day [s]
Justitia.obliquity = 0  #Obliquity to plane of ecliptic [radians]
Justitia.Lequinox = None  #Longitude of equinox
Justitia.Lp = 0.  #Longitude of perihelion [radians]
#
Justitia.Tsavg = 110. #Mean surface temperature [K]
Justitia.Tsmax = 400. #Maximum surface temperature [K]


# (10253) Westerwald 
Westerwald = Planet()
Westerwald.name = 'Westerwald' #Name of the planet
Westerwald.R = 1135. #Mean radius of planet [m] 
Westerwald.g = 1.0e-5 #Surface gravitational acceleration [m.s-2]
Westerwald.albedo = 0.07825 #Bond albedo # From geometric albedo 0.313 (SBD), assuming phase integral of 0.25
Westerwald.emissivity = 0.95 #IR emissivity
Westerwald.Qb = 0.0 #basal heat flow [W.m-2]

# Thermophysical properties: # Going off thermal inertia of ~200 from delbo et al 2007 size gamma plot
Westerwald.ks = 0.024829 # Solid (phonon) conductivity at surface [W.m-1.K-1] 
Westerwald.kd = 0.024829 # Solid (phonon) conductivity at depth z>>H [W.m.K-1]
Westerwald.rhos = 2685 # Density at surface [kg.m-3] # From Harish's doc (OC type)
Westerwald.rhod = 2685 # Density at depth z>>H [kg.m-3]
#
Westerwald.H = Moon.H # e-folding scale of conductivity and density [m]
Westerwald.cp0 = Moon.cp0 # heat capacity at average surface temp. [J.kg.K-1]
Westerwald.cpCoeff = Moon.cpCoeff # Heat capacity polynomial coefficients
#         
Westerwald.rAU = 	2.297    #Semi-major axis [AU]
Westerwald.rsm = Westerwald.rAU * AU  #Semi-major axis [m] 
Westerwald.year = 1.098e8  #Sidereal length of year [s]
Westerwald.eccentricity = 0.149  # Eccentricity
Westerwald.day = 18000 #Mean length of solar day [s] UNKNOWN, extrapolated from plot based on diameterfrom Johannson et al 2014
Westerwald.obliquity = 0 #np.radians( Westerwald.calcObliquity(0, 0, 0.378, 53.6776)) #Obliquity to plane of ecliptic [radians]
Westerwald.Lequinox = None  #Longitude of equinox
Westerwald.Lp = np.radians(53.6776 + 63.36)  #Longitude of perihelion [radians], equal to the sum of the longitude of the ascending node + argument of periapsis 
#
Westerwald.S = 258. #Annual mean solar constant [W.m-2]
Westerwald.Tsavg = 182. #Mean surface temperature [K]
Westerwald.Tsmax = 290. #Maximum surface temperature [K]




# (623) Chimaera
Chimaera = Planet()
Chimaera.name = 'Chimaera' #Name of the planet
Chimaera.R = 22.045e3 #Mean radius of planet [m] 
Chimaera.g = 0.006 #Surface gravitational acceleration [m.s-2]
Chimaera.S = 225 #Annual mean solar constant [W.m-2]
Chimaera.albedo = 0.01 #Bond albedo # Roughly from geometric albedo 0.037 (SBD), assuming phase integral of 0.25, also from Paul's entry 
Chimaera.emissivity = 0.95 #IR emissivity
Chimaera.Qb = 0.0 #basal heat flow [W.m-2]

# Thermophysical properties: # Going off thermal inertia of ~100 from delbo et al 2007 size gamma plot, but also running lunar 
Chimaera.ks = Moon.ks # Solid (phonon) conductivity at surface [W.m-1.K-1] 
Chimaera.kd = Moon.kd # Solid (phonon) conductivity at depth z>>H [W.m.K-1] 
Chimaera.rhos = Moon.rhos # Density at surface [kg.m-3] # From Harish's doc (XC/CM type), although Paul had 2940 in for this 
Chimaera.rhod = Moon.rhod # Density at depth z>>H [kg.m-3] # Paul had 2940 in for this 
#
Chimaera.H = Moon.H # e-folding scale of conductivity and density [m]
Chimaera.cp0 = Moon.cp0 # heat capacity at average surface temp. [J.kg.K-1]
Chimaera.cpCoeff = Moon.cpCoeff # Heat capacity polynomial coefficients
#        
Chimaera.rsm = 3.683e11  #Semi-major axis [m] 
Chimaera.rAU = 	Chimaera.rsm / AU    #Semi-major axis [AU] 
Chimaera.year = 1.22e8  #Sidereal length of year [s]
Chimaera.eccentricity = 0.1114  # Eccentricity
Chimaera.day = 52686  #Mean length of solar day [s]
Chimaera.obliquity = np.radians(78.31)  #Obliquity to plane of ecliptic [radians]
Chimaera.Lequinox = None  #Longitude of equinox
Chimaera.Lp = np.radians(308.28+124.88)  #Longitude of perihelion [radians] equal to the sum of the longitude of the ascending node + argument of periapsis 
#
Chimaera.Tsavg = 165. #Mean surface temperature [K]
Chimaera.Tsmax = 264. #Maximum surface temperature [K]




# (13294) Rockox
Rockox = Planet()
Rockox.name = 'Rockox' #Name of the planet
Rockox.R = 2619. #Mean radius of planet [m] 
Rockox.g = 1.0e-5 #Surface gravitational acceleration [m.s-2]
Rockox.albedo = 0.03475 #Bond albedo # From geometric albedo 0.139 (SBD), assuming phase integral of 0.25
Rockox.emissivity = 0.95 #IR emissivity
Rockox.Qb = 0.0 #basal heat flow [W.m-2]

# Thermophysical properties: # Going off thermal inertia of ~100 from delbo et al 2007 size gamma plot 
Rockox.ks = 0.00811 # Solid (phonon) conductivity at surface [W.m-1.K-1] 
Rockox.kd = 0.00811 # Solid (phonon) conductivity at depth z>>H [W.m.K-1]
Rockox.rhos = 2056. # Density at surface [kg.m-3] # From Harish's doc (OC type)
Rockox.rhod = 2056. # Density at depth z>>H [kg.m-3]
#
Rockox.H = Moon.H # e-folding scale of conductivity and density [m]
Rockox.cp0 = Moon.cp0 # heat capacity at average surface temp. [J.kg.K-1]
Rockox.cpCoeff = Moon.cpCoeff # Heat capacity polynomial coefficients
#         
Rockox.rAU = 	2.258    #Semi-major axis [AU]
Rockox.rsm = 3.377968e11  #Semi-major axis [m] 
Rockox.year = 1.07e8  #Sidereal length of year [s]
Rockox.eccentricity = 0.098  # Eccentricity
Rockox.day = 11675  #Mean length of solar day [s]
Rockox.obliquity = 0 #np.radians(Rockox.calcObliquity(0, 0, 5.7365, 48.70465))  #Obliquity to plane of ecliptic [radians]
Rockox.Lequinox = None  #Longitude of equinox
Rockox.Lp = np.radians(149.29 + 48.70465)  #Longitude of perihelion [radians]
#
Rockox.S = 267. #Annual mean solar constant [W.m-2]
Rockox.Tsavg = 185. #Mean surface temperature [K]
Rockox.Tsmax = 275. #Maximum surface temperature [K]




#  (88055) 2000 VA28
VA28_2000 = Planet()
VA28_2000.name = '2000VA28' #Name of the planet
VA28_2000.R = 2.721e3 #Mean radius of planet [m] 
VA28_2000.g = 1.0e-5 #Surface gravitational acceleration [m.s-2]
VA28_2000.albedo = 0.013 #Bond albedo # From geometric albedo 0.054 (SBD), assuming phase integral of 0.25
VA28_2000.emissivity = 0.95 #IR emissivity
VA28_2000.Qb = 0.0 #basal heat flow [W.m-2]

# Thermophysical properties: Going off thermal inertia of ~100 from delbo et al 2007 size gamma plot 
VA28_2000.ks = 0.016667 # Solid (phonon) conductivity at surface [W.m-1.K-1] 
VA28_2000.kd = 0.016667 # Solid (phonon) conductivity at depth z>>H [W.m.K-1]
VA28_2000.rhos = 1000. # Density at surface [kg.m-3] # No source, placeholder used to get Gamma = 100 
VA28_2000.rhod = 1000. # Density at depth z>>H [kg.m-3]
#
VA28_2000.H = Moon.H # e-folding scale of conductivity and density [m]
VA28_2000.cp0 = Moon.cp0 # heat capacity at average surface temp. [J.kg.K-1]
VA28_2000.cpCoeff = Moon.cpCoeff # Heat capacity polynomial coefficients
#         
VA28_2000.rAU = 2.735    #Semi-major axis [AU]
VA28_2000.rsm = 4.09156e11  #Semi-major axis [m] 
VA28_2000.year = 1.4257e8  #Sidereal length of year [s]
VA28_2000.eccentricity = 0.1045 # Eccentricity
VA28_2000.day = 36000  #Mean length of solar day [s] # UNKNOWN
VA28_2000.obliquity = 0 #np.radians(VA28_2000.calcObliquity(0, 0, 6.736, 106.133))  #Obliquity to plane of ecliptic [radians]
VA28_2000.Lequinox = None  #Longitude of equinox
VA28_2000.Lp = np.radians(106.133 + 266.68)  #Longitude of perihelion [radians], equal to the sum of the longitude of the ascending node + argument of periapsis 
#
VA28_2000.S = 181. #Annual mean solar constant [W.m-2]
VA28_2000.Tsavg = 170#Mean surface temperature [K]
VA28_2000.Tsmax = 260 #Maximum surface temperature [K]



#  (23871)1998 RC76 (NOT DONE AT ALL)
RC76_1998 = Planet()
RC76_1998.name = '1998RC76' #Name of the planet
RC76_1998.R = 3.3415e3 #Mean radius of planet [m] 
RC76_1998.g = 1.0e-5 #Surface gravitational acceleration [m.s-2]
RC76_1998.albedo = 0.01275 #Bond albedo # From geometric albedo 0.051 (SBD), assuming phase integral of 0.25
RC76_1998.emissivity = 0.95 #IR emissivity
RC76_1998.Qb = 0.0 #basal heat flow [W.m-2]

# Thermophysical properties: Going off thermal inertia of ~100 from delbo et al 2007 size gamma plot 
RC76_1998.ks = 0.0093110 # Solid (phonon) conductivity at surface [W.m-1.K-1] 
RC76_1998.kd = 0.0093110 # Solid (phonon) conductivity at depth z>>H [W.m.K-1]
RC76_1998.rhos = 1790. # Density at surface [kg.m-3] # From Harish's doc (OC type)
RC76_1998.rhod = 1790. # Density at depth z>>H [kg.m-3]
#
RC76_1998.H = Moon.H # e-folding scale of conductivity and density [m]
RC76_1998.cp0 = Moon.cp0 # heat capacity at average surface temp. [J.kg.K-1]
RC76_1998.cpCoeff = Moon.cpCoeff # Heat capacity polynomial coefficients
#         
RC76_1998.rAU = 2.385    #Semi-major axis [AU]
RC76_1998.rsm = RC76_1998.rAU * AU  #Semi-major axis [m] 
RC76_1998.year = 1.162077e8  #Sidereal length of year [s]
RC76_1998.eccentricity = 0.222  # Eccentricity
RC76_1998.day = 36000  #Mean length of solar day [s] # UNKNOWN
RC76_1998.obliquity = 0 #np.radians(RC76_1998.calcObliquity(0, 0, 4.996, 149.139))  #Obliquity to plane of ecliptic [radians]
RC76_1998.Lequinox = None  #Longitude of equinox
RC76_1998.Lp = np.radians(149.139 + 131.049)  #Longitude of perihelion [radians] (Small Bodies Database),  sum of the longitude of the ascending node + argument of periapsis  
#
RC76_1998.S = 240 #Annual mean solar constant [W.m-2]
RC76_1998.Tsavg = 185 #Mean surface temperature [K]
RC76_1998.Tsmax = 300#Maximum surface temperature [K]




#  (23871) 1999 SG6 (NOT DONE AT ALL)
SG6_1999 = Planet()
SG6_1999.name = '1999SG6' #Name of the planet
SG6_1999.R = 3.9845e3 #Mean radius of planet [m] 
SG6_1999.g = 1.0e-5 #Surface gravitational acceleration [m.s-2]
SG6_1999.albedo = 0.03025 #Bond albedo # From geometric albedo 0.121 (SBD), assuming phase integral of 0.25
SG6_1999.emissivity = 0.95 #IR emissivity
SG6_1999.Qb = 0.0 #basal heat flow [W.m-2]

# Thermophysical properties: Going off thermal inertia of ~100 from delbo et al 2007 size gamma plot 
SG6_1999.ks = 0.0062073 # Solid (phonon) conductivity at surface [W.m-1.K-1] 
SG6_1999.kd = 0.0062073 # Solid (phonon) conductivity at depth z>>H [W.m.K-1]
SG6_1999.rhos = 2685. # Density at surface [kg.m-3] # From Harish's doc (OC type)
SG6_1999.rhod = 2685. # Density at depth z>>H [kg.m-3]
#
SG6_1999.H = Moon.H # e-folding scale of conductivity and density [m]
SG6_1999.cp0 = Moon.cp0 # heat capacity at average surface temp. [J.kg.K-1]
SG6_1999.cpCoeff = Moon.cpCoeff # Heat capacity polynomial coefficients
#         
SG6_1999.rAU = 	3.054   #Semi-major axis [AU]
SG6_1999.rsm = SG6_1999.rAU * AU  #Semi-major axis [m] 
SG6_1999.year = 1.68493e8  #Sidereal length of year [s]
SG6_1999.eccentricity = 0.031  # Eccentricity
SG6_1999.day = 8172  #Mean length of solar day [s]
SG6_1999.obliquity = np.radians(SG6_1999.calcObliquity(67, 37, 11.758, 142.582))  #Obliquity to plane of ecliptic [radians]
SG6_1999.Lequinox = None  #Longitude of equinox
SG6_1999.Lp = np.radians(142.582 + 308.67241)  #Longitude of perihelion [radians],  sum of the longitude of the ascending node + argument of periapsis 
#
SG6_1999.S = 146 #Annual mean solar constant [W.m-2]
SG6_1999.Tsavg = 160 #Mean surface temperature [K]
SG6_1999.Tsmax = 235 #Maximum surface temperature [K]



# Small body secondaries

#--------------------------------------------------------------------------


# 1991 VH Secondary
VH1991_Second = Planet()
VH1991_Second.name = 'VH1991_Second' #Name of the planet
VH1991_Second.R = 210. #Mean radius of planet [m]
VH1991_Second.g = 1.0e-5 #Surface gravitational acceleration [m.s-2]
VH1991_Second.S = 1072.7 #Annual mean solar constant [W.m-2]
VH1991_Second.albedo = 0.0475 #Bond albedo
VH1991_Second.emissivity = 0.95 #IR emissivity
VH1991_Second.Qb = 0.0 #basal heat flow [W.m-2]
#Thermophysical properties:
# Currently Bennu-like (Assuming Thermal inertia of ~303.3 from Emery et al 2019, Rozitis et al 2019)
VH1991_Second.ks = 0.1288 # Solid (phonon) conductivity at surface [W.m-1.K-1]
                          # 0.1288  #Bennu-like (Assuming Thermal inertia of ~303.3 from Emery et al 2019, Rozitis et al 2019)
VH1991_Second.kd = 0.1288 # Solid (phonon) conductivity at depth z>>H [W.m.K-1]
VH1991_Second.rhos = 1190 # Density at surface [kg.m-3]
VH1991_Second.rhod = 1190 # Density at depth z>>H [kg.m-3]
VH1991_Second.H = Moon.H  # e-folding scale of conductivity and density [m]
VH1991_Second.cp0 = Moon.cp0 # heat capacity at average surface temp. [J.kg.K-1]
VH1991_Second.cpCoeff = Moon.cpCoeff # Heat capacity polynomial coefficients
#       
VH1991_Second.rsm = 1.9149e11   #Semi-major axis [m] 
                                #1.4561e11 standard, 1.4561e+11 perihelion, 1.9471e+11 aphelion, 1.9149e11 intersect
VH1991_Second.rAU = 1.28     #Semi-major axis [AU] 
                                #1.137 standard, 0.97337 perihelion, 1.30154 aphelion, 1.28 intersect
VH1991_Second.year = 4.57e7     #Sidereal length of year [s]
VH1991_Second.eccentricity = 0 #0.144  # Eccentricity
VH1991_Second.day = 9445.32     #Mean length of solar day [s]
VH1991_Second.obliquity = 0     #Obliquity to plane of ecliptic [radians]
VH1991_Second.Lequinox = None   #Longitude of equinox
VH1991_Second.Lp = 0.           #Longitude of perihelion [radians]
#
VH1991_Second.Tsavg = 270. #Mean surface temperature [K]
VH1991_Second.Tsmax = 400. #Maximum surface temperature [K]


# 1996 FG3 Secondary
FG31996_Second = Planet()
FG31996_Second.name = 'FG31996_Second' #Name of the planet
FG31996_Second.R = 245.          #Mean radius of planet [m]
FG31996_Second.g = 1.0e-5        #Surface gravitational acceleration [m.s-2]
FG31996_Second.S = 1072.7        #Annual mean solar constant [W.m-2]
FG31996_Second.albedo =  0.011    #Bond albedo (one fourth of geometric albedo = 0.044)
FG31996_Second.emissivity = 0.95 #IR emissivity
FG31996_Second.Qb = 0.0          #basal heat flow [W.m-2]
#Thermophysical properties:
# Currently Bennu-like (Assuming Thermal inertia of ~303.3 from Emery et al 2019, Rozitis et al 2019)
FG31996_Second.ks = 0.1260504 # Solid (phonon) conductivity at surface [W.m-1.K-1]
FG31996_Second.kd = 0.1260504 # Solid (phonon) conductivity at depth z>>H [W.m.K-1]
FG31996_Second.rhos = 1190 # Density at surface [kg.m-3]
FG31996_Second.rhod = 1190 # Density at depth z>>H [kg.m-3]
FG31996_Second.H = Moon.H  # e-folding scale of conductivity and density [m]
FG31996_Second.cp0 = Moon.cp0 # heat capacity at average surface temp. [J.kg.K-1]
FG31996_Second.cpCoeff = Moon.cpCoeff # Heat capacity polynomial coefficients 
#
FG31996_Second.rsm = 1.685e11   #Semi-major axis [m] 
                                #1.685e11 standard, 1.025e+11 perihelion, 2.128e+11 aphelion, 2.124e11 intersect
FG31996_Second.rAU = 1.054     #Semi-major axis [AU] 
                                #1.054 standard, 0.6853 at perihelion, 1.4223 at aphelion, 1.42 intersect 
FG31996_Second.year = 3.41412e7#5.34e7    #Sidereal length of year [s]
FG31996_Second.eccentricity = 0.349 # Eccentricity
FG31996_Second.day = 58143    #Mean length of solar day [s]
FG31996_Second.obliquity = -3.0542 #-1.44862328    #Obliquity to plane of ecliptic [radians] obliq = dec - 90 - inclination
FG31996_Second.Lequinox = None  #Longitude of equinox
FG31996_Second.Lp = 0.          #Longitude of perihelion [radians]
#
FG31996_Second.Tsavg = 270. #Mean surface temperature [K]
FG31996_Second.Tsmax = 400. #Maximum surface temperature [K]

FG31996_Second.separation = 2.46 # Binary separation [km]
FG31996_Second.secondaryMass = 6.20e10 # Secondary mass [kg] derived by Janus team on DRB
FG31996_Second.secondaryVolume = 6.8e7 # Secondary volume [m^3]. Modeled by Lance Benner, from Janus 2020 DRB
FG31996_Second.radiusSphere = (FG31996_Second.secondaryVolume / (4/3 * np.pi))**(1/3) # Radius of volume equivalent sphere [m]


# Didymos B
Didb = Planet()
Didb.name = 'Dimorphos' #Name of the planet
Didb.R = 74.9 #Mean radius of planet [m]
Didb.g = 1.0e-5 #Surface gravitational acceleration [m.s-2]
Didb.S = 1072.7 #Annual mean solar constant [W.m-2]
Didb.albedo = 0.08 #Bond albedo #0.07
Didb.emissivity = 0.95 #IR emissivity
Didb.Qb = 0.0 #basal heat flow [W.m-2]
#Thermophysical properties:
#Sb type 
Didb.ks = 0.1920 # Solid (phonon) conductivity at surface [W.m-1.K-1]
Didb.kd = 0.1920 # Solid (phonon) conductivity at depth z>>H [W.m.K-1]
Didb.rhos = 2790 # Density at surface [kg.m-3]
Didb.rhod = 2790 # Density at depth z>>H [kg.m-3]
Didb.H = Moon.H # e-folding scale of conductivity and density [m]
Didb.cp0 = Moon.cp0 # heat capacity at average surface temp. [J.kg.K-1]
Didb.cpCoeff = Moon.cpCoeff # Heat capacity polynomial coefficients
#        
Didb.rsm = 2.45739195e11  #Semi-major axis [m] 
Didb.rAU = 1.642665059   #Semi-major axis [AU] 
Didb.year = 6.653e7  #Sidereal length of year [s]
Didb.eccentricity = 0.383264789  # Eccentricity
Didb.day = 42917.37  #Mean length of solar day [s] (pre impact) 
Didb.obliquity = 3.054  #Obliquity to plane of ecliptic [radians]
Didb.Lequinox = None  #Longitude of equinox
Didb.Lp = 0.  #Longitude of perihelion [radians]
#
Didb.Tsavg = 270. #Mean surface temperature [K]
Didb.Tsmax = 450. #Maximum surface temperature [K]

Didb.separation = 1.189 # Binary separation [km]

Didb.radiusSphere = 149.8 / 2. # [m]
Didb.secondaryVolume = (4. /3.) * np.pi * Didb.radiusSphere**3 # Secondary volume [m^3]. 
Didb.secondaryMass = Didb.rhos * Didb.secondaryVolume # Secondary mass [kg]

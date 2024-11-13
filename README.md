# BinaryThermophysicalModel
A 3D thermophysical model for interacting binary asteroid pairs, known as the Binary Thermophysical Model (BTM). The BTM utilizes triangulated meshes to calculate temperatures for the surface and near subsurface of airless bodies. It is capable of and has been utilized for: 
  1) Single asteroids, such as Bennu, for which a shape model exists
  2) Digital Elevation Models (DEMs) and Digital Terrain Models (DTMs)
  3) 3D interacting binary pairs, including eclipses and moonshine.

The model is composed of separate modularized scripts that each are responsible for different components. For example, the shapeModule.py file imports shape models and initializes a "shape" instance. This defines facets, normals, meshes and other properties used by the model later. Is is also used to move secondary shape models into their orbital location, as most shape models are cemtered at the origin. Specific parameters, such as separation between the primary and secondary, are entered in the planets.py file or can be manually changed when calculating shadows. Other modules include those handling BYORP, orbit information and advancement, ray tracing, etc. The BTM itself is run through the User interaction should be limited with these. Scripts that users will need to interact with are described below. 

1) Planet parameters: Handled by planets.py. For each body that the user is attempting to model, an entry must be made in this file. These entries contain orbital, compositional and other pertinent information. Several entries already exist, and new ones can be added. Users can select a planet in other scripts by defining an instance of the planet class such as planet = planets.Moon. This specific parameters can then be altered in the external script. A planet instance must be fed to the model.
2) Shadowing Calculation. Users should interact with the preCalculateShadows.py script, as functions are stored in shadowModule.py. For binary models, it is suggested that users pre-calculate shadowing files. These can be used during equilibration for all runs with a given set of parameters, saving significant computational time. To use, edit the information in lines 31-77 of preCalculateShadows.py. Choose whether you want a single body or binary, enter file paths and select the shadowing resolution you want. Unless specified, obliquity is 0.0 degrees. Though pre-calulation is recommended for binaries, shadowModule.py holds functions used by the model to calculate custom exact shadows at each timestep for the current system geometry. An option has been added that will do this after temperature equilibration to give exact temperatures during readout. This is recommended for any BYORP calculation, as BYORP is very sensitive to changes in temperature and shadow. To enable this, choose calcShadows = True when initializing the model. 
6) View Factor Calculation: Functions for the view factor calculation are contained in visibilityFunction.py, but viewFactorCalculation.py should be edited and run by user. To use, edit lines 55-56 to choose which model you want to run (Single or Binary). Then enter the shape models, file paths for results and number of divisions of the orbit you want to run (labeled steps). For binary systems, enter the planet as well in order to include the mutual orbit separation. This can also be entered directly. Do not alter the multiprocessing component that follows, unless you want to reduce the number of cores the model runs on. This can be done by switching p = mp.Pool(multiprocessing.cpu_count()) to p = mp.Pool(# of cores). 
7) Model run script. This should be generated by the user. A sample script has been provided for a binary system. A single body model requries a similar but simpler implementation, using the heat3d_Single.py script. Enter the paths for shape models, view factors and shadows, as well as the parameters the model should be run with (described in script). A planet must be selected for each body that will be modeled. Timing is enabled, and will print how long the model takes to run. 


For questions, please see Sorli et al. (2024) (In prep), or direct questions to Kya Sorli (kya.sorli@colorado.edu). 

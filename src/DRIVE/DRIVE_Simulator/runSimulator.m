% RUNSIMULATOR Runs the simulation. 
%  Loads the settings from simSettings.m and runs the simulator. 
%
% Usage: runSimulation
%
% Copyright (c) 2019-2020, Ioannis Mavromatis
% email: ioan.mavromatis@bristol.ac.uk
% email: ioannis.mavromatis@toshiba-bril.com

clc; clf; clear; clear global; close all;

fprintf('DRIVE: Digital Twin for self-dRiving Intelligent VEhicles\n');
fprintf('Copyright (c) 2019-2020, Ioannis Mavromatis\n');
fprintf('email: ioan.mavromatis@bristol.ac.uk\n');
fprintf('email: ioannis.mavromatis@toshiba-bril.com\n\n');

% Load the simulation settings (for further explanation see simSettings.m)
simSettings;

% Add the different modules of the simulator to MATLAB path
setupSimulator;

% Load LAPACK&BLAS libraries
setenv('BLAS_VERSION', 'mkl.dll')
setenv('LAPACK_VERSION', 'mkl.dll')

% Main function (for further explanation see runMain.m)

runMain(map,sumo,BS,linkBudget);


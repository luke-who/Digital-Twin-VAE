# [MATLAB Search Path](https://uk.mathworks.com/help/matlab/matlab_env/what-is-the-matlab-search-path.html#responsive_offcanvas) - userpath Folder on the Search Path
On Windows OS, it is recommended that the [userpath](https://uk.mathworks.com/help/matlab/ref/userpath.html) folder is set to the default according to the OS
(if the default isn't any one of these shown below):

* Windows® platforms — %USERPROFILE%/Documents/MATLAB.
* Mac platforms — $home/Documents/MATLAB.
* Linux® platforms — $home/Documents/MATLAB if $home/Documents exists.
* MATLAB Online™ — /users/youruserid.

Example to set the default `userpath` on windows 10 MATLAB:
```
newpath = 'C:\Users\<your_user_name>\Documents\MATLAB';
userpath(newpath)
```
View the effect of the change on the search path:

```
userpath
```

# Potential issues when running [SUMO](https://sumo.dlr.de/docs/Installing/index.html) 
Ensure `$SUMO_HOME` [environmental variable is configured correctly](https://sumo.dlr.de/docs/Basics/Basic_Computer_Skills.html#sumo_home)
(If you have installed SUMO via the windows *.msi* installer file, this is done automatically.)

# Potential issues when running [traci4matlab](https://github.com/pipeacosta/traci4matlab)
## Could not connect to TraCI server at 127.0.0.1:≭. Java exception occurred
### Solution(Windows):
Make sure TraCI4Matlab is added to the directory path in the MATLAB's default userpath folder (instructions can also be found in 2.2.4 [user_manual.pdf](https://github.com/pipeacosta/traci4matlab/blob/master/user_manual.pdf)):
```
pathtool
```
*Add Folder...* -> Select "*C:\Users\<your_user_name>\Documents\MATLAB\traci4matlab*" -> *Save* -> *Close*
    
# Potential issues when running [DRIVE_Simulator](https://github.com/ioannismavromatis/DRIVE_Simulator)
## [LAPACK/BLAS loading error](https://uk.mathworks.com/matlabcentral/answers/269035-hot-to-fix-lapack-blas-loading-error)
### Solution:
Enter two lines of code in the MATLAB Command Window: 
```
setenv('BLAS_VERSION', 'mkl.dll');
setenv('LAPACK_VERSION', 'mkl.dll');
```
(Alternatively add them to [runSimulator.m](https://github.com/ioannismavromatis/DRIVE_Simulator/blob/master/runSimulator.m)).
Then you can verify the two libraries have been loaded: 
```
version -blas;
version -lapack;
```
## Warning: Escaped character '\P' is not valid. See 'doc sprintf' for supported special characters.
This is because the `SIMULATOR.sumoPath` variable in [*/DRIVE_Simulator/simSettings.m*](https://github.com/ioannismavromatis/DRIVE_Simulator/blob/master/simSettings.m) is not set correctly
### Example Solution:
```
SIMULATOR.sumoPath = 'C:\Program Files\Eclipse\Sumo\bin\' %the sumo's bin path depends on its installation
```

## Error in height (line 10) H = size(X,1); Error in loadRATs (line 84) BS.(ratName).height = height;
This is because the *[height](https://uk.mathworks.com/help/matlab/ref/height.html#mw_0c0894e1-3181-4045-923b-45aab2f657d4)* function returns the number of rows of an input array in newer version of MATLAB(changed in R2020b). In the original version of DRIVE, the variable that defines mmWave basestation hight range ([*DRIVE_Simulator/ratToolbox/available/mmWaves.m*](https://github.com/ioannismavromatis/DRIVE_Simulator/blob/440cbab1e6f1d4c6b0f28f074b02c4fce0379ee0/ratToolbox/available/mmWaves.m)) is also `height`. This causes a conflict in MATLAB as it thinks we're trying to call the *[height](https://uk.mathworks.com/help/matlab/ref/height.html#mw_0c0894e1-3181-4045-923b-45aab2f657d4)* function instead of our defined variable `height`.
### Solution:
Simply change the variable to a different name (ie `height` -> `mmWaveheight`):
```
mmWaveheight = [ 5 15 ];
```

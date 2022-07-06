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
1. Make sure SUMO is correctly installed
2. Ensure [`$SUMO_HOME` environmental variable is configured correctly](https://sumo.dlr.de/docs/Basics/Basic_Computer_Skills.html#sumo_home)
(If you have installed SUMO via the windows *.msi* installer file, this is done automatically.)

# Potential issues when running [traci4matlab](https://github.com/pipeacosta/traci4matlab)
1. Download/Clone TraCI4Matlab from the official GitHub repository and add the directory in the MATLAB path (instructions can also be found in 2.2.4 [user_manual.pdf](https://github.com/pipeacosta/traci4matlab/blob/master/user_manual.pdf)):
    ```
    pathtool
    ```
    On windows *Add Folder...* -> Select "*C:\Users\<your_user_name>\Documents\MATLAB\traci4matlab*" -> *Save* -> *Close*

3. Add TraCI4Matlab additional dependencies to the MATLAB’s static JAVA path. To do so:
    * Create a text file containing the path to traci4matlab.jar, e.g.: *TRACI4MATLAB_HOME/traci4matlab.jar*(*TRACI4MATLAB_HOME* is the path to the root folder of TraCI4Matlab obtained in the previous step.)
    * Save this file as “javaclasspath.txt” in the preferences directory of MATLAB. You can identify this folder running `prefdir` command inside MATLAB.
    * Restart MATLAB, so the new static path takes effect.
# Potential issues when running [DRIVE_Simulator](https://github.com/ioannismavromatis/DRIVE_Simulator)
1. Download or clone DRIVE from the official GitHub Repository
2. Compile the required external libraries as MEX files. At the moment one external library needs to be compiled, i.e., *segment_intersection* Toolbox.
    To do that, from the root directory of DRIVE, navigate inside *./externalToolbox/segment_intersection/* and run the following commands on MATLAB’s command window:
    ```
    mex -v segment_intersect_test.c
    mex -v segments_intersect_test.c
    mex -v segments_intersect_test_vector.c
    ```

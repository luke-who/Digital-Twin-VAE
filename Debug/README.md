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
Enter two lines of code in the MATLAB Command Window, the 1st: 
```
setenv('BLAS_VERSION', 'mkl.dll')
```
and the 2nd: 
```
setenv('LAPACK_VERSION', 'mkl.dll')
```

Then you can verify the two libraries have been loaded: 
```
version -blas;version -lapack
```

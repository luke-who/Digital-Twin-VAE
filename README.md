# Digital Twin and Variational Autoencoder Integrated Neural Network for RSRP Prediction
-----------------------------------------------------------------------------------
<!--
User Equipment Distribution             |  UE with regard to BS
:-------------------------:|:-------------------------:
![](https://github.com/luke-who/Digital-Twin-4th-Year/blob/main/asset/UE-Distribution.png)  |  ![](https://github.com/luke-who/Digital-Twin-4th-Year/blob/main/asset/UE-BS.gif)
-->

<p float="center">
  <img src="https://github.com/luke-who/Digital-Twin-4th-Year/blob/main/asset/UE-Distribution.png" height="362" width = auto/>
  <img src="https://github.com/luke-who/Digital-Twin-4th-Year/blob/main/asset/UE-BS.gif" height="362" width = auto/> 
</p>

<!-- Note the spaces in the next line for Figs are special Em Space (U+2003)-->
       **Fig 1: User Equipment Distribution            Fig 2: UE with regard to BS**

[![python](https://img.shields.io/badge/python-3.9.8-blue?style=plastic&logo=python)](https://www.python.org/downloads/release/python-398/)
[![pip](https://img.shields.io/badge/pip-v21.2.4-informational?&logo=pypi)](https://pypi.org/project/pip/21.2.4/)
[![matlab](https://img.shields.io/badge/MATLAB-R2022a-E88D37?&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciICB2aWV3Qm94PSIwIDAgNDggNDgiIHdpZHRoPSI0OHB4IiBoZWlnaHQ9IjQ4cHgiPjxsaW5lYXJHcmFkaWVudCBpZD0iWjhiRzg5VG5aVzh+QndKanpxbW5YYSIgeDE9IjIyLjY0NSIgeDI9IjI2Ljc1NyIgeTE9IjEwLjg4MSIgeTI9IjIzLjg1NCIgZ3JhZGllbnRVbml0cz0idXNlclNwYWNlT25Vc2UiPjxzdG9wIG9mZnNldD0iMCIgc3RvcC1jb2xvcj0iIzRhZGRkZiIvPjxzdG9wIG9mZnNldD0iLjY5OSIgc3RvcC1jb2xvcj0iIzNmNTM1MiIvPjxzdG9wIG9mZnNldD0iLjg2MyIgc3RvcC1jb2xvcj0iIzQ0MjcyOSIvPjwvbGluZWFyR3JhZGllbnQ+PHBhdGggZmlsbD0idXJsKCNaOGJHODlUblpXOH5Cd0pqenFtblhhKSIgZD0iTTIxLDI3bC03LTZjMCwwLDEtMS41LDIuNS0zczIuNzM2LTEuODUyLDQuNS0zYzMuNTExLTIuMjg0LDYuNS0xMiwxMS0xMkwyMSwyN3oiLz48bGluZWFyR3JhZGllbnQgaWQ9Ilo4Ykc4OVRuWlc4fkJ3Smp6cW1uWGIiIHgxPSIxIiB4Mj0iMzcuNzc1IiB5MT0iMjcuMDMzIiB5Mj0iMjcuMDMzIiBncmFkaWVudFVuaXRzPSJ1c2VyU3BhY2VPblVzZSI+PHN0b3Agb2Zmc2V0PSIwIiBzdG9wLWNvbG9yPSIjNGFkZGRmIi8+PHN0b3Agb2Zmc2V0PSIuNzkyIiBzdG9wLWNvbG9yPSIjM2Y1MzUyIi8+PHN0b3Agb2Zmc2V0PSIxIiBzdG9wLWNvbG9yPSIjNDQyNzI5Ii8+PC9saW5lYXJHcmFkaWVudD48cG9seWdvbiBmaWxsPSJ1cmwoI1o4Ykc4OVRuWlc4fkJ3Smp6cW1uWGIpIiBwb2ludHM9IjExLDMzLjA2NiAxLDI2IDE0LDIxIDIxLjI3NywyNi40NjUgMTQsMzIuMDY2Ii8+PGxpbmVhckdyYWRpZW50IGlkPSJaOGJHODlUblpXOH5Cd0pqenFtblhjIiB4MT0iMTEiIHgyPSI0NyIgeTE9IjI0IiB5Mj0iMjQiIGdyYWRpZW50VW5pdHM9InVzZXJTcGFjZU9uVXNlIj48c3RvcCBvZmZzZXQ9Ii4yMDYiIHN0b3AtY29sb3I9IiM1MzE0MGYiLz48c3RvcCBvZmZzZXQ9Ii4zIiBzdG9wLWNvbG9yPSIjODQzNjBmIi8+PHN0b3Agb2Zmc2V0PSIuNDEzIiBzdG9wLWNvbG9yPSIjYjg1YjEwIi8+PHN0b3Agb2Zmc2V0PSIuNTExIiBzdG9wLWNvbG9yPSIjZGY3NjEwIi8+PHN0b3Agb2Zmc2V0PSIuNTkiIHN0b3AtY29sb3I9IiNmNjg3MTAiLz48c3RvcCBvZmZzZXQ9Ii42MzkiIHN0b3AtY29sb3I9IiNmZjhkMTAiLz48c3RvcCBvZmZzZXQ9Ii43MjkiIHN0b3AtY29sb3I9IiNmZDhhMTAiLz48c3RvcCBvZmZzZXQ9Ii44IiBzdG9wLWNvbG9yPSIjZjU4MDEwIi8+PHN0b3Agb2Zmc2V0PSIuODY1IiBzdG9wLWNvbG9yPSIjZTg2ZjEwIi8+PHN0b3Agb2Zmc2V0PSIuOTI1IiBzdG9wLWNvbG9yPSIjZDY1ODExIi8+PHN0b3Agb2Zmc2V0PSIuOTgyIiBzdG9wLWNvbG9yPSIjYzAzYTExIi8+PHN0b3Agb2Zmc2V0PSIxIiBzdG9wLWNvbG9yPSIjYjcyZjExIi8+PC9saW5lYXJHcmFkaWVudD48cGF0aCBmaWxsPSJ1cmwoI1o4Ykc4OVRuWlc4fkJ3Smp6cW1uWGMpIiBkPSJNMzIsM2M1LDAsMTMsMjcsMTUsMzRjMCwwLTcuMDE3LTYuNjMtMTEtNnMtNS40Nyw2LjU0OC05LjcyNSwxMC43NTZDMjMuNSw0NC41LDIxLDQ1LDIxLDQ1CXMtMC4yMDYtOC4xMjQtNS0xMWMtMi41LTEuNS01LTEtNS0xczYuMDQ5LTIuOTAxLDkuNDc0LTguMTc0UzI4LjUsMywzMiwzeiIvPjwvc3ZnPg==)](https://uk.mathworks.com/downloads/)
<!-- [![matlab](asset/matlab_2021a.svg)](https://uk.mathworks.com/downloads/) -->

[![PyTorch](https://img.shields.io/badge/PyTorch-1.10-orange?logo=PyTorch)](https://github.com/pytorch/pytorch/releases/tag/v1.10.0)
[![macOS Monterey 12.0](https://img.shields.io/badge/macOS%20Monterey-12.0-megenta?&color=E95420&logo=macOS)](https://www.apple.com/uk/macos/monterey/)
[![build running](https://img.shields.io/badge/build-passing-brightgreen)](#)

## Motivation 
Design and develop a scalable and lightweight signal strength prediction model with the help of a DT([DRIVE](https://github.com/ioannismavromatis/DRIVE_Simulator)) and VAE to improve prediction accuracy for the existing mobile networks.
## Aim
This project aims to explore the use of a 2-stage Neural Network (NN) which consists of a Variational Auto-Encoder(VAE) and another NN for Reference Signal Received Power (RSRP) prediction to improve QoS (Quality of Service).

## Objectives
* Explore the use of an existing DT (self-dRiving Intelligent Vehicles (DRIVE)) and modify it to generate synthetic data which combines both real-world data and spatial data from Open Stree Map (OSM).
* Develop a first stage model with a VAE architecture, train it with the generated synthetic data as input and make use of model optimisation techniques such as data augmentation and normalisation to improve the converge time and generalisation of the model
* Extract the computed environmental features from the encoder part of VAE after training the VAE.
* Integrate the extracted environmental features and real-world data into a second stage model for training.
* Use only real-world data to train a MLP in parallel, with the same hidden layers as the second part of the 2-stage NN for comparison.
* Evaluate the performance of the trained 2-stage NN against the trained MLP in terms of MAE in RSRP prediction.

# Getting Started
-----------------------------------------------------------------------------------
<!-- TODO: Guide users through getting your code up and running on their own system. In this section you can talk about: -->

## 1.    Installation process
### Install Pytorch
1. Install the Python development environment on your system

    `pip3 install torch torchvision`

### Install [DRIVE_Simulator](https://github.com/ioannismavromatis/DRIVE_Simulator)
The installation process for DRIVE can be found in [userManualDRIVE.pdf](https://github.com/ioannismavromatis/DRIVE_Simulator/blob/master/doc/userManualDRIVE.pdf)

<!--
1. Install 

    ``

2. Test DRIVE

    ``
-->

### Install [traci4matlab](https://github.com/pipeacosta/traci4matlab)
See instructions [here](https://github.com/pipeacosta/traci4matlab/blob/master/user_manual.pdf)

## 2.    Software dependencies
### Add-Ons Requirements for [DRIVE_Simulator](https://github.com/ioannismavromatis/DRIVE_Simulator)
* [Parallel Computing Toolbox](https://uk.mathworks.com/help/parallel-computing/getting-started-with-parallel-computing-toolbox.html) Perform parallel computations on multicore computers, GPUs, and computer clusters
* [Mapping Toolbox](https://uk.mathworks.com/help/map/index.html) Analyze and visualize geographic information
* [Statistics and Machine Learning Toolbox](https://uk.mathworks.com/help/stats/) Analyze and model data using statistics and machine learning
* [MATLAB Support for MinGW-w64 C/C++ Compiler](https://uk.mathworks.com/matlabcentral/fileexchange/52848-matlab-support-for-mingw-w64-c-c-compiler) Install the MinGW-w64 C/C++ compiler for Windows (If the OS running is Windows)
    
### pip packages
Output a list of available conda env
```
conda env list
```
Active the conda environment
```
conda activate py38_torch
```
Alternatively install all the necessssary packages from [`src/requirements.txt`](src/requirements.txt) (which is generated by `pip freeze >> requirements.txt`
```
pip install -r /src/requirements.txt
```


## 3. Debugging 
Any installation debugging can be found in [Debug/README.md](Debug/README.md)

<!-- ## 3.	Latest releases -->
<!-- ## 4.	API references -->

# Running the DRIVE Simulator(Digital Twin)
-----------------------------------------------------------------------------------
<!-- TODO: Describe and show how to run your code and run the tests.  -->
## runSimulator.m setup output
If the setup all goes well you should see output from [Command Window](output/DRIVE_setup/DRIVE_runSimulator_cmdWindow_output.txt),[Command Prompt](output/DRIVE_setup/Traci_ServerPort_cmd_Prompt_output.txt) & [figure](output/DRIVE_setup/fig)(which contains the simulation of ambulance, passenger&pedestrian over 200 timesteps). In addition to the main output, you should also see preprocessed *.mat* files output in a new folder (/DRIVE_Simulator/mobilityFiles/preprocessedFiles/sumo). 

## Modify the DRIVE source file as required

After running the modified DRIVE source file, DRIVE generates synthetic data {numerical(real world dataset(𝑥‚𝑦)) + spacial(Open Street Map data)} as input/data for the next stage

# Running the 2-stage Neural Network

## Running VAE(1st stage):
Use the sythetic data from the previous step as input to train the VAE
```
python3 src/VAE/VAE_train.py
```
Then extract the environmental features 𝒵(mean(μ_e) and log variance(log(σ_e))) 
```
python3 src/VAE/VAE_exactor.py
```

## Running MLP(2nd stage):

Use the extracted environmental features(𝒵) along with numerical data (𝑥‚𝑦) as input to train & test a MLP for RSRP prediction.
For Comparison, use only the numerical data (𝑥‚𝑦) as input to train & test MLP as baseline.
```
python3 src/FCN/FCN.py
```
Then plot the graph to see results:
```
python3 src/FCN/Plot_DP_results.py
```

Final output using Mean Average Error (MAE): **2-stage NN(VAE+MLP)** vs **MLP**
<p align="center">
    <a href="https://github.com/luke-who/Digital-Twin-4th-Year/blob/a6d488e4141b4073c8aa48e366707ed5fbbf7572/src/FCN/plot_result/MAE_boxplot.svg">
        <img src="https://github.com/luke-who/Digital-Twin-4th-Year/blob/a6d488e4141b4073c8aa48e366707ed5fbbf7572/src/FCN/plot_result/MAE_boxplot.svg" width = 500/ height=auto>
    </a>
</p>

             **Fig3: MAE boxplot for the 2-stage NN and MLP comparison**
                      
-----------------------------------------------------------------------------------
<!-- TODO: Explain how other users and developers can contribute to make your code better.  -->

<!-- If you want to learn more about creating good readme files then refer the following [guidelines](https://docs.microsoft.com/en-us/azure/devops/repos/git/create-a-readme?view=azure-devops). You can also seek inspiration from the below readme files:
- [ASP.NET Core](https://github.com/aspnet/Home)
- [Visual Studio Code](https://github.com/Microsoft/vscode)
- [Chakra Core](https://github.com/Microsoft/ChakraCore) -->

# LICENSE
-----------------------------------------------------------------------------------
[MIT License](LICENSE)

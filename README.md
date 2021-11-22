# Digital Twins for future digital infrastructures
-----------------------------------------------------------------------------------
[![python](https://img.shields.io/badge/python-3.9.8-blue?style=plastic&logo=python)](https://www.python.org/downloads/release/python-398/)
[![pip](https://img.shields.io/badge/pip-v21.2.4-informational?&logo=pypi)](https://pypi.org/project/pip/21.2.4/)

[![PyTorch](https://img.shields.io/badge/PyTorch-1.10-orange?logo=PyTorch)](https://github.com/pytorch/pytorch/releases/tag/v1.10.0)
[![macOS Monterey 12.0](https://img.shields.io/badge/macOS%20Monterey-12.0-megenta?&color=E95420&logo=macOS)](https://www.apple.com/uk/macos/monterey/)
[![build running](https://camo.githubusercontent.com/5f0fabd6204876617cdc4ef33c48da33ac049e53a38eae71b239f8450776fe48/68747470733a2f2f706f77657263692e6f73756f736c2e6f72672f6a6f622f7079746f7263682d6d61737465722d6e696768746c792d7079332d6c696e75782d70706336346c652d6770752f62616467652f69636f6e)](https://camo.githubusercontent.com/5f0fabd6204876617cdc4ef33c48da33ac049e53a38eae71b239f8450776fe48/68747470733a2f2f706f77657263692e6f73756f736c2e6f72672f6a6f622f7079746f7263682d6d61737465722d6e696768746c792d7079332d6c696e75782d70706336346c652d6770752f62616467652f69636f6e)

## Motivation 
This project attempted to explore the communication efficiency and scalibility of the existing [DRIVE](https://github.com/ioannismavromatis/DRIVE_Simulator) simulator and possible improvements and futher optimisation 
## Aim
The aim is to build a scalable, lightweight digital twin solution for the existing 5G network and future networks

## Objectives
* Research the necessary library and development environment to conduct various FL simulations
* Create suitable unbalanced dataset, to simulate real world FL system and evaluate corresponding methods
* Build FL model with existing machine learning framework using basic model aggregation such as averaged weights update ($Fed\Avg$)
* Investigate the effect of parameters: number of clients, rounds, epochs, learning rate, optimisation functions on the global model 
* Benchmark the FL algorithm, use metrics such as learning accuracy and loss to evaluate model performance and convergence rate by deploying different communication reduction strategies
* Choose the best method out of all proposed reduction strategies for optimising communication, calculate the amount of reduction achieved

# Getting Started
-----------------------------------------------------------------------------------
<!-- TODO: Guide users through getting your code up and running on their own system. In this section you can talk about: -->
## 1.    Installation process

For setting up GCP, since the algorithm is not memory optimised, the memory usage was very intensive while running different reduction functions in `tff_vary_num_clients_and_rounds.py`.  This is likely due to the fact that TFF is not currently optimised for selecting a varying number of clients as it seems to mess up 
with the state during the iterative process and taking up huge accumulative memory.  As a result, the RAM in VM required on GCP for running tff_vary_num_clients_and_rounds.py was 128GB, at its peak it's using around 50% of the total memory so it's sth to keep in mind. 
### Install Pytorch
1. Install the Python development environment on your system

    `pip3 install torch torchvision`


### Install [DRIVE](https://github.com/ioannismavromatis/DRIVE_Simulator)
1. Install 

    ``

2. Test DRIVE

    ``


## 2.    Software dependencies
### Python version
The python version in this project throughout was [3.9.8](https://www.python.org/downloads/release/python-388/), [pyenv](https://github.com/pyenv/pyenv) was used to manage different python versions
### pypi packages
All the dependacies, versions and necessary packages are exported & listed in [requirements.txt](requirements.txt)(albeit not all of them are useful to run on local machines). 
### To install the requirements, do `pip3 install -r requirements.txt`


## 3.	Latest releases
## 4.	API references

# Build and Test
-----------------------------------------------------------------------------------
<!-- TODO: Describe and show how to build your code and run the tests.  -->
## Running tff_vary_num_clients_and_rounds.py:

`python3 tff_vary_num_clients_and_rounds.py MODE` to run the script with different mode arguments.
<!-- `python3 tff_vary_num_clients_and_rounds.py mode &` to run it in the background -->


The mode you can select are: MODE = `[constant,exponential,linear,sigmoid,reciprocal]`


## Running tff_UNIFORM_vs_NUM_EXAMPLES.py & tff_train_test_split.py (in 'other' folder):

`python3 tff_UNIFORM_vs_NUM_EXAMPLES.py` and `python3 tff_train_test_split.py` respectively to run these two scripts, no arguments/mode needed.
<!-- `python3 tff_vary_num_clients_and_rounds.py mode &` to run it in the background -->


## Running nn.py:

`python3 plot.py mode` to run the script with different mode arguments.
<!-- `python3 tff_vary_num_clients_and_rounds.py mode &` to run it in the background -->


The mode you can select are: mode = `['reduction_functions', 'femnist_distribution', 'uniform_vs_num_clients_weighting', 'accuracy_10percent_vs_50percent_clients_comparison', 'accuracy_5_34_338_comparison', 'reduction_functions_comparison','updates_comparison']`
# Contribute
-----------------------------------------------------------------------------------
<!-- TODO: Explain how other users and developers can contribute to make your code better.  -->

<!-- If you want to learn more about creating good readme files then refer the following [guidelines](https://docs.microsoft.com/en-us/azure/devops/repos/git/create-a-readme?view=azure-devops). You can also seek inspiration from the below readme files:
- [ASP.NET Core](https://github.com/aspnet/Home)
- [Visual Studio Code](https://github.com/Microsoft/vscode)
- [Chakra Core](https://github.com/Microsoft/ChakraCore) -->

# LICENSE
-----------------------------------------------------------------------------------
[MIT License](LICENSE)

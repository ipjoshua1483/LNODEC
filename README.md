# Lyapunov Neural Ordinary Differential Equation State-Feedback Policies

Continuous-time optimal control problems play a role in many decision making tasks and this paper presents a novel Lyapunov-based formulation for neural ODE-based control policies. An exponentially-stabilitizing control Lyapunov function is incorporated into the state feedback policy, which leads to stability guarantees and adversarial robustness to pertubations in the initial state. This repo presents the implementation of the algorithm for the double integrator case study in the paper.

## Installation
Install required dependencies with  `pip install -r requirements.txt`.

This repo uses `.ipynb` files for visualization and it is assumed the user has a valid environment for them.

## Demo
To obtain the plots in the paper, open `example_double_integrator.ipynb` and click `Run all`. 

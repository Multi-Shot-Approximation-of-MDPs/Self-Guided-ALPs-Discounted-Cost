# Self-guided Approximate Linear Programs
###### **Last update:** January 2020
###### The manuscript correpsoding to this repositotry is avilable at [arXiv](https://arxiv.org/abs/2001.02798) and [SSRN](https://ssrn.com/abstract=3512665).
###### **Topics:** Approximate dynamic programming, Approximate linear programming, Reinforcement learning, Random Fourier features, Inventory management
---

### Overview:
Computing high-quality control policies in sequential decision making problems is an important task across several application domains. Markov decision processes (MDPs) provide a powerful framework to find optimal policies in such problems but are often intractable to solve exactly due to known curses of dimensionality. Approximate dynamic programming (ADP) and reinforcement learning (RL) techniques are needed to tackle challenging MDPs. Some of these ADP/RL methods use value function approximation (VFA) to handle MDPs with large state spaces.

Approximate linear programming is a known ADP/RL approach for computing VFAs that has been applied to a wide variety of domains. VFAs in an approximate linear program (ALP) are represented as a linear combination of functions, referred to as basis functions, defined on the MDP state space. Solving ALP thus provides the (linear combination) weights associated with basis functions defining a VFA, which can be used to compute (i) a control policy and (ii) a lower bound on the optimal policy cost, which can be used to assees the optimality gap of a feasible policy, including the one from ALP. 

This repository provides an implementation of self-guided approximate linear programs and related benchmarks developed in [2] for solving MDPs with continuous state and action spaces. Specifically, it contains modules to compute VFAs, control policies, and optimality gaps for a given MDP. In a standard implementation of ALP, a user specifies a VFA by engineering basis functions using domain/application knowledge, which can be bottleneck to using ALP. Self-guided approximate linear programs sidestep the need for such basis function selection using random basis functions [3] in an iterative scheme. Since there is no basis function engineeering involved, the code provides an application-agnostic approach for solving high-dimensional MDPs.

Instances from two applications are used for testing. The first one is a variant of the perishable inventory control (PIC) problem, which gives rise to a challenging discounted cost infinite-horizon MDP. Here we show how the Fourier basis functions can be used in our ALP models and solved using constraint sampling. We also use a technique from [4] based on a saddle-point formulation and Metropolis-Hastings sampling to estimate a lower bound. The second application relates to generalized joint replenishment (GJR), which [1] model as an averaged-cost infinite horizon semi-MDP and approximately solve using an ALP with random basis functions generated in a dynamic manner exploiting problem-specific structure. On this application, we formulate our ALPs using random signum basis functions and show how they can be solved using constraint generation.

### **How to use this code?** 
 1. Download this repository on your local system and extract the zip file.
 2. Make sure that python 3.7 is installed on your machine.
 3. This repository depends on the following packages:
    * numpy
    * scipy
    * gurobipy (**version 8.1**)
    * pandas
    * sampyl
    * math
    * time
    * os
    * gc
    * sys
    * textwrap
    * itertools
    * multiprocessing
    * functools
    * shutil
    * importlib
 4. Please run the following code in your terminal to see which of the above Python libraries are missing on your machine.
 ```
  python checkLibraries.py 
 ```
 5. Typically, two packages *gurobipy* and *sampyl* will be missing. To install these libraries please visit [GUROBI](https://www.gurobi.com/gurobi-and-anaconda-for-windows/) and [SAMPYL](https://github.com/mcleonard/sampyl). **Please ensure that version 8.1 of Gurobi is set up in your sysyte.** If you are using conda enviroment, you can use the following code to install Gurobi 8.1:
 ```
  conda install gurobi=8.1
 ```
 6. Give read-and-write permission to two files *run_GJR.sh* and *run_PIC.sh*. For example, in Ubuntu or macOS, you can use the following code in the terminal to give permission.
 ```
  chmod 777 run_*.sh
 ```
 7. To check if the code is properly set up on your system, you can run the following code which solves a test instance of the GJR application.
 ```
  ./run_GJR.sh
 ```
 8. Also, a test instance of the PIC application can be run using the following code.
 ```
  ./run_PIC.sh
 ```
 9. Congratulations, the code is properly set up. You can now run different instances of GJR and PIC. The code can also be used for solving new applications using self-guided approximate linear programs. 

---
#### **References:**  
[1] Adelman, Daniel, and Diego Klabjan, *Computing near-optimal policies in generalized joint replenishment (2012)*. INFORMS Journal on Computing 24(1), pp 148-164.

[2] Pakiman, Parshan and Nadarajah, Selvaprabu and Soheili, Negar and Lin, Qihang, *Self-guided Approximate Linear Programs (2020)*. Available at SSRN: https://ssrn.com/abstract=3512665.

[3] Rahimi, Ali, and Benjamin Recht, *Uniform approximation of functions with random bases (2008)*. In 46th Annual Allerton Conference on Communication, Control, and Computing, pp 555-561.

[4] Lin, Qihang and Nadarajah, Selvaprabu and Soheili, Negar, *Revisiting approximate linear programming: Constraint-violation learning with applications to inventory control and energy storage (2019)*. Management Science, Forthcoming. 

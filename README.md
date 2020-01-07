# Self-guided Approximate Linear Programs
###### **Last update:** January 2020
###### The manuscript correpsoding to this repositotry is avilable at [SSRN](https://ssrn.com/abstract=3512665).
###### **Related topics:** Approximate dynamic programming, Approximate linear programming, Reinforcement learning, Random Fourier features, Inventory management
---

### Overview:
Computing high-quality control policies in sequential decision making problems is an important task across several application domains. Markov decision processes (MDPs) provide a powerful framework to find optimal policies in such problems but are often intractable to solve exactly due to known curses of dimensionality. Therefore, a class of approximate dynamic programming (ADP) and reinforcement learning (RL) techniques are developed to tackle challenging MDPs. Some of these ADP/RL methods use value function approximation (VFA) to handle the MDPs with large state space.

Approximate linear programming is a known ADP/RL approach for computing VFAs that has been applied to a wide variety of domains, including operations research, reinforcement learning, and artificial intelligence. VFAs in an approximate linear program (ALP) are represented as a linear combination of functions, referred to as basis functions, defined on the MDP state space. Solving ALP thus provides the (linear combination) weights associated with basis functions defining a VFA, which can be used to compute (i) control policy and (ii) lower bound on the optimal policy cost, which can be used to compute an optimality gap of the ALP policy. 

This repository provides an implementation of self-guided approximate linear programs, which is an algorithm developed in [2] for solving MDPs with continuous state and action spaces. This algorithm computes VFAs, control policies, and optimality gaps for a given MDP using a novel variant of the ALPs. In a standard implementation of ALP, a user should specify basis functions defining a VFA, that typically requires domain knowledge specific to applications, a potential bottleneck when using ALP. Self-guided approximate linear programs, however, sidestep the need for such basis function selection in ALP by embedding random basis functions [3] in a sequence of ALPs with increasing numbers of basis functions. Therefore, self-guided approximate linear programs is an application-agnostic approach for solving high-dimensional MDPs that bypasses the potential need for basis function selection by an inexpensive sampling of random basis functions.

The code implements self-guided approximate linear programs for two challenging applications. The first one is a variant of the perishable inventory control (PIC) problem, which gives rise to a challenging discounted cost infinite-horizon MDP. The second application relates to generalized joint replenishment (GJR), which [1] model as an averaged-cost infinite horizon semi-MDP and approximately solve using an ALP with basis functions generated in a dynamic manner exploiting problem-specific structure. An implementation of this benchmark is also provided in this repository.

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
 4. Please run the following code in your terminal to see which of the above Python libraries are missing on your machine.
 ```
  python checkLibraries.py 
 ```
 5. Typically, two packages *gurobipy* and *sampyl* are missing. To install these libraries please visit [GUROBI](https://www.gurobi.com/gurobi-and-anaconda-for-windows/) and [SAMPYL](https://github.com/mcleonard/sampyl). **Please ensure that version 8.1 of Gurobi is set up in your sysyte.** If you are using conda enviroment, you can use the following code to install Gurobi 8.1:
 ```
  conda install gurobi=8.1
 ```
 6. Give read-and-write permission to two files *run_GJR.sh* and *run_PIC.sh*. For example, in Ubuntu or macOS, you can use the following code in the terminal to give permission.
 ```
  chmod 777 run_*.sh
 ```
 7. To check if the code is properly set up on your system, you can run the following code which solves a test instance of GJR application.
 ```
  ./run_GJR.sh
 ```
 8. Also, a test instance of the PIC application can be run using the following code.
 ```
  ./run_PIC.sh
 ```
  

#### **References:**  
[1] Adelman, Daniel, and Diego Klabjan, *Computing near-optimal policies in generalized joint replenishment (2012)*. INFORMS Journal on Computing 24, no. 1 (2012): 148-164.

[2] Pakiman, Parshan and Nadarajah, Selvaprabu and Soheili, Negar and Lin, Qihang, *Self-guided Approximate Linear Programs (2020)*. Available at SSRN: https://ssrn.com/abstract=3512665.


[3] Rahimi, Ali, and Benjamin Recht, *Uniform approximation of functions with random bases (2008)*. In 2008 46th Annual Allerton Conference on Communication, Control, and Computing, pp. 555-561. IEEE, 2008.

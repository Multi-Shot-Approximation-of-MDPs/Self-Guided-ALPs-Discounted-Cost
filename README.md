# Self-guided Approximate Linear Programs
###### **Last update:** January 2020
***
###### The paper correpsoding to this repositotry is avilable at [SSRN](https://ssrn.com/abstract=3512665).

#### Overview:
Computing high-quality control policies in sequential decision making problems is an important task across several application domains. Markov decision processes (MDPs) provide a powerful framework to find optimal policies in such problems but are often intractable to solve exactly due to known curses of dimensionality. Therefore, approximate dynamic programming (ADP) and reinforcement learning (RL) techniques are developed for managing challenging MDPs. The are various ADP/RL methods that use value function approximations (VFAs) to handle the MDPs with large state space.

Approximate linear programming is a known ADP/RL approach for computing VFAs that has been applied to a wide variety of domains, including operations research, reinforcement learning, and artificial intelligence. VFAs in an approximate linear program (ALP) are represented as a linear combination of functions, referred to as basis functions, defined on the MDP state space. Solving ALP thus provides the (linear combination) weights associated with basis functions defining a VFA, which can be used to compute control policy and lower bound on the optimal policy cost, which can be used to compute an optimality gap of the ALP policy as well as other heuristic policies. 

This repository provides an implementation of self-guided approximate linear programs, which is an algorithm developed in [1] for solving MDPs with continuous state and action spaces. This algorithm computes VFAs, control policies, and optimality gaps for a given MDP using a novel variant of the ALPs. In a standard implementation of ALP, a user should specify basis functions defining a VFA, that typically requires domain knowledge specific to applications, a potential bottleneck when using ALP. Self-guided approximate linear programs, however, sidestep the need for such basis function selection in ALP by embedding random basis functions [3] in a sequence of ALPs with increasing numbers of basis functions. Therefore, self-guided approximate linear programs is an application-agnostic approach for solving high-dimensional MDPs that bypasses the potential need for basis function selection by an inexpensive sampling of random basis functions.

The code implements self-guided approximate linear programs for two challenging applications. The first one is a variant of the perishable inventory control (PIC) problem considered in [Karaesmen et al. (2011)], which gives rise to a challenging discounted cost infinite-horizon MDP. The second application relates to generalized joint replenishment (GJR), which [Adelman and Klabjan (2012)] model as an averaged-cost infinite horizon semi-MDP and approximately solve using an ALP with basis functions generated in a dynamic manner exploiting problem-specific structure. An implementation of this benchmark is also provided in this repository.

#### **Related topics:**  
  * Approximate dynamic programming,
  * Approximate linear programming,
  * Reinforcement learning,
  * Random Fourier features,
  * Inventory management










##### **Abstract** 

Approximate linear programs (ALPs) are well-known models based on value function approximations (VFAs)
to obtain heuristic policies and lower bounds on the optimal policy cost of Markov decision processes (MDPs).
The ALP VFA is a linear combination of predefined basis functions that are chosen using domain knowledge
and updated heuristically if the ALP optimality gap is large. We side-step the need for such basis function
engineering in ALP – an implementation bottleneck – by proposing a sequence of ALPs that embed increasing
numbers of random basis functions obtained via inexpensive sampling. We provide a sampling guarantee and
show that the VFAs from this sequence of models converge to the exact value function. Nevertheless, the per-
formance of the ALP policy can fluctuate significantly as more basis functions are sampled. To mitigate these
fluctuations, we “self-guide” our convergent sequence of ALPs using past VFA information such that a worst-
case measure of policy performance is improved. We perform numerical experiments on perishable inven-
tory control and generalized joint replenishment applications, which, respectively, give rise to challenging
discounted-cost MDPs and average-cost semi-MDPs. We find that self-guided ALPs (i) significantly reduce
policy fluctuations and improve the optimality gaps from an ALP approach that employs basis functions
tailored to the former application, and (ii) deliver optimality gaps that are comparable to a known adaptive
basis function generation approach targeting the latter application. More broadly, our methodology provides
application-agnostic policies and lower bounds to benchmark approaches that exploit application structure.


##### **How to use this code?** 



1. Make sure that Python 3.7 is already installed in your machine.
2. Run the configure.py to see list of python libraries missing on your machine. Likely gurobipy and sampyl are the missing libraries. Please install then from the following links
  sampyl: http://mcleonard.github.io/sampyl/
  gurobipy: https://www.gurobi.com/gurobi-and-anaconda-for-windows/
  
  
  # Line 23 of ALPSolver.py
  # run_GJR.sh
  # The code has ben tested on Gurobi 8.1. conda install gurobi=8.1

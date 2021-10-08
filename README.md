# Self-guided Approximate Linear Programs
###### **Last update:** October 2021
###### The manuscript correpsoding to this repositotry is avilable at [arXiv](https://arxiv.org/abs/2001.02798) and [SSRN](https://ssrn.com/abstract=3512665).
###### **Topics:** Approximate dynamic programming, Approximate linear programming, Reinforcement learning, Random Fourier features, Inventory management
---

### Overview:
Approximate linear programs (ALPs) are well-known models based on value function approximations (VFAs) to obtain policies and lower bounds on the optimal policy cost of discounted-cost Markov decision processes (MDPs). Formulating an ALP requires (i) basis functions, the linear combination of which defines the VFA, and (ii) a state-relevance distribution, which determines the relative importance of different states in the ALP objective for the purpose of minimizing VFA error. Both these choices are typically heuristic: basis function selection relies on domain knowledge while the state-relevance distribution is specified using the frequency of states visited by a heuristic policy. We propose a self-guided sequence of ALPs that embeds random basis functions obtained via inexpensive sampling and uses the known VFA from the previous iteration to guide VFA computation in the current iteration. Self-guided ALPs mitigate the need for domain knowledge during basis function selection as well as the impact of the initial choice of the state-relevance distribution, thus significantly reducing the ALP implementation burden. We establish high probability error bounds on the VFAs from this sequence and show that a worst-case measure of policy performance is improved. We find that these favorable implementation and theoretical properties translate to encouraging numerical results on perishable inventory control and options pricing applications, where self-guided ALP policies improve upon policies from problem-specific methods. More broadly, our research takes a meaningful step toward application-agnostic policies and bounds for MDPs.

### **How to use this code?** 
 1. Download this repository on your local system and extract the zip file.
 2. Make sure that python 3.8 or above is installed on your machine.
 3. Please run the following code in your terminal to see if all needed Python libraries are installed on your machine or not.
 ```
  python3.8 checkLibraries.py 
 ```
 4. Typically, two packages *gurobipy* and *sampyl* will be missing. To install these libraries please visit [GUROBI](https://www.gurobi.com/gurobi-and-anaconda-for-windows/) and [SAMPYL](https://github.com/mcleonard/sampyl). 
 5. To check if the code is properly set up on your system, you can run the following code which solves a test instance of the GJR application.
 ```
  python3.8 main_PIC.py
 ```
 6. Congratulations, the code is properly set up. You can now run different instances of GJR and PIC. The code can also be used for solving new applications using self-guided approximate linear programs. 


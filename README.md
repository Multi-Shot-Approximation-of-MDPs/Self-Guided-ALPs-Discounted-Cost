# Self-guided Approximate Linear Programs
###### **Last update:** Dec 2023
###### This repository contains the code for the Self-Guided Approximate Linear Programming algorithm described in the paper titled *Self-Guided Approximate Linear Programs: Randomized Multi-Shot Approximation of Discounted Cost Markov Decision Processes* by Parshan Pakiman, Selvaprabu Nadarajah, Negar Soheili, and Qihang Lin. The paper is available at [Management Science](https://ssrn.com/abstract=3512665).

###### **Topics:** approximate dynamic programming, approximate linear programming, model-based reinforcement learning, random Fourier features, inventory management
---

### Abstract
Approximate linear programs (ALPs) are well-known models based on value function approximations (VFAs) to obtain policies and lower bounds on the optimal policy cost of discounted-cost Markov decision processes (MDPs). Formulating an ALP requires (i) basis functions, the linear combination of which defines the VFA, and (ii) a state-relevance distribution, which determines the relative importance of different states in the ALP objective for the purpose of minimizing VFA error. Both these choices are typically heuristic: basis function selection relies on domain knowledge while the state-relevance distribution is specified using the frequency of states visited by a baseline policy. We propose a self-guided sequence of ALPs that embeds random basis functions obtained via inexpensive sampling and uses the known VFA from the previous iteration to guide VFA computation in the current iteration. In other words, this sequence takes multiple shots at randomly approximating the MDP value function with VFA-based guidance between consecutive approximation attempts. Self-guided ALPs mitigate domain knowledge during basis function selection and the impact of the state-relevance-distribution choice, thus reducing the ALP implementation burden. We establish high probability error bounds on the VFAs from this sequence and show that a worst-case measure of policy performance is improved. We find that these favorable implementation and theoretical properties translate to encouraging numerical results on perishable inventory control and options pricing applications, where self-guided ALP policies improve upon policies from problem-specific methods. More broadly, our research takes a meaningful step toward application-agnostic policies and bounds for MDPs.

### **How to use this repository** 
The steps detailed below have been validated on Mac OS 14.1.2 and Ubuntu 20.04 systems configured with Python Python 3.10.6. Windows users are encouraged to create similar procedures accordingly.

1. Download the [repository](https://github.com/Multi-Shot-Approximation-of-MDPs/Self-Guided-ALPs-Discounted-Cost) on your local system and extract the zip file.

2. Open *Terminal* on your machine.

3. Check version of your Python. For example, run the code below:
```
python3 --version
```

4. Please confirm that Python 3.10.6 is installed on your machine. There are various methods available for installing Python. We leave this step to the user's discretion.

5. Create a virtual environment called "ALP". For example, use the following code:
```
python -m venv ALP   
```

6. Activate ALP environment as follows
```
source ALP/bin/activate
``` 

7. Update pip as, e.g., run 
```
pip install --upgrade pip
```

7. Install following libraries on the ALP environment:
    - numpy (python -m pip install numpy)
    - pandas (python -m pip install pandas)
    - scipy (python -m pip install scipy)
    - numba (python -m pip install numba)
    - multiprocessing (python -m pip install multiprocessing)
    - tqdm (python -m pip install tqdm)
    - emcee (python -m pip install emcee) 
    - sampyl (python -m pip install sampyl)
    - importlib (python -m pip install importlib)
    - sampyl_mcmc (python -m pip install sampyl_mcmc)
    - nengo (python -m pip install nengo) 
    - gurobipy (python -m pip install gurobipy)

8. To utilize Gurobi for solving large-scale linear programs, please ensure the installation of the Gurobi license. If you're affiliated with academia, Gurobi offers a free academic license. For further details, visit [this page](https://www.gurobi.com/academia/academic-program-and-licenses/).




9. Provided that the ALP Python environment and Gurobi license are correctly installed, you can employ the following code to solve test instances of perishable inventory control application. Please run the following code:
```
  ./run_PIC.sh
```

10. The provided code solves instance number 1 using the FALP algorithm with 20 random Fourier features employing a uniform state-relevance distribution. You can find specifications of instance number 1 under *MDP/PIC/Instances/instance_1.py*. To solve this instance using an alternate algorithm, you can modify the file run_PIC.sh. For instance, changing the *algo_name* from "FALP" to "SG-FALP" in this file and rerunning *run_PIC.sh* will display the output for the self-guided FALP algorithm applied to instance number 1. A screenshot of the output of these algorithms is attached below:

<img title="a title" alt="Alt text" src="output.png">

11. To solve the test instance of Bermudan options pricing problem, please run the file named *run_BerOpt.sh*. 

12. To run other instances or use other algorithms implemented in this repository, you need to modify file *run_PIC.sh* and *run_BerOpt.sh*.

---
"""
-------------------------------------------------------------------------------

    Authors:    Parshan Pakiman  | https://parshanpakiman.github.io/homepage/
                Selva Nadarajah  | https://selvan.people.uic.edu/
                         
    Licensing Information: The MIT License
-------------------------------------------------------------------------------
"""
import time
import textwrap
import numpy as np
import gurobipy as gb
from BasisFunction.hatBasisFunctions import HatBasis
from scipy.stats import randint
from gurobipy import *
from utils import printWithPickle,powerset,whereIsElementInPowerSet,arreqclose_in_list, dropZeros,AK_Header
from multiprocessing import Pool
from os import unlink

""" Global Gurobi Variables """
INFINITY      = gb.GRB.INFINITY
CONTINUOUS    = gb.GRB.CONTINUOUS
BINARY        = gb.GRB.BINARY
MAXIMIZE      = gb.GRB.MAXIMIZE
MINIMIZE      = gb.GRB.MINIMIZE
quicksum      = gb.quicksum
LinExpr       = gb.LinExpr

"""
    This following class implements the ALgorithm in [1].
"""
class GJR_Benchmark():

    """
    The constructor of the class that initializes an instance described in
    expInfo.
    """
    def __init__(self,expInfo):
        #----------------------------------------------------------------------
        # Configuring the MDP, basis functions, and other parameters need for 
        # declaring a GJR instance.
        self.mdp            = expInfo['mdp']['type'](expInfo['mdp'])
        self.bf             = expInfo['bf']
        self.ub             = expInfo['ub']
        self.runTime        = expInfo['misc']['runTime']
        self.dimX           = self.mdp.dimX
        self.numThreads     = self.mdp.Threads 
        self.transTime      = self.mdp.transTime
        self.powSet         = powerset(range(0,self.dimX))
        self.powSetLen      = len(self.powSet)
        self.RNG_DIM_X      = range(self.dimX)
        self.RNG_POW_SET    = range(self.powSetLen)
        self.OMEGA          = np.ceil(np.sqrt(self.dimX*np.linalg.norm(self.mdp.invUppBounds,ord=2)))
        
        #----------------------------------------------------------------------
        # Configuring initial hat basis functions
        self.hatSetting     = {}
        unitVecs            = [self.getUnitVec(_) for  _ in range(self.dimX)]
        pairVecs            =  self.getPairVec()
        
        self.hatSetting.update({'dimX'           : self.dimX})
        self.hatSetting.update({'diamX'          : np.linalg.norm(self.mdp.invUppBounds,2)})
        self.hatSetting.update({'BF_num'         : None})
        self.hatSetting.update({'ridgeVector'    : unitVecs+pairVecs })     
        self.hatSetting.update({'numRidgeVec'    : len(self.hatSetting['ridgeVector'])})
        self.hatSetting.update({'indexUnitVec'   : range(self.dimX)})
        self.hatSetting.update({'indexPairVec'   : range(self.dimX, self.dimX+len(pairVecs))})
        self.hatSetting.update({'breakPoints'    : [self.getInitBreakPts(j) for j in range(len(self.hatSetting['ridgeVector']))]})
        self.hatSetting.update({'numBreakPts'    : [len(self.hatSetting['breakPoints'][_])-2 for _ in range(self.hatSetting['numRidgeVec'])] })
        
        
    """ 
        Unit vector with 1 at the i-th coordinate and zero elsewhere
    """
    def getUnitVec(self,i):
        unitVec       = np.zeros(self.dimX)
        unitVec[i]    = 1.0
        return unitVec

    """ 
        Pair ridge vector defined in [1]
    """
    def getPairVec(self):
        #----------------------------------------------------------------------
        # Define all pair ridge vectors defined in [1]
        pairVec = []
        for i in range(self.dimX):
            for j  in range(i+1, self.dimX):
                #--------------------------------------------------------------
                # Start with a zero vector
                zero        = np.zeros(self.dimX)
                
                #--------------------------------------------------------------
                # Use the consumption rate to update coordinate i and j of
                # the zero vector
                zero[i]     =   1/self.mdp.consumptionRate[i]
                zero[j]     =  -1/self.mdp.consumptionRate[j]
                pairVec.append(zero)
        
        #----------------------------------------------------------------------
        # Return the collection of pair vectors        
        return pairVec     
    
    """
    The following Algorithm implements the adaptive basis function generation
    in [1].
    """
    def AK_GJR_Algorithm(self,trial):
        
        #----------------------------------------------------------------------
        # Load an initial set of state-action pairs needed to perform 
        # row generation
        MVC_StateList, \
            MVC_ActionList, \
                MVC_NStateList  = self.loadStateActionsForFeasibleALP(trial)
        
        #----------------------------------------------------------------------
        # Unit and pair ridge vectors that are defined in AK 2012.        
        numB_Unit               = 0
        numB_Pair               = 0
        numB_Unit_init          = 0
        numB_Pair_init          = 0
        numNewRidgeAdded        = 0
        basis_Itr               = 0
        basisNum                = 0
        
        #----------------------------------------------------------------------
        # Cost of policy and check if a policy is cyclic or not;
        # if yes, what is its length?
        cost                    = float('inf')
        isDualCyclic            = False
        cycleLength             = float('inf')
        
        #----------------------------------------------------------------------
        # As we iterate, we store the previous Gorubu model and basis function
        ALP_MODEL               = None
        BF                      = None
        
        #----------------------------------------------------------------------
        # Track the bounds and runtime
        bestUB                  = float('inf')
        initialLB               = -float('inf')
        initialUB               = float('inf')
        runTimeLimit            = self.runTime
        
        #----------------------------------------------------------------------
        # If true, generate basis; otherwise, use affine VFA
        useVFA                  = False    
        
        #----------------------------------------------------------------------
        # Update the number of unit and pair ridge bases 
        for i in self.hatSetting['indexUnitVec']:
            numB_Unit += self.hatSetting['numBreakPts'][i]
        for i in  list(set(range(self.hatSetting['numRidgeVec']))- set( self.hatSetting['indexUnitVec'])):
            numB_Pair += self.hatSetting['numBreakPts'][i]
           
        #----------------------------------------------------------------------
        # Store initial number of unit and pair bases
        numB_Unit_init          = numB_Unit
        numB_Pair_init          = numB_Pair
        
        #----------------------------------------------------------------------
        # Define a random initial state for policy simulation
        s = [np.random.uniform(low=0.0,high=self.mdp.invUppBounds[_]) for _ in range(self.dimX)]
        s[np.random.randint(self.dimX)] = 0.0
        s = np.asarray(s)
        initialState = s      
            
        #----------------------------------------------------------------------
        # Setup Loggers 
        logger          = open(self.mdp.Adrr+ '/summary_trial_'+str(trial)+".txt","w+")
        timeLogger      = open(self.mdp.Adrr+ '/time_gap_list_'+str(trial)+".txt","w+")        
        timeLogger.write('{:^5} | {:^15} | {:^15} | {:^15} |'.format('B', 'GAP (%)', 'Iter-RT (s)', 'Cum-RT (s)'))
        timeLogger.write("\n") 
        
        #----------------------------------------------------------------------
        # Initialize the result table
        resTable        = np.zeros(shape = (5000,14))
        timeGapTable    = np.zeros(shape = (5000,4))
        cumTime         = 0    
        
        #----------------------------------------------------------------------
        # Print out the header for showing the results
        AK_Header(self.mdp.mdpName,trial,"Generalized Joint Replenishment",148)
        printWithPickle('{:^3} | {:^15} | {:^8} | {:^8} | {:^8} | {:^8} | {:^8} | {:^8} | {:^5} | {:^5} | {:^5} | {:^5} | {:^5} | {:^5} | {:^8} |'.format(
                                'B','VIOL','I-LB','LB','I-UB','UB','I-GAP','GAP','CYC?','C-LEN','MAX-B','R','#b U', '#b P', 'T(s)'),logger)
        printWithPickle('-'*148,logger)

        #----------------------------------------------------------------------
        # If there is a constraint violation, perform row generation
        stillAddConstr = False
        
        #----------------------------------------------------------------------
        # Iterate until a stopping criteria is met
        while True:
            rowGenCounter = 0
            
            #------------------------------------------------------------------
            # Store previous state and action
            prevS = np.asarray([-float('inf') for _ in range(self.mdp.dimX)])
            prevA = np.asarray([-float('inf') for _ in range(self.mdp.dimX)])
            
            #------------------------------------------------------------------
            # If there is an ill condition situation, increase illCondBasisCount
            illCondBasisCount = 0
            
            #------------------------------------------------------------------
            # Set the starting time to zero
            start_time =  time.time()
            
            #------------------------------------------------------------------
            # Generate Rows 
            while True:
        
                #--------------------------------------------------------------
                # Check if any basis function should be generated or not.
                # If not, use affine VFA.
                if BF == None:
                    useVFA = False
                else:
                    useVFA = True
                    
                #--------------------------------------------------------------
                # If we did not already solve an ALP, then solve ALP and get
                # Gurobi model
                    
                if not stillAddConstr:
                    intercept,\
                    linCoef,\
                    BF_Coef,\
                    ALPval,\
                    dual,\
                    ALP_MODEL = self.ALP_GJR(BF,
                                             stateList  = MVC_StateList,
                                             actionList = MVC_ActionList,
                                             nStateList = MVC_NStateList,
                                             useVFA     = useVFA,
                                             basisItr   = basis_Itr,
                                             model      = None,
                                             addedS     = None,
                                             addedA     = None)  
                #--------------------------------------------------------------
                # Otherwise, use the already computed Gurobi model and solve ALP
                else:
                    intercept,\
                    linCoef,\
                    BF_Coef,\
                    ALPval,\
                    dual,\
                    ALP_MODEL = self.ALP_GJR(BF,
                                             stateList  = MVC_StateList,
                                             actionList = MVC_ActionList,
                                             nStateList = MVC_NStateList,
                                             useVFA     = useVFA,
                                             basisItr   = basis_Itr,
                                             model      = ALP_MODEL,
                                             addedS     = prevS,
                                             addedA     = prevA)                  
                    
                #--------------------------------------------------------------
                # Given computed VFA, perform row generation.
                s, a, MVCval, NS   = self.getMostViolatedConstraint(intercept,
                                                                   linCoef,
                                                                    BF_Coef,
                                                                    BF)
                
                #--------------------------------------------------------------
                # Set convergence tolerance for row generation
                if basis_Itr ==0:
                    isALPConverged   = 1e-3
                else:
                    isALPConverged   = 0.001*bestUB
                
                #--------------------------------------------------------------
                # In the case that there is no violation,
                if  MVCval >= -isALPConverged:
                    #----------------------------------------------------------
                    # There is no need for additional row generation
                    stillAddConstr   = False
                    printWithPickle('{:>3d} | {:>15.5f} | {} |'.format(basisNum,
                                                                       (MVCval),
                                                                       ''.ljust(122))
                                    ,logger)
                    
                    #----------------------------------------------------------
                    # Analyze dual ALP for the existence of a cyclic policy 
                    # and compute some parameters such as (support) for basis 
                    # generation.
                    support,\
                    Q,\
                    flowViolation,\
                    isDualCyclic,\
                    costDual,\
                    cycleLengthDual = self.dualALP_GJR(dual,
                                                       MVC_StateList,
                                                       MVC_ActionList,
                                                       MVC_NStateList )
                    
                    #----------------------------------------------------------
                    # If a cyclic policy is found, then stop as an
                    # optimal one is found.
                    if isDualCyclic:
                        cost        = costDual
                        cycleLength = cycleLengthDual
                        
                        #------------------------------------------------------
                        # Get the upper bound
                        cyclicPolFound,\
                        cycleLength,\
                        cost =  self.getUpperBound(state_ini = initialState,
                                                   BF        = BF ,
                                                   intercept = intercept,
                                                   linCoef   = linCoef,
                                                   BF_Coef   = BF_Coef)
                        
                        #------------------------------------------------------
                        # Set the best upper bound
                        bestUB = min(bestUB,cost)
                        
                        #------------------------------------------------------
                        # Store initial upper and lower bounds
                        if basis_Itr == 0 :
                            initialLB = ALPval
                            initialUB = cost
                        
                        #------------------------------------------------------
                        # Pickle and print bounds and improvement
                        printWithPickle('{:>3d} | {:^15} | {:>8.2f} | {:>8.2f} | {:>8.2f} | {:>8.2f} | {:>8.2f} | {:>8.2f} | {:>5} | {:>5d} | {:>5d} | {:>5d} | {:>5d} | {:>5d} | {:>8.0f} |'.format(
                                        basisNum,   ''.ljust(15),initialLB,
                                        ALPval,     initialUB,   bestUB,
                                        (1-initialLB/initialUB)*100,      
                                        (1-ALPval/bestUB)*100,
                                        isDualCyclic,           cycleLength,
                                        basis_Itr,              numNewRidgeAdded,
                                        numB_Unit,              numB_Pair,
                                        cumTime),logger) 
                        printWithPickle('-'*148,logger)
                        logger.flush()
                        
                        #------------------------------------------------------
                        # Increment the number of basis functions
                        basisNum=0
                        for j in range(self.hatSetting['numRidgeVec']):
                            basisNum += len(self.hatSetting['breakPoints'][j])
                        break
    
                    #----------------------------------------------------------
                    # If a cyclic policy is not found, then compute upper bound
                    else:
                        #------------------------------------------------------
                        # Less frequently update upper bound and more focus on
                        # improving lower bound
                        if basis_Itr%15 == 0 :
                            cyclicPolFound,\
                            cycleLength,\
                            cost =  self.getUpperBound(state_ini    = initialState, 
                                                       BF           = BF,
                                                       intercept    = intercept,
                                                       linCoef      = linCoef,
                                                       BF_Coef      = BF_Coef)
                            
                        #------------------------------------------------------
                        # Set the best upper bound
                        bestUB = min(bestUB,cost)
                        
                        #------------------------------------------------------
                        # Store initial upper and lower bounds
                        if basis_Itr == 0 :
                            initialLB = ALPval
                            initialUB = cost
                        
                        #------------------------------------------------------
                        # Pickle and print bounds and improvement   
                        printWithPickle('{:>3d} | {:^15} | {:>8.2f} | {:>8.2f} | {:>8.2f} | {:>8.2f} | {:>8.2f} | {:>8.2f} | {:>5} | {:>5d} | {:>5d} | {:>5d} | {:>5d} | {:>5d} | {:>8.0f} |'.format(
                                        basisNum,           ''.ljust(15),
                                        initialLB,          ALPval,
                                        initialUB,          bestUB, 
                                        (1-initialLB/initialUB)*100,
                                        (1-ALPval/bestUB)*100,
                                        isDualCyclic,       cycleLength,
                                        basis_Itr,          numNewRidgeAdded,
                                        numB_Unit,          numB_Pair,
                                        cumTime),logger)  
                        printWithPickle('-'*148,logger)
                    
    
                    #----------------------------------------------------------
                    # Store results                        
                    resTable[basis_Itr,0]   = trial
                    resTable[basis_Itr,1]   = basisNum
                    resTable[basis_Itr,2]   = initialLB
                    resTable[basis_Itr,3]   = ALPval
                    resTable[basis_Itr,4]   = initialUB
                    resTable[basis_Itr,5]   = cost
                    resTable[basis_Itr,6]   = cycleLength
                    resTable[basis_Itr,7]   = (1-initialLB/initialUB)*100
                    resTable[basis_Itr,8]   = (1-ALPval/cost)*100
                    resTable[basis_Itr,9]   = (1-initialLB/ALPval)*100
                    resTable[basis_Itr,10]  = (1-cost/initialUB)*100
                    resTable[basis_Itr,11]  = resTable[basis_Itr,7] - resTable[basis_Itr,8]
                    resTable[basis_Itr,12]  = numB_Unit 
                    resTable[basis_Itr,13]  = numB_Pair 
                    break
                
                #--------------------------------------------------------------
                # However, if violation is not negligible
                else:
                    #----------------------------------------------------------
                    # Every 50 iterations show the value of the violation
                    if rowGenCounter%50==0:
                        printWithPickle('{:>3d} | {:>15.5f} | {} |'.format(basisNum,
                                                                           (MVCval),
                                                                           ''.ljust(122))
                                        ,logger)           
                    
                    #----------------------------------------------------------
                    """
                            If the most violating state-action pair of the 
                            previous and  current iterations are identical, then
                            there is an issue. In fact, row generation could not
                            cut off the previous infeasible solution. If this 
                            issue happens, Gurobi parameters need to be adjusted.
                    """
                    if  np.linalg.norm(s-prevS,ord=np.inf) < 1e-50 and\
                        np.linalg.norm(a-prevA,ord=np.inf) < 1e-50:
        
                        illCondBasisCount +=1
                        #I = np.random.randint(self.mdp.dimX)

                        if illCondBasisCount >=100:
                            illCondBasisCount       = 0
                            msg = 'WARNING! Row generation could not cut off the previous infeasible solution. Gurobi parameters need to be adjusted for addressing this issue.'
                            print('\n\n'+textwrap.TextWrapper(width=50).fill(text=msg)+'\n\n')
                            break
                    
                    #----------------------------------------------------------
                    # Store previous state-action pairs and add the generated
                    # violating state-action pair to the list of violating 
                    # constraints.
                    prevS = s
                    prevA = a
                    MVC_StateList.append(s)
                    MVC_ActionList.append(a)
                    MVC_NStateList.append(NS)

                    #----------------------------------------------------------
                    # Continuous generating rows
                    stillAddConstr = True  
                    
                rowGenCounter+=1
                logger.flush()
                
            #------------------------------------------------------------------
            # Row generation is done, record obtained stats
            for j in range(self.hatSetting['numRidgeVec']):
                basisNum += len(self.hatSetting['breakPoints'][j])-4
            
            #------------------------------------------------------------------
            # Collecting runtime 
            GAP             = (1-ALPval/bestUB)*100
            runTime         = time.time() - start_time
            cumTime        += runTime
            
            #------------------------------------------------------------------
            # Check if the stopping criterion based on runtime is met
            if cumTime >= runTimeLimit:
                printWithPickle('The runtime stoppitng criterion is met ({:>8.3} m). The terminal optimality gap is ({:>8.3} %).'.format(cumTime/60,GAP),
                                logger)
                timeLogger.write('The runtime stoppitng criterion is met ({:>8.3} m). The terminal optimality gap is ({:>8.3} %).'.format(cumTime/60,GAP))
                return self.outPut(0,logger,timeLogger,resTable,timeGapTable)      
            
            #------------------------------------------------------------------
            # Otherwise, record timing and other stats
            timeLogger.write('{:>5d} | {:>15.2f} | {:>15.2f} | {:>15.2f} |'.format(basisNum,GAP,runTime,cumTime))
            timeLogger.write("\n")  
            timeLogger.flush()
            timeGapTable[basis_Itr,0] = basisNum
            timeGapTable[basis_Itr,1] = GAP
            timeGapTable[basis_Itr,2] = runTime
            timeGapTable[basis_Itr,3] = cumTime
            
            #------------------------------------------------------------------
            # If a 2% optimal policy is found, print stats and stop
            if  GAP<= 2.00 or isDualCyclic:
                printWithPickle('{:>3d} | {:^15} | {:>8.2f} | {:>8.2f} | {:>8.2f} | {:>8.2f} | {:>8.2f} | {:>8.2f} | {:>5} | {:>5d} | {:>5d} | {:>5d} | {:>5d} | {:>5d} | {:>8.0f} |'.format(
                                    self.hatSetting['numRidgeVec'],
                                    ''.ljust(15),
                                    initialLB, 
                                    ALPval,
                                    initialUB,
                                    bestUB,
                                    (1-initialLB/initialUB)*100,
                                    (1-ALPval/bestUB)*100,
                                    isDualCyclic,
                                    cycleLength,
                                    basis_Itr,
                                    numNewRidgeAdded,
                                    numB_Unit,
                                    numB_Pair,
                                    cumTime),logger)  
                #--------------------------------------------------------------
                # Print the gap and runtime and return the results
                printWithPickle('The algorithm is converged with the terminal optimality gap of ({:>8.3} %) in ({:>8.3} m).'.format(GAP,cumTime/60),logger)
                return self.outPut(0,logger,timeLogger,resTable,timeGapTable)
            
            #------------------------------------------------------------------
            # If a 2% optimal policy is not found, continue basis generation
            else: 
                #--------------------------------------------------------------
                # Increment the number of bases and generate new ridge bases
                basis_Itr          +=1
                BF, numNewRidge     = self.getNewHatBases(basis_Itr,
                                                          support,
                                                          Q,
                                                          flowViolation)
                numNewRidgeAdded   += numNewRidge    
                
                #--------------------------------------------------------------
                # Start row generation from scratch
                stillAddConstr      = False
                
                #--------------------------------------------------------------
                # Update the number of unit and pair vectors and remove the 
                # initial number of bases
                numB_Unit = -numB_Unit_init
                numB_Pair = -numB_Pair_init
                for i in self.hatSetting['indexUnitVec']:
                    numB_Unit += self.hatSetting['numBreakPts'][i]
                for i in  list(set(range(self.hatSetting['numRidgeVec']))- set( self.hatSetting['indexUnitVec'])):
                    numB_Pair += self.hatSetting['numBreakPts'][i]


    """
        Initializing the breakpoints of ridge bases as described in [1]
    """
    def getInitBreakPts(self,j):
        #----------------------------------------------------------------------
        """
            The following program is defined on page 160 of [1] that computes
                1) b^j_1 via min_x  r^jx
                2) b^j_2 via max_x  r^jx
            This corresponds with two hat functions, one centered at the leftmost
            point of the domain and the other centered at the rightmost
            point of the domain.
        """   
        breakPoint      = gb.Model('left')
        breakPoint.setParam('OutputFlag',False)
        breakPoint.setParam('LogFile','Output/GJR/groubiLogFile.log')
        unlink('gurobi.log')
        allBreakPts     = []
        r               = self.hatSetting['ridgeVector'][j]
        x               = [breakPoint.addVar(lb = 0.0,vtype = CONTINUOUS) for _ in range(len(r))]
        xMin            =  breakPoint.addVar(lb = 0.0,vtype = CONTINUOUS)
        breakPoint.setObjective(gb.LinExpr(r,x),MINIMIZE)
        breakPoint.addConstrs(x[i] <= self.mdp.invUppBounds[i] for i in range(len(r)))
        breakPoint.addGenConstrMin(xMin, x)
        breakPoint.addConstr(xMin == 0)
        breakPoint.optimize()
        lBreakPoint     = breakPoint.objVal
        breakPoint.setObjective(gb.LinExpr(r,x),MAXIMIZE)
        breakPoint.optimize()
        rBreakPoint     = breakPoint.objVal
            
        #----------------------------------------------------------------------
        # The left most breakpoint
        allBreakPts.append(-self.OMEGA)
        
        #----------------------------------------------------------------------
        # The middle breakpoint (lBreakPoint, rBreakPoint)
        allBreakPts.append(lBreakPoint)
        allBreakPts.append(rBreakPoint)
        
        #----------------------------------------------------------------------
        # The right most breakpoint
        allBreakPts.append(self.OMEGA)
        allBreakPts     = list(np.sort(allBreakPts, kind = 'mergesort'))
        return allBreakPts
    

    """ 
        Finding the most violating constraint for a given VFA (row generation)
    """   
    def getMostViolatedConstraint(self,intercept, linCoef, BF_Coef,BF):
        
        #----------------------------------------------------------------------
        """
            The following program is defined on page 154 of [1], named Phi, and
            finds the most violating state-action pair.
        """
        MVC = gb.Model('MVC')
        MVC.setParam('OutputFlag',False)
        MVC.setParam('LogFile','Output/GJR/groubiLogFile.log')
        unlink('gurobi.log')
        MVC.setParam('Threads',self.numThreads) 
        #MVC.setParam('MIPGap'   , 0.05) 
        MVC.setParam('FeasibilityTol',1e-9)
        MVC.setParam('IntFeasTol',1e-9)
        MVC.setParam('NumericFocus',3)
        
        #----------------------------------------------------------------------
        # Variables    
        t      = MVC.addVar(ub = INFINITY,lb = 0,vtype=CONTINUOUS) 
        
        act    = MVC.addVars(self.RNG_DIM_X, 
                             ub     = [self.mdp.invUppBounds[_] for _ in self.RNG_DIM_X],
                             lb     = [0        for _ in self.RNG_DIM_X],
                             vtype  = CONTINUOUS)
                             
        state  = MVC.addVars(self.RNG_DIM_X,
                             ub     = [self.mdp.invUppBounds[_] for _ in self.RNG_DIM_X],
                             lb     = [0        for _ in self.RNG_DIM_X],
                             vtype  = CONTINUOUS)  
        
        nState = MVC.addVars(self.RNG_DIM_X,
                             ub     = [self.mdp.invUppBounds[_] for _ in self.RNG_DIM_X],
                             lb     = [0        for _ in self.RNG_DIM_X],
                             vtype  = CONTINUOUS)
        
        Y      = MVC.addVars(self.RNG_POW_SET,vtype   = BINARY) 
        
        U      = MVC.addVars(self.RNG_DIM_X,vtype     = BINARY)        
        
        Up     = MVC.addVars(self.RNG_DIM_X,vtype     = BINARY)               
        
        R      = MVC.addVars(self.RNG_DIM_X,vtype     = BINARY)              
        
        #----------------------------------------------------------------------
        # Objective function: major/minor cost  + affine VFA

        MVC.setObjective(quicksum(self.mdp.getFixCost(self.powSet[i])*Y[i] for i in self.RNG_POW_SET) - \
                         intercept*t - quicksum(linCoef[i]*act[i] for i in self.RNG_DIM_X),
                             MINIMIZE)
       
        #----------------------------------------------------------------------
        # Objective function: ridge VFA
        # Introduce some abbreviations 
        addVar      = MVC.addVar
        linExpr     = gb.LinExpr
        PWL         = MVC.setPWLObj
        addConstrs  = MVC.addConstrs        
        Svec        = [state[i]  for i in self.RNG_DIM_X]
        NSvec       = [nState[i] for i in self.RNG_DIM_X]
             
        #----------------------------------------------------------------------
        # If the VFA contains ridge basis functions (additional to affine VFA), then ...
        if not BF == None:
            #------------------------------------------------------------------
            # Define variables to model the VFA
            Hat_State  = [addVar(ub =    INFINITY,
                                 lb =   -INFINITY,
                                 vtype = CONTINUOUS)  for j in range(BF.numRidgeVec)]
       
            Hat_NState = [addVar(ub =    INFINITY,
                                 lb =   -INFINITY,
                                 vtype = CONTINUOUS)  for j in range(BF.numRidgeVec)]
            
            #------------------------------------------------------------------
            # Fpr each ridge vector, ...
            for j in range(BF.numRidgeVec): 
                #--------------------------------------------------------------
                # [VFA] has the value of PWL VFA defined by the ridge vectors on
                # the breakpoints. A PWL objective can be defined in Gurobi via
                # a table of x-y value.
                #--------------------------------------------------------------
                # The breakpoints at the boundary make the value of VFA euqal zero
                VFA = [0.0]
                for i in range(BF.numBreakPts[j]+2):
                    if i <= BF.numBreakPts[j]-1:
                        #-------------------------------------------------------
                        # The left most and right most ridge bases are zero
                        if i ==0  or i ==  BF.numBreakPts[j]-1:
                            VFA.append(0.0)
                        #-------------------------------------------------------
                        # If the coefficient of a basis  with ridge vector (j) 
                        # and  breakpoint (i) is small, make the VFA at this point zero
                        elif abs(BF_Coef[j][i]) < 1e-4:
                            VFA.append(0.0)
                        else:
                            VFA.append(BF_Coef[j][i])
                            
                #--------------------------------------------------------------
                # The breakpoints at the boundary make the value of VFA euqal zero
                VFA.append(0.0)
       
                #--------------------------------------------------------------
                # Define a PWL objective function               
                PWL(Hat_State[j], x=BF.breakPoints[j]    ,y= VFA)                    
                PWL(Hat_NState[j],x=BF.breakPoints[j]    ,y= [-VFA[i] for i in range(len(VFA))])
            
            #------------------------------------------------------------------
            # Define the inner product <r^j,s>
            addConstrs(Hat_State[j]  == linExpr( BF.ridgeVector[j],Svec)  for j in range(BF.numRidgeVec))
            addConstrs(Hat_NState[j] == linExpr( BF.ridgeVector[j],NSvec) for j in range(BF.numRidgeVec))
               
        #----------------------------------------------------------------------
        # Subset constraints 
        MVC.addConstr(quicksum(Y[_] for _ in self.RNG_POW_SET) == 1.0)
        
        MVC.addConstrs(R[i] == quicksum(Y[j] for j in whereIsElementInPowerSet(self.powSet,i)) \
                                       for i in self.RNG_DIM_X)
        
        MVC.addConstrs(act[i] <= self.mdp.invUppBounds[i]*R[i] for i in self.RNG_DIM_X)
            
        #----------------------------------------------------------------------
        # State-actions constraints
        MVC.addConstrs(nState[i]  + self.mdp.consumptionRate[i]*t == state[i] + act[i] \
                               for i in self.RNG_DIM_X)
        
        MVC.addConstrs(state[i] + act[i] <= self.mdp.invUppBounds[i] for i in self.RNG_DIM_X)
        
        MVC.addConstr(quicksum(act[i] for i in self.RNG_DIM_X) <= self.mdp.maxOrder)
        
        #----------------------------------------------------------------------
        # Just-in-time constraints 
        ### MVC.addConstrs(state[i] <= mdp['invUppBounds'][i]*(1-U[i]) for i in RNG_DIM_X)
        MVC.addConstrs(state[i] + U[i]*self.mdp.invUppBounds[i] <= self.mdp.invUppBounds [i] for i in self.RNG_DIM_X)
        
        MVC.addConstr(quicksum(U[i] for i in self.RNG_DIM_X) >= 1.0)
       
        ### MVC.addConstrs((nState[i] <= mdp['invUppBounds'][i]*(1-Up[i])) for i in RNG_DIM_X)  
        MVC.addConstrs(nState[i] +self.mdp.invUppBounds[i]*Up[i]
                                <= self.mdp.invUppBounds[i] for i in self.RNG_DIM_X)  
        
        MVC.addConstr(quicksum(Up[i] for i in self.RNG_DIM_X) >= 1.0)
        
        MVC.addConstrs(U[i] <= R[i] for i in self.RNG_DIM_X)  
        
        #----------------------------------------------------------------------
        # Optimize The Program
        MVC.update()
        MVC.optimize()
        
        #----------------------------------------------------------------------
        # Make the output ready & return MVC state, action, value, and next state
        MVC_state    = np.round([dropZeros(state[i].X )     for i in self.RNG_DIM_X],5)
        MVC_action   = np.round([dropZeros(act[i].X)         for i in self.RNG_DIM_X],5)
        MVC_NState   = self.mdp.getNextState(MVC_state,MVC_action) #np.round([dropZeros(nState[i].X)         for i in RNG_DIM_X],5)#
        
        return MVC_state, MVC_action, MVC.objVal, MVC_NState


    
    """ 
        ALP model to compute VFA in [1]
    """   
    def ALP_GJR(self,   BF,
                        stateList,
                        actionList,
                        nStateList, 
                        useVFA          = True,
                        basisItr        = 0,
                        model           = None,
                        addedS          = None,
                        addedA          = None): 
    
        #----------------------------------------------------------------------
        # If a Gurobi model does not exist, then it set up
        if model == None:  
            
            #------------------------------------------------------------------
            # Compute immediate cost
            getCost     = self.mdp.getExpectedCost
            pool        = Pool(self.numThreads)
            COST        = pool.starmap(getCost, zip(stateList,actionList))
            pool.close()
            pool.join()
            
            #------------------------------------------------------------------
            # Compute transition time
            transTime   = self.mdp.transTime
            pool        = Pool(self.numThreads)
            TRAN_TIME   = pool.starmap(transTime, zip(stateList,actionList))
            pool.close()
            pool.join()
            
            #------------------------------------------------------------------
            # Gurobi model defining an ALP
            ALP = gb.Model('ALP')
            ALP.setParam('OutputFlag',False)
            ALP.setParam('LogFile','Output/GJR/groubiLogFile.log')
            unlink('gurobi.log')
            ALP.setParam('Threads',self.numThreads) 
            ALP.setParam('NumericFocus',3)
            ALP.setParam('FeasibilityTol',1e-9)
            
            #------------------------------------------------------------------
            # Variables
            intercept   = ALP.addVar(ub      =  INFINITY,
                                    lb       = -INFINITY,
                                    vtype    = CONTINUOUS,
                                    name     = 'intercept')
            linCoefVar  = ALP.addVars(self.RNG_DIM_X,
                                     ub     = [ INFINITY for _ in self.RNG_DIM_X],
                                     lb     = [-INFINITY for _ in self.RNG_DIM_X],
                                     vtype  = CONTINUOUS,
                                     name   = 'linCoefVar')
            
            linCoef     = [linCoefVar[i] for i in self.RNG_DIM_X]
            
            #------------------------------------------------------------------
            # Set ALP objective
            ALP.setObjective(intercept + gb.LinExpr(self.mdp.consumptionRate,linCoef), MAXIMIZE)  
                
            #------------------------------------------------------------------
            # If ridge bases should be added, ...
            if useVFA:
                
                #--------------------------------------------------------------
                # Define coefficients associated with ridge vectors
                numRidgeVec     = BF.numRidgeVec
                numBreakPts     = BF.numBreakPts
                RNG_BF_NUMBER   = [(j,i) for j in range(numRidgeVec) for i in range(numBreakPts[j])] 
                BF_CoefVar      = [[ALP.addVar(ub    =  INFINITY,
                                               lb    = -INFINITY ,
                                               vtype = CONTINUOUS,
                                               name  = 'BF_CoefVar') 
                                        for i in range(numBreakPts[j])]
                                          for j in range(numRidgeVec)]
            
                #--------------------------------------------------------------
                # The coefficients of ridge bases at the boundary are zero
                for j in range(numRidgeVec):
                    ALP.addConstr(BF_CoefVar[j][0]                  == 0.0)
                    ALP.addConstr(BF_CoefVar[j][numBreakPts[j]-1]   == 0.0)
                    
                #--------------------------------------------------------------    
                # ALP constraints
                delta   = BF.deltaHat
                dual    = ALP.addConstrs(TRAN_TIME[itr]*intercept +\
                                             gb.LinExpr(actionList[itr],linCoef) +\
                                             quicksum(BF_CoefVar[j][i]*delta(j,i,nStateList[itr],stateList[itr]) for (j,i) in RNG_BF_NUMBER) \
                                             <= COST[itr]  for itr in range(len(stateList))) 
        
                #--------------------------------------------------------------  
                # Solve ALP & return optimal coefficients
                ALP.update() 
                ALP.optimize()
                
                #--------------------------------------------------------------  
                # Compute dual variables used for updating bases and return VFA
                optVal      = ALP.objVal
                DUAL_VAL    = [dual[i].getAttr('Pi')     for i in  range(len(stateList))]
                
                return intercept.X, \
                       [linCoef[_].X for _ in self.RNG_DIM_X], \
                       [[BF_CoefVar[j][i].X for i in range(numBreakPts[j])] for j in range(numRidgeVec)],\
                       optVal, \
                       DUAL_VAL, \
                       ALP
            
            #------------------------------------------------------------------
            # If we only have affine VFA, ...
            else:                           
                #--------------------------------------------------------------    
                # ALP constraints with affine VFA
                dual    = ALP.addConstrs(TRAN_TIME[itr]*intercept +\
                                       gb.LinExpr(actionList[itr],linCoef) \
                                         <=  COST[itr] for itr in range(len(stateList)))  

                #--------------------------------------------------------------  
                # Compute dual variables used for updating bases and return VFA
                ALP.update()       
                ALP.optimize() 
                intercept.X 
                DUAL_VAL = [(dual[i].getAttr('Pi'))     for i in  range(len(stateList))]
                            
                return intercept.X, \
                        [linCoef[_].X for _ in self.RNG_DIM_X], \
                        [ ],\
                        ALP.objVal, \
                        DUAL_VAL, \
                        ALP
                        
        #----------------------------------------------------------------------
        # If a Gurobi model already exists, then load it
        else:
            
            #------------------------------------------------------------------
            # Set Gurobi parameters
            model.setParam('FeasibilityTol',1e-9)
            model.setParam('NumericFocus',3)
            
            #------------------------------------------------------------------
            # If there is no basis model, then define it
            if not BF == None:
                
                #--------------------------------------------------------------
                # Retrieve decision variables of the ALP model, that are, bases
                # weights
                numRidgeVec     = BF.numRidgeVec
                numBreakPts     = BF.numBreakPts
                RNG_BF_NUMBER   = [(j,i) for j in range(numRidgeVec) for i in range(numBreakPts[j])]
                vars            = model.getVars()
                intercept       =  vars[0]
                linCoef         = [vars[_+1] for _ in self.RNG_DIM_X]
                BF_Coef         = [vars[_+1+self.dimX] for _ in range(len(RNG_BF_NUMBER))]  
                BF_CoefVar      = [[None for i in range(numBreakPts[j])] for j in range(numRidgeVec)]
                
                #--------------------------------------------------------------
                # Cast coefficients variable
                k   = 0
                for j in range(numRidgeVec):
                    for i in range(numBreakPts[j]):
                        BF_CoefVar[j][i] = BF_Coef[k]
                        k+=1

                #--------------------------------------------------------------
                # Define constraints of ALP at the new state-action pair 
                # obtained from the MVC
                for j in range(numRidgeVec):
                    model.addConstr(BF_CoefVar[j][0]                    == 0.0)
                    model.addConstr(BF_CoefVar[j][numBreakPts[j]-1]     == 0.0)
            
                #--------------------------------------------------------------
                # Define ALP constraints
                addedNs     = self.mdp.getNextState(addedS,addedA)
                delta       = BF.deltaHat
                model.addConstr(self.mdp.transTime(addedS,addedA)*intercept +\
                                        gb.LinExpr(addedA,linCoef) +\
                                        quicksum(BF_CoefVar[j][i]*delta(j,i,addedNs,addedS) for (j,i) in RNG_BF_NUMBER) \
                                        <= self.mdp.getExpectedCost(addedS,addedA)) 

                #--------------------------------------------------------------  
                # Compute dual variables used for updating bases and return VFA                
                model.update() 
                model.optimize()
                optVal      = model.objVal
                dual        = model.getConstrs()
                DUAL_VAL    = [dual[i].getAttr('Pi') 
                                   for i in  range(len(stateList))]
                
                return intercept.X, \
                       [linCoef[_].X for _ in self.RNG_DIM_X], \
                       [[BF_CoefVar[j][i].X for i in range(numBreakPts[j])] for j in range(numRidgeVec)],\
                       optVal,\
                       DUAL_VAL,\
                       model 
                       
            #------------------------------------------------------------------
            # If there is  basis model, then load it    
            else:
                vars        = model.getVars()
                intercept   =  vars[0]
                linCoef     = [vars[_+1] for _ in self.RNG_DIM_X]
       
                #--------------------------------------------------------------
                # Define constraints of ALP at the new state-action pair 
                model.addConstr(self.transTime(addedS,addedA)*intercept +\
                                        gb.LinExpr(addedA,linCoef) 
                                        <= self.mdp.getExpectedCost(addedS,addedA)) 
                
                #--------------------------------------------------------------  
                # Compute dual variables used for updating bases and return VFA 
                model.update() 
                model.optimize()
                optVal      = model.objVal
                dual        = model.getConstrs()
                DUAL_VAL    = [dual[i].getAttr('Pi')     for i in  range(len(stateList))]
    
                return intercept.X,\
                       [linCoef[_].X for _ in self.RNG_DIM_X],\
                       [ ],\
                       optVal,\
                       DUAL_VAL,\
                       model
    
        
    """ 
        Dual ALP model used for computing cyclic policies, flow imbalance,
        and support set
    """   
    def dualALP_GJR(self,Dual, stateList, actionList, nStateList ):
        #----------------------------------------------------------------------  
        # Compute the support set of Z (see Q in [1])
        support = []
        for (i,x) in enumerate(Dual):
            if x >  1e-8:
                support.append([i, list(stateList[i]), list(actionList[i]), Dual[i]])   
        
        #----------------------------------------------------------------------         
        # States visited under dual solution 
        Q       = []
        for (i,s,a,z_s_a) in support:
            s   = np.asarray(s)
            a   = np.asarray(a)
            nS  = np.asarray(self.mdp.getNextState(s,a))
            if not arreqclose_in_list(s,Q):
                Q.append(s)
            if not arreqclose_in_list(nS,Q):
                Q.append(nS)
                
        #----------------------------------------------------------------------  
        # Compute the flow imbalance on page 160 in [1]
        flowViolation = []
        for x in Q:
            flow_out  = 0.0
            flow_in   = 0.0
            for (d,s,a,z_s_a) in support: 
                if np.allclose(x,np.asarray(s),atol=1e-8):
                    flow_in  += z_s_a  
                    
                if np.allclose(x,self.mdp.getNextState(s,a),atol=1e-8):
                    flow_out += z_s_a
    
            flowViolation.append([x,abs(flow_in-flow_out)])
        
        #---------------------------------------------------------------------- 
        flowViolation   = sorted(flowViolation,key = lambda x: x[1],reverse=True)
        Q               = [flowViolation[i][0] for i in range(len(Q))]
        
        #---------------------------------------------------------------------- 
        # Check if a cyclic schedule is detected
        def isCyclic(lst):
            l   = len(lst)
            nS  = self.mdp.getNextState(np.asarray(lst[l-1][1]),np.asarray(lst[l-1][2]))
            
            if not np.allclose(np.asarray(lst[0][1]),nS,atol = 1e-4):
                return False
            
            for i in range(0,l-1):
                ns=np.asarray(self.mdp.getNextState(lst[i][1],lst[i][2]))
                if not np.allclose(ns,np.asarray(lst[i+1][1]) ,atol = 1e-4):
                    return False
                
            return True    
        
        #---------------------------------------------------------------------- 
        # Cycle detetion 
        cycLen          = float('inf')  
        isDualCyclic    = False
        I               = -1
        
        #---------------------------------------------------------------------- 
        # Check if a cycle exists in the states achieved in the support set
        for (i,y,a,z_y_a) in support:
            J = -1
            I += 1
            for (j,x,u,z_x_u) in support:
                J +=1
                if j > i:
                    subSupport = [support[_] for _ in range(I,J)]
                    if  len(subSupport) >  1:
    
                        if isCyclic(subSupport) :
                            isDualCyclic = True
                            cycLen = len(subSupport)
                            break   
            if isDualCyclic:
                break
        
        #---------------------------------------------------------------------- 
        # Compute the cost of a cyclic/non-cyclic policy
        aveCost       = float('inf')  
        if isDualCyclic:
            totalTime   = 0.0
            cumCost     = 0.0
            for (i,x,b,z_y_a) in subSupport:
                cumCost     += self.mdp.getCost(x,b)
                totalTime   += self.mdp.transTime(x,b)                    
    
            aveCost         = cumCost/totalTime        
            
        #---------------------------------------------------------------------- 
        # Return results 
        return support, Q, flowViolation, isDualCyclic, aveCost,cycLen
    
    

    """
        Ridge basis function generation in Algorithm 3 of [1]
    """
    def getNewHatBases(self,basis_Itr,support, Q, flowViolationSorted):  
        #---------------------------------------------------------------------- 
        # Ridge unit vector, ridge pair vector, ...
        U_unit              = None
        U_non_unit          = None
        ridgeNeeded         = 0
        lenQ                = len(Q)
        equalityThreshold   = 1e-2
        #----------------------------------------------------------------------
        # K: how many bases to generate at most
        K               = 0
        
        #----------------------------------------------------------------------
        # Iterate over set Q defined on page 160 of [1] 
        for q in range(lenQ):
            
            #------------------------------------------------------------------
            # If already sampled 10 bases, then stop
            if K>=9:
                break
            
            #------------------------------------------------------------------
            #Set E and O refer to the index of the unit and pair ridge vectors
            E           = self.hatSetting['indexUnitVec']
            O           = list(set(range(self.hatSetting['numRidgeVec']))- set(E))  
            U_unit      = self.getUniqueMaps(Q,q,E)
            U_non_unit  = self.getUniqueMaps(Q,q,O)
            
            #------------------------------------------------------------------
            # If there is a unique map for existing unit ridge vectors (1st 
            # if statement in Algorithm 3 of [1]), then add a new breakpoint to
            # get a new basis.
            if not U_unit == []:
                #--------------------------------------------------------------
                # Uniformly pick a ridge vector
                idx         = U_unit[randint(0,len(U_unit)).rvs(1)[0]]
                prevBrPts   = list(self.hatSetting['breakPoints'][idx])
                
                #--------------------------------------------------------------
                # Set the corresponding breakpoint; delta is the difference 
                # between the br and existing breakpoints
                br          = round(np.dot(self.hatSetting['ridgeVector'][idx],Q[q]),5)
                delta       = [abs(br - _) for _ in prevBrPts]
                
                #--------------------------------------------------------------
                # If the new basis does not hurt the unique map (see Definition
                # 2 in [1]), append the new ridge vector
                if min(delta)>equalityThreshold:
                    prevBrPts.append(br)
                    prevBrPts   = np.unique(prevBrPts)
                    prevBrPts   = list(np.sort(prevBrPts, kind = 'mergesort'))
                    self.hatSetting['breakPoints'][idx]     = prevBrPts        
                    self.hatSetting.update({'numBreakPts'   : [len(self.hatSetting['breakPoints'][_])-2 for _ in range(self.hatSetting['numRidgeVec'])] })
                    K+=1
                    
            #------------------------------------------------------------------
            # If there is a unique map for existing nonunit ridge vectors (2nd 
            # if statement in Algorithm 3 of [1]), then add a new breakpoint to
            # get a new basis. 
            elif not U_non_unit == []:
                #--------------------------------------------------------------
                # Uniformly pick a ridge vector
                idx         = U_non_unit[randint(0,len(U_non_unit)).rvs(1)[0]]
                prevBrPts   = list(self.hatSetting['breakPoints'][idx])
                
                #--------------------------------------------------------------
                # Set the corresponding breakpoint; delta is the difference 
                # between the br and existing breakpoints
                br          = round(np.dot(self.hatSetting['ridgeVector'][idx],Q[q]),5)
                delta       = [abs(br - _) for _ in prevBrPts]
               
                #--------------------------------------------------------------
                # If the new basis does not hurt the unique map (see Definition
                # 2 in [1]), append the new ridge vector
                if min(delta)>equalityThreshold:
                    prevBrPts.append(br)
                    prevBrPts   = np.unique(prevBrPts)
                    prevBrPts   = list(np.sort(prevBrPts, kind = 'mergesort'))
                    self.hatSetting['breakPoints'][idx]     = prevBrPts
                    self.hatSetting.update({'numBreakPts'   : [len(self.hatSetting['breakPoints'][_])-2 for _ in range(self.hatSetting['numRidgeVec'])] }) 
                    K+=1
            
            #------------------------------------------------------------------
            # If we cannot add a breakpoint and obtain a unique map, then add a
            # new ridge vector
            else:
                #--------------------------------------------------------------
                # Increment the number of ridge vector generated
                ridgeNeeded     +=1
                
                #--------------------------------------------------------------
                # Index sets corresponding to the unit and nonunit ridge vectors
                E                = self.hatSetting['indexUnitVec']
                O                = list(set(range(self.hatSetting['numRidgeVec']))- set(E))
                
                #--------------------------------------------------------------
                # Page 160 of [1], solve linear complementarity problem
                idx              = list(set(range(lenQ)) - set([q]))
                ridgeModel       = gb.Model('findRidge')
                ridgeModel.setParam('OutputFlag',False)
                ridgeModel.setParam('LogFile','Output/GJR/groubiLogFile.log')
                unlink('gurobi.log')
                ridgeModel.setParam('NumericFocus',3)
                ridgeModel.setParam('FeasibilityTol',1e-9)
                ridgeModel.setParam('MIPGap',0.00)
                
                #--------------------------------------------------------------
                # Please see the explanation in the 1st paragraph of the 2nd 
                # column of page 160 in [1]. Since we use \tilde{Q} ={x} with a
                # single state, the linear complementarity problem is easy to solve.
                theta       = ridgeModel.addVar(lb        = 0.0,
                                                ub        = INFINITY,
                                                vtype     = CONTINUOUS)
                
                alpha       = ridgeModel.addVars(idx,
                                                 lb       = [-INFINITY for _ in idx],
                                                 ub       = [ INFINITY for _ in idx],
                                                 vtype    = CONTINUOUS)
                
                alphaAbs    = ridgeModel.addVars(idx,
                                                 lb       = [0.0       for _ in idx],
                                                 ub       = [ INFINITY for _ in idx],
                                                 vtype    = CONTINUOUS)
                
                beta        = ridgeModel.addVars(idx,
                                                 lb       = [-INFINITY for _ in idx],
                                                 ub       = [ INFINITY for _ in idx],
                                                 vtype    = CONTINUOUS)
                
                betaAbs     = ridgeModel.addVars(idx,
                                                 lb       = [ 0.0      for _ in idx],
                                                 ub       = [ INFINITY for _ in idx],
                                                 vtype    = CONTINUOUS)
                    
                maxAbs      = ridgeModel.addVars(idx,
                                                 lb       = [ 0.0      for _ in idx],
                                                 ub       =[ INFINITY for _ in idx],
                                                 vtype    = CONTINUOUS)
                
                rVec        = [ridgeModel.addVar(lb       = -1.0,
                                                 ub       =  1.0,
                                                 vtype    = CONTINUOUS)
                                   for _ in range(self.mdp.dimX)]
                
                #--------------------------------------------------------------
                # Set the objective function
                ridgeModel.setObjective(theta,MAXIMIZE)
                
                #--------------------------------------------------------------
                # Add constraints
                fixState    = np.asarray(Q[q])
                for qq in idx:
                    ridgeModel.addConstr(theta           <= alpha[qq] +beta[qq])
                    ridgeModel.addConstr(LinExpr(np.asarray(Q[qq])-fixState,rVec) \
                                                         <= alpha[qq] - beta[qq])
                        
                    #----------------------------------------------------------
                    # Modeling constraint alpha*beta = 0 as follows
                    ridgeModel.addConstr(alphaAbs[qq]   == gb.abs_(alpha[qq]))
                    ridgeModel.addConstr(betaAbs[qq]    == gb.abs_(beta[qq] ))
                    ridgeModel.addConstr(maxAbs[qq]     == gb.max_(alphaAbs[qq],betaAbs[qq]))
                    ridgeModel.addConstr(alphaAbs[qq] + betaAbs[qq] 
                                                        <= maxAbs[qq])
                
                #--------------------------------------------------------------
                # Solve the model to get the optimal ridge vector
                ridgeModel.update()
                ridgeModel.optimize()
                optRidge = np.round([rVec[_].X for _ in range(self.mdp.dimX)],5)
                
                #--------------------------------------------------------------
                # Construct breaking points for the new ridge vector 
                # (see Page 157 of [1])
                ALPHA       = []
                ZETA        = []
                innerProd   = np.dot(optRidge,Q[q])
                
                #--------------------------------------------------------------
                # Append those breakpoints giving us unique map
                for d in Q:
                    if np.around(np.dot(optRidge,d),5)   < innerProd +equalityThreshold :
                            ALPHA.append(np.round(np.dot(optRidge,d),5) )
                    if np.around(np.dot(optRidge,d),5)    +equalityThreshold > innerProd :
                            ZETA.append(np.around(np.dot(optRidge,d),5) )
                     
                #--------------------------------------------------------------
                # Set the middle breakpoint
                Br_center           = innerProd
                
                #--------------------------------------------------------------
                # If there is not a unique map, then define extreme breakpoints
                if ALPHA == [] and ZETA  == []:
                    Break           = [-self.OMEGA,
                                        Br_center,
                                        self.OMEGA]
                
                #--------------------------------------------------------------
                # If there is a unique map, 
                elif ALPHA == [] and (not ZETA  == []):
                    #----------------------------------------------------------
                    # If it is distinct from the middle breakpoint, then add 
                    # the right breakpoint
                    if abs(Br_center -   min(ZETA)) > equalityThreshold:
                        Break       = [-self.OMEGA,
                                        Br_center,
                                        min(ZETA),
                                        self.OMEGA]
                    
                    #----------------------------------------------------------
                    # Otherwise, use the extreme breakpoints
                    else:
                        Break       = [-self.OMEGA,
                                        Br_center,
                                        self.OMEGA]
                
                #--------------------------------------------------------------
                # If there is a unique map, 
                elif (not ALPHA == []) and (not ZETA  == []):
                    #----------------------------------------------------------
                    # If they are distinct from the middle breakpoint, then add
                    # both left and right breakpoints
                    if abs(max(ALPHA) -  Br_center) > equalityThreshold \
                        and abs(min(ZETA) -  Br_center)  > equalityThreshold:
                        Break       = [-self.OMEGA,
                                        max(ALPHA),
                                        Br_center, 
                                        min(ZETA),
                                        self.OMEGA]

                    #----------------------------------------------------------
                    # If only the right breakpoint is distinct from the middle
                    # one, then add the right one                        
                    elif abs(max(ALPHA) -  Br_center) < equalityThreshold \
                        and abs(min(ZETA) -  Br_center) > equalityThreshold:
                        Break       = [-self.OMEGA,
                                        Br_center,
                                        min(ZETA),
                                        self.OMEGA]
                        
                    #----------------------------------------------------------
                    # If only the left breakpoint is distinct from the middle
                    # one, then add the left one
                    elif abs(max(ALPHA) -  Br_center) > equalityThreshold \
                        and abs(min(ZETA) -  Br_center) < equalityThreshold:
                            Break   = [-self.OMEGA,
                                        max(ALPHA),
                                        Br_center,
                                        self.OMEGA]
                    
                    #----------------------------------------------------------
                    # If both are the same as the middle breakpoint, use the 
                    # extreme breakpoints   
                    else:
                        Break       = [-self.OMEGA,
                                        Br_center,
                                        self.OMEGA]

                #--------------------------------------------------------------
                # If there is a unique map,                         
                elif (not ALPHA  == []) and ZETA  == []:
                    #----------------------------------------------------------
                    # If it is distinct from the middle breakpoint, then add 
                    # the left breakpoint
                    if abs(Br_center -   max(ALPHA)) > equalityThreshold:
                        Break       = [-self.OMEGA,
                                        np.around(max(ALPHA)),
                                        Br_center,
                                        self.OMEGA]
                        
                    #----------------------------------------------------------
                    # Otherwise, use the extreme breakpoints
                    else:
                        Break       = [-self.OMEGA,
                                        Br_center,
                                        self.OMEGA]
                
                #--------------------------------------------------------------
                # Store the new ridge vecto and breakpoints
                prevRidge = self.hatSetting['ridgeVector']
                if not arreqclose_in_list(np.asarray(optRidge),prevRidge):
                    #----------------------------------------------------------
                    prevRidge.append(np.asarray(optRidge))
                    Break = list(np.round(np.sort(Break, kind = 'mergesort'),5))
                    K+=1
               
                    #----------------------------------------------------------
                    self.hatSetting.update({'ridgeVector'    : prevRidge})
                    self.hatSetting.update({'numRidgeVec'    : len(self.hatSetting['ridgeVector'])})
                    self.hatSetting['breakPoints'].append(Break)
                    self.hatSetting.update({'numBreakPts'   : [len(self.hatSetting['breakPoints'][_])-2 for _ in range(self.hatSetting['numRidgeVec'])] })  
                    E = self.hatSetting['indexUnitVec']
                    O = list(set(range(self.hatSetting['numRidgeVec']))- set(E))
                    
                    #----------------------------------------------------------
                    # Show a message if a numerical issue happened
                    if self.getUniqueMaps(Q,q,E) == [] and self.getUniqueMaps(Q,q,O)==[]:
                        msg = "WARNING! The generated ridge vector and breakpoints do not cut the lead to a unique map. One may fix this by modifying variable [equalityThreshold] in function [getNewHatBases] of class [GJR_Benchmark]."
                        print('\n\n'+textwrap.TextWrapper(width=50).fill(text=msg)+'\n\n')
        
        #----------------------------------------------------------------------
        # Return the generated bases
        return HatBasis(self.hatSetting), ridgeNeeded



    """ 
        This function return indexes in a set that are defining a unique map.
        Please see U_d^{j} on page 160 of [1].
    """    
    def getUniqueMaps(self,Q,q,idxSet):
        U           = []
        for i in idxSet:
            r       = self.hatSetting['ridgeVector'][i]
            val     = dropZeros(np.dot(r,Q[q]))
            flag    = True
            for k in range(len(Q)):
                if not q == k:
                    vval = dropZeros(np.dot(Q[k],r))
                    if abs(val - vval) <1e-6:
                        flag = False  
            if flag:
                 U.append(i)             
        return U
    

    """ 
        Greedy policy optimization (PD) on page 155 of [1]
    """       
    def getGreedyAction(self,state_ini, BF ,intercept, linCoef, BF_Coef):
        #----------------------------------------------------------------------
        # Configure the Gurobi model
        PD = gb.Model('PD')
        PD.setParam('OutputFlag',False)
        PD.setParam('LogFile','Output/GJR/groubiLogFile.log')
        unlink('gurobi.log')
        PD.setParam('MIPGap',0.01)
        PD.setParam('Threads',self.numThreads)
        PD.setParam('FeasibilityTol',1e-9)
        PD.setParam('NumericFocus',3)     
        
        #----------------------------------------------------------------------
        # Defining some ranges and some list of indices
        horizon         = self.ub['roleHorizon']
        rangeHorizon    = range(horizon)
        rangeHorizon_1  = range(horizon-1)
        idx             = [(r,d) for r in rangeHorizon   for d in       self.RNG_DIM_X]
        idx_1           = [(r,d) for r in rangeHorizon_1 for d in       self.RNG_DIM_X]
        psIdx           = [(r,i) for r in rangeHorizon_1 for i in       self.RNG_POW_SET]
    
        #----------------------------------------------------------------------
        # Time upper bound
        upT    = INFINITY
        
        #----------------------------------------------------------------------
        # Vector of transition time decision variables
        t      = PD.addVars(rangeHorizon_1,
                            ub      = [upT for _ in rangeHorizon_1],
                            lb      = [0        for _ in rangeHorizon_1],
                            vtype   = CONTINUOUS) 
             
        #----------------------------------------------------------------------
        # Vector of action decision variables
        act    = PD.addVars(idx_1, 
                            ub      = [self.mdp.invUppBounds[d] for (r,d) in idx_1],
                            lb      = [0        for _ in idx_1],
                            vtype   = CONTINUOUS)
    
        #----------------------------------------------------------------------
        # Vector of next state decision variables       
        nState = PD.addVars(idx,   
                            ub      = [self.mdp.invUppBounds[d] for (r,d) in idx],
                            lb      = [0 for _ in idx],
                            vtype   = CONTINUOUS)
        
        #----------------------------------------------------------------------
        # Auxiliary decision variables that are modeling the MDP transition 
        # function and replenishment time   
        Y      = PD.addVars(psIdx, vtype   = BINARY) 
        U      = PD.addVars(idx,   vtype   = BINARY)     
        R      = PD.addVars(idx_1, vtype   = BINARY)              
    
        
        #----------------------------------------------------------------------
        """ 1) Set the linear part of the PD's objective """
        PD.setObjective(quicksum(quicksum(self.mdp.getFixCost(self.powSet[i])*Y[r,i]\
                                         for i in self.RNG_POW_SET) - \
                                 intercept*t[r] - quicksum(linCoef[i]*act[r,i] 
                                                           for i in self.RNG_DIM_X) \
                                 for r in rangeHorizon_1),
                        MINIMIZE)
       
        varLastState    = [nState[horizon-1,_] for _ in self.RNG_DIM_X]
        
        #----------------------------------------------------------------------
        # If we have affine + ridge bases
        if not BF == None:
             #-----------------------------------------------------------------
             """ 2) Set the PWL part of the PD's objective """
             addVar     = PD.addVar
             linExpr    = gb.LinExpr
             PWL        = PD.setPWLObj
             addConstrs = PD.addConstrs     
             Hat_NState = [addVar(ub        =    INFINITY,
                                  lb        =   -INFINITY,
                                  vtype     = CONTINUOUS)
                           for j in range(BF.numRidgeVec)]
             
             #-----------------------------------------------------------------
             # For each ridge vector,
             for j in range(BF.numRidgeVec):
                 #-------------------------------------------------------------
                 # Pull out the corresponding breakpoint. Set the value of VFA
                 # at the boundary of the state space to zero
                 B           = BF.breakPoints[j]    
                 VFA         = [0.0]
                 
                 #-------------------------------------------------------------
                 # For each breakpoint,
                 for i in range(BF.numBreakPts[j]+2):
                     
                    #----------------------------------------------------------
                    # The If statement corresponds to the value of VFA at the 
                    # boundary of the state space to zero
                    if i <= BF.numBreakPts[j]-1:
                        #------------------------------------------------------
                        # When coefficients of basis function are close to zero,
                        # make them zero
                        if abs(BF_Coef[j][i]) < 1e-4:
                            VFA.append(0.0)
                        
                        #------------------------------------------------------
                        # Add the value of VFA
                        else:
                            VFA.append(-BF_Coef[j][i])
                 #-------------------------------------------------------------
                 # Value of the VFA at the boundary of the state space is zero          
                 VFA.append(0.0)
                 
                 #-------------------------------------------------------------
                 # Set the PWL part of the objective function
                 PWL(Hat_NState[j],x=B,y=VFA)
                 
             #-----------------------------------------------------------------
             # Set the value of hat bases
             addConstrs(Hat_NState[j] == linExpr(BF.ridgeVector[j],varLastState) for j in range(BF.numRidgeVec))
        PD.update()
             
        #----------------------------------------------------------------------
        # Subset Constraints 
        PD.addConstrs(quicksum(Y[r,i] for i in self.RNG_POW_SET) == 1.0 for r in rangeHorizon_1)
             
        PD.addConstrs(R[r,i] == quicksum(Y[r,j] for j in whereIsElementInPowerSet(self.powSet,i)) for \
                                         i in self.RNG_DIM_X for r in rangeHorizon_1)
             
        PD.addConstrs(act[r,i] <= self.mdp.invUppBounds[i]*R[r,i] for i in self.RNG_DIM_X for \
                                             r in rangeHorizon_1)
        
        #----------------------------------------------------------------------
        # State-actions Constraints
        PD.addConstrs(nState[0,i] == float(state_ini[i]) for i in self.RNG_DIM_X) 
        PD.addConstrs(nState[r+1,i] +self.mdp.consumptionRate[i]*t[r] == nState[r,i] + act[r,i] \
                                     for i in self.RNG_DIM_X for r in rangeHorizon_1)
        
        PD.addConstrs(nState[r,i] + act[r,i] <= self.mdp.invUppBounds[i] for i in
                                        self.RNG_DIM_X for r in rangeHorizon_1)
             
        PD.addConstrs((quicksum(act[r,i] for i in self.RNG_DIM_X)) <= self.mdp.maxOrder for r in rangeHorizon_1)
             
        #----------------------------------------------------------------------
        # Just-in-time Constraints 
        PD.addConstrs(nState[r,i] +U[r,i]*self.mdp.invUppBounds[i] <= self.mdp.invUppBounds[i] \
                          for r in rangeHorizon for i in self.RNG_DIM_X)
        PD.addConstrs(quicksum(U[r,i] for i in self.RNG_DIM_X) >= 1.0 for r in rangeHorizon)
    
        PD.addConstrs(U[r,i] <= R[r,i] for i in self.RNG_DIM_X for r in rangeHorizon_1)  
        
        #----------------------------------------------------------------------
        # Optimize this MILP
        PD.update()             
        PD.optimize()
        return [act[0,i].X for i in self.RNG_DIM_X]


    """
        Simulating policy and computing upper bound
    """
    def getUpperBound(self,state_ini, BF ,intercept, linCoef, BF_Coef):
        #----------------------------------------------------------------------
        # List of state, action, and transition time 
        visitedStates           = []
        visitedActions          = []
        visitedTimes            = []
        
        #----------------------------------------------------------------------
        # Initial state
        state                   = state_ini
        
        #----------------------------------------------------------------------
        # Does a cyclic policy found? Save index of the beginning of the cycle
        cyclicPolFound          = False
        cumCost                 = 0
        
        #----------------------------------------------------------------------
        # Total time and cost 
        totalTime               = 0
        cycleStartStateIndex    = 0

        #----------------------------------------------------------------------
        # Iterate over the trajectory length
        for traj in range(self.ub['trajLen']): 
            
            #------------------------------------------------------------------
            # Get the optimal action 
            optAct  = self.getGreedyAction(state, BF ,intercept, linCoef, BF_Coef)
            t       = self.mdp.transTime(state,optAct)
            
            #------------------------------------------------------------------
            # Store state, action, and transition time
            visitedStates.append(state)
            visitedActions.append(optAct)
            visitedTimes.append(t)   
            
            #------------------------------------------------------------------
            state=np.add(np.add(state,optAct), -t*self.mdp.consumptionRate) 
                  
            #------------------------------------------------------------------
            # Is there any cycle?
            R = [np.linalg.norm(state-x,ord=np.inf)<1e-14 for x in visitedStates]
            if  any(R):
                cyclicPolFound = True  
                cycleStartStateIndex = R.index(True)
                break
        
        #----------------------------------------------------------------------
        # Iterate over the cycle and compute the cost of the cycle    
        for _ in range(cycleStartStateIndex,len(visitedStates)):
            totalTime += visitedTimes[_]
            cumCost   += self.mdp.getExpectedCost(visitedStates[_],visitedActions[_])
        
        #----------------------------------------------------------------------
        # Set the average time
        if abs(totalTime) <1e-6:
            AveCost = float('inf')
        else:
            AveCost = (cumCost/totalTime)
        
        #----------------------------------------------------------------------
        # Return the results 
        return cyclicPolFound,  len(visitedStates) - cycleStartStateIndex    , AveCost
    

    def outPut(self,error,logger,timeLogger,resTable,timeGapTable):
        #----------------------------------------------------------------------
        # If there is not an error, save results
        if error == 0:
            logger.close() 
            timeLogger.close()
            resTable        = resTable[~ np.all(resTable==0, axis=1)]     
            timeGapTable    = timeGapTable[~ np.all(timeGapTable==0, axis=1)]
            np.savetxt(self.mdp.Adrr+'/TIME_GAP_TABLE_AK.csv',
                       timeGapTable,
                       delimiter    = ',',
                       header       = 'B,GAP (%),Iter-RT(s),Cum-RT(s)')
        

            resTable        = resTable[~ np.all(resTable==0, axis=1)]
            np.savetxt(self.mdp.Adrr+'/RESULTS_TABLE_SG.csv',
                       resTable,
                       delimiter    = ',',
                       header       = 'T,B,I-LB,LB,I-UB,UB,C-LEN,I-GAP(%),GAP(%),LB-IMP(%),UB-IMP(%),GAP-IMP(%), Unit, Pair')
            #------------------------------------------------------------------
            # Return results
            return 0, resTable
        else:
            return error, None    
    
    
    def loadStateActionsForFeasibleALP(self,trial):
        MVC_StateList  = np.load('Output/GJR/'+self.mdp.mdpName+'/SampleConstraints/TRIAL_'+str(trial)+'/MVC_StateList.npy')
        MVC_ActionList = np.load('Output/GJR/'+self.mdp.mdpName+'/SampleConstraints/TRIAL_'+str(trial)+'/MVC_ActionList.npy')
        MVC_NStateList = np.load('Output/GJR/'+self.mdp.mdpName+'/SampleConstraints/TRIAL_'+str(trial)+'/MVC_NStateList.npy')
        return list(MVC_StateList), list(MVC_ActionList),list(MVC_NStateList)
    
"""
-------------------------------------------------------------------------------

    Authors:    Parshan Pakiman  | https://parshanpakiman.github.io/homepage/
                Selva Nadarajah  | https://selvan.people.uic.edu/
                         
    Licensing Information: The MIT License
-------------------------------------------------------------------------------
"""
import numpy as np
import os,time,gc,sys,textwrap
import pandas as pd
from Bounds.lowerBoundEstimator_PIC import get_LNS_LB
from Bounds.upperBoundFromGreedyPolicy import simulateGreedyPolicy
from ALP.ALPSolver import solve_randomized_ALP,solve_Feature_Guided_ALP,resolve_randomized_ALP
from numpy import load
from sys import exc_info

"""
This class models Algorithm 1 in Pakiman et al. 2019.
"""
class SelfGuidedALP:
    
    """
    The constructor of the class that initializes an instance encapsulated in
    expInfo.
    """
    def __init__(self,iPath,expInfo):
        #----------------------------------------------------------------------
        # Load setting of an experiment using expInfo
        self.iPath = iPath
        self.mdpSetup           = expInfo['mdp'] 
        self.BF_Setup           = expInfo['bf'] 
        self.CS_Setup           = expInfo['cs'] 
        self.LB_Setup           = expInfo['lb'] 
        self.misc_Setup         = expInfo['misc'] 
        #----------------------------------------------------------------------
        # Load already sampled ALP constraints and sampled initial states
        try:
            self.constraintMatrix   = load(self.iPath+'/ALP_COMPONENTS/ALPConstMatrix.npy')
            self.objectiveVector    = load(self.iPath+'/ALP_COMPONENTS/ALPobjectiveVector.npy')
            self.RHSVector          = load(self.iPath+'/ALP_COMPONENTS/ALPRHSVector.npy')
            self.exp_VFA            = load(self.iPath+'/ALP_COMPONENTS/exp_VFA.npy')
            self.stateList          = load(self.iPath+'/SAMPLED_STATE_ACTION_BASIS/stateList.npy')
            self.initList  	        = load(self.iPath+'/SAMPLED_STATE_ACTION_BASIS/initialStateSamples.npy')
        except:
            msg = 'WARNING! Constraints of ALP are not sampled yet. Please modify the main function to first perform the constraint sampling. The algorithm is halted.'
            print('\n\n'+textwrap.TextWrapper(width=50).fill(text=msg)+'\n\n')
            sys.exit()
        
        #----------------------------------------------------------------------
        # Initialize arrays tracking the algorithm's performance
        self.ALG_OUTPUT         = np.zeros(shape=(len(self.BF_Setup['BF_List'])*self.BF_Setup['BF_Trials']+1,27))
        self.UBFluct            = np.zeros(shape=(len(self.BF_Setup['BF_List'])*4,self.BF_Setup['BF_Trials'] + 1))
        self.LBFluct            = np.zeros(shape=(len(self.BF_Setup['BF_List'])*4,self.BF_Setup['BF_Trials'] + 1))
        self.numBasisDist       = np.full(shape=(self.BF_Setup['BF_Trials'],4),
                                          fill_value=float(self.BF_Setup['BF_List'][len(self.BF_Setup['BF_List'])-1]))
        self.numJumps           = np.zeros(shape=(self.BF_Setup['BF_Trials'],4))
        self.jumpsMagnitude     = np.full(shape=(self.BF_Setup['BF_Trials'],4),fill_value=100)
        self.outPutHeader       = ''
        self.optGap             = self.misc_Setup['optGap']
    
    """
    Making required directories to save the output of self-guided ALPs.
    """        
    def makeTrialFolders(self,expTrial):
        path = self.iPath + '/TRIAL_'+str(expTrial)
        if not os.path.exists(path):
            os.makedirs(path) 
    
    """
    The following function sets the title of a table that is showed to a 
    user who uses self-guided ALPs for solving PIC instances using FALP or FGLP.
    """
    def printHeaderAndTitle(self):
        title  = '{:>3}   {:>3} | {:^47} | {:^47} | {:^47} | {:^47} |    '.format('   ','   ', 'FALP_unif','FALP_nu','FGLP_nu','FGLP_unif')
        self.outPutHeader  = '{:^3} | {:^3} | '.format('T','B')
        self.outPutHeader += '{:^7} | {:^7}({:^4}) | {:^7}({:^4}) | {:^5} | '.format('OBJ','UB','SE','LB', 'SE', 'GAP')
        self.outPutHeader += '{:^7} | {:^7}({:^4}) | {:^7}({:^4}) | {:^5} | '.format('OBJ','UB','SE','LB', 'SE', 'GAP')
        self.outPutHeader += '{:^7} | {:^7}({:^4}) | {:^7}({:^4}) | {:^5} | '.format('OBJ','UB','SE','LB', 'SE', 'GAP')
        self.outPutHeader += '{:^7} | {:^7}({:^4}) | {:^7}({:^4}) | {:^5} | '.format('OBJ','UB','SE','LB', 'SE', 'GAP')
        self.outPutHeader += '{:^4}'.format('T(m)')
        print('{:{fill}^{w}}'.format('-',fill='-',w=len(self.outPutHeader)))
        print(title)
        print(self.outPutHeader)
        print('{:{fill}^{w}}'.format('-',fill='-',w=len(self.outPutHeader)))

    """
    Impelemntation of Algorithm 1 in Pakiman et al. 2019.
    """
    def SG_ALP(self):
          #--------------------------------------------------------------------
          # Loading already sampled ALP constraints and setting up MDP and basis 
          # functions objects.          
          start_time = time.time()      
          print('Loading ALP, Desceretizing States & Actions, Initializing Cost, ...')  
          loadedThetaList       = load(self.iPath+'/SAMPLED_STATE_ACTION_BASIS/FourierTheta.npy')
          loadedInterceptList   = load(self.iPath+'/SAMPLED_STATE_ACTION_BASIS/FourierIntercept.npy')
          dmndList              = load(self.iPath+'/SAMPLED_STATE_ACTION_BASIS/demandList.npy')
          mdp                   = self.mdpSetup['type'](self.mdpSetup)
          bf                    = self.BF_Setup['type'](self.BF_Setup)

          #--------------------------------------------------------------------
          # Sampling (trajNum) number of sample paths with the length of (trajLen).
          # These trajectories will be used for estimating the policy cost.
          splPathsForGreedySimulation = [mdp.getBatchSampleFromExogenousInfo(mdp.trajLen)
                                          for _ in range(mdp.trajNum)]
          
          #--------------------------------------------------------------------
          # Sampling and fixing demand realizations.
          mdp.fixedListDemand   = dmndList[0:min(mdp.SA_numSamples,1000)] 
          mdp.SA_numSamples     = min(mdp.SA_numSamples,1000)
          
          #--------------------------------------------------------------------
          # Desceretize action set to solve greedy optimization via enumeration.
          mdp.setDesceretizedStatesAndActions()
          
          print('ALP components are loaded in {:>6.2f} (m).'.format((time.time() - start_time)/60))  
          
          #--------------------------------------------------------------------
          # Resolving an instance of PIC with multiple times (trial) with 
          # different random bases.
          trial = 0
          while trial < self.BF_Setup['BF_Trials']:
               try:
                  #-------------------------------------------------------------
                  # Creating a trial-specific folder, and initializing auxiliary
                  # variables.
                  self.makeTrialFolders(trial+1) 
                  FALPObj_INI = FALP_UB_INI = FALP_UB_SE_INI = FALP_LB_INI = FALP_LB_SE_INI  = 0
                  FALPObj_SIM = FALP_UB_SIM = FALP_UB_SE_SIM = FALP_LB_SIM = FALP_LB_SE_SIM  = 0
                  FGLPObj_SIM = FGLP_UB_SIM = FGLP_UB_SE_SIM = FGLP_LB_SIM = FGLP_LB_SE_SIM  = 0
                  FGLPObj_INI = FGLP_UB_INI = FGLP_UB_SE_INI = FGLP_LB_INI = FGLP_LB_SE_INI  = 0
                  selfGuidedLB_INI          = None      
                  selfGuidedLB_SIM          = None
                  FALP_MODEL_INI            = None
                  FALP_MODEL_SIM            = None
                  IS_FALP_INI_CONVERGED     = False
                  IS_FALP_SIM_CONVERGED     = False
                  IS_FGLP_SIM_CONVERGED     = False
                  IS_FGLP_INI_CONVERGED     = False
                  TIME                      = 0
                  FG_States                 = self.stateList
                  Opt_gap                   = self.optGap # Optimality gap tolerance.
    
                  #-------------------------------------------------------------
                  # Iteratively sampling random bases 
                  for basisItr in range(len(self.BF_Setup['BF_List'])):
                      #---------------------------------------------------------
                      # Retrieving a subset (Z) of already sampled random bases
                      # and saving the index of sampled bases for possible 
                      # future replication. 
                      start_time = time.time()
                      if basisItr == 0:
                          indexOfBases = np.random.choice(np.arange(1,self.BF_Setup['BF_num']),
                                               self.BF_Setup['BF_List'][len(self.BF_Setup['BF_List'])-1],
                                               replace=False)
                          indexOfBases[0] = 0  # Always keeping the intercept in VFA.
                          np.savetxt(self.iPath+'/TRIAL_'+str(trial+1)+'/basesIdx.csv',indexOfBases.astype(int),delimiter=",")
                      else:
                          indexOfBases = np.genfromtxt(self.iPath+'/TRIAL_'+str(trial+1)+'/basesIdx.csv',delimiter=',')
                          indexOfBases = indexOfBases.astype(int)
                      
                      #--------------------------------------------------------
                      # Fixing the number of random bases. 
                      numBasis = self.BF_Setup['BF_List'][basisItr]  
                      
                      #--------------------------------------------------------
                      # Given sampled random bases, we select a subset of ALP 
                      # columns that are already generated and saved in a .npy file.
                      ALP_cols_corr_sampled_bases = indexOfBases[0:numBasis]                        
                      bf.setSampledParms(theta=loadedThetaList[ALP_cols_corr_sampled_bases,:],
                                         intercept=loadedInterceptList[ALP_cols_corr_sampled_bases])
                      exp_VFA                     = self.exp_VFA[:,ALP_cols_corr_sampled_bases]
     
                      #--------------------------------------------------------
                      # Value of objective of ALP for nu(5,5,5)=1  
                      objVec    = sum(map(bf.evalBasisList,mdp.getSamplesFromInitalDist(False,mdp.SA_numSamples)))/mdp.SA_numSamples   
                      inistates = np.asarray(mdp.getSamplesFromInitalDist(False,mdp.SA_numSamples))

                      """ 
                          FALP Model With Uniform Initial Distribution
                      """
                      #--------------------------------------------------------
                      # Solving FALP with uniform initial distribution to obtain:
                      # 1) FALP_MODEL_INI: a Gurobi model that has all FALP 
                      #    variables and constraints.
                      # 2) FALPObj_INI: objective value of FALP.
                      # 3) ALPSln_INI: optimal solution to FALP.
                      obj_INI=sum(map(bf.evalBasisList,FG_States))/len(FG_States)
                      FALP_MODEL_INI, FALPObj_INI, ALPSln_INI = solve_randomized_ALP(self.mdpSetup['CPU_CORE'],
                                                                                      self.iPath+'/TRIAL_'+str(trial+1),
                                                                                      numBasis,
                                                                                      obj_INI, 
                                                                                      self.constraintMatrix[:,ALP_cols_corr_sampled_bases],
                                                                                      self.RHSVector)
                      bf.setOptimalCoef(ALPSln_INI)
                      
                      #--------------------------------------------------------
                      # If stopping criteria not triggered yet: 
                      # 1) simulate greedy policy to obtain upper bound
                      # 2) estimate a "true" lower bound via simulation
                      if IS_FALP_INI_CONVERGED == False:                              
                          FALP_UB_INI, FALP_UB_SE_INI = simulateGreedyPolicy(mdp,bf,splPathsForGreedySimulation)
                          LB, SE                      = get_LNS_LB(mdp,bf,inistates,self.LB_Setup, FALPObj_INI)
                          if LB >= FALP_LB_INI:
                              FALP_LB_INI    = LB
                              FALP_LB_SE_INI = SE
                      
                      #--------------------------------------------------------
                      # Compute optimality gap and check convergence
                      if not FALP_UB_INI == 0:
                          if (1 - abs(FALP_LB_INI/FALP_UB_INI)) < Opt_gap or FALP_UB_INI < FALP_LB_INI:
                              if not IS_FALP_INI_CONVERGED:
                                  self.numBasisDist[trial, 0] = numBasis    
                              IS_FALP_INI_CONVERGED = True
                      
                      #--------------------------------------------------------
                      # Adjusts the message showed to a user.
                      if basisItr ==0 and trial ==0:
                          self.printHeaderAndTitle()      
                      
                      #--------------------------------------------------------
                      # Save a copy of Gurobi model FALP_MODEL_INI for future
                      # use in FGLP (this avoids creating the Gurobi model from
                      # scratch.)
                      FALP_MODEL_INI.update()
                      FALP_MODEL_INI_COPY =  FALP_MODEL_INI.copy()
                        
                      """
                          FGLP Model With nu(5,5,5)=1 Initial Distribution
                      """
                      #--------------------------------------------------------
                      # FALP = FGLP at the very first iteration.
                      if basisItr == 0:
                          FGLP_UB_INI       = FALP_UB_INI
                          FGLP_UB_SE_INI    = FALP_UB_SE_INI
                          FGLP_LB_INI       = FALP_LB_INI
                          FGLP_LB_SE_INI    = FALP_LB_SE_INI
                          FGLPObj_INI       = FALPObj_INI
                    
                      #--------------------------------------------------------
                      # If FGLP is not converged and we are not in the very 
                      # first iteration, then
                      if basisItr > 0 and IS_FGLP_INI_CONVERGED == False:
                            #----------------------------------------------------
                            # Consider constraints:
                            #     V_cur <= c(s,a) + gamma*E[V_cur]  ---> VFA_UB
                            #     V_cur >= V_prev                   ---> FGLP_LB_INI
                            # Then, 
                            #     1) FGLP_LB_INI: previous VFA used in self-guiding 
                            #        constraints
                            #     2) VFA_UB: upper bound on the VFA w.r.t. the current VFA
                            #     
                            curVFAEval_INI  = np.asarray([bf.evalBasisList(state) for state in FG_States])  
                            VFA_UB          = [self.RHSVector[c] + np.inner(exp_VFA[c][:],ALPSln_INI) 
                                                for c in range(len(self.constraintMatrix))]
                            
                            #----------------------------------------------------
                            # Solve FGLP model to obtain the Gurobi model 
                            # FGLPObj_INI with optimal solutrion FGLPSln_INI.
                            FGLPObj_INI , FGLPSln_INI = solve_Feature_Guided_ALP( FALP_MODEL_INI_COPY,
                                                                               len(self.constraintMatrix),
                                                                               curVFAEval_INI,
                                                                               selfGuidedLB_INI,
                                                                               VFA_UB,
                                                                               None)   
                            bf.setOptimalCoef(FGLPSln_INI)  
                            
                            #--------------------------------------------------
                            # 1) simulate greedy policy to obtain upper bound
                            # 2) estimate a "true" lower bound via simulation
                            FGLP_UB_INI , FGLP_UB_SE_INI = simulateGreedyPolicy(mdp, bf, splPathsForGreedySimulation)
                            LB, SE                       = get_LNS_LB( mdp, bf, inistates, self.LB_Setup, FGLPObj_INI)
                            if LB >= FGLP_LB_INI:
                                FGLP_LB_INI              = LB
                                FGLP_LB_SE_INI           = SE
                      
                      #--------------------------------------------------------
                      # Compute optimality gap and check convergence
                      if not FGLP_UB_INI == 0:
                          if abs(1 - abs(FGLP_LB_INI/FGLP_UB_INI)) < Opt_gap or FGLP_UB_INI < FGLP_LB_INI:
                              if not IS_FGLP_INI_CONVERGED:
                                 self.numBasisDist[trial,3] = numBasis
                              IS_FGLP_INI_CONVERGED = True  
                      
                      #--------------------------------------------------------
                      # Compute the value of "prev" VFA on the sampled states 
                      # such that is can be used with the self-guiding constraint
                      # in the next iteration.
                      if not IS_FGLP_INI_CONVERGED:  
                          selfGuidedLB_INI = [bf.getVFA(state) for state in FG_States]   
                      
                      """
                          FALP Model With nu(5,5,5)=1 Initial Distribution 
                      """
                      #--------------------------------------------------------
                      # Reoptimize the already computed FALP model with initial
                      # distribution nu(5,5,5)=1 to obtain:
                      #     1) Gurobi model: FALP_MODEL_SIM
                      #     2) Optimal value: FALPObj_SIM
                      #     3) Optimal solution: ALPSln_SIM
                      FALP_MODEL_SIM, FALPObj_SIM, ALPSln_SIM = resolve_randomized_ALP(FALP_MODEL_INI,objVec)
                      bf.setOptimalCoef(ALPSln_SIM)
                      
                      #--------------------------------------------------------
                      # Compute optimality gap if FALP is not converged yet.
                      if IS_FALP_SIM_CONVERGED == False:    
                          FALP_UB_SIM , FALP_UB_SE_SIM  = simulateGreedyPolicy(mdp,bf,splPathsForGreedySimulation)
                          LB, SE                        = get_LNS_LB(mdp,bf,inistates,self.LB_Setup, FALPObj_SIM)
                          if LB >= FALP_LB_SIM:
                              FALP_LB_SIM               = LB
                              FALP_LB_SE_SIM            = SE
                              
                      #--------------------------------------------------------
                      # Check convergence.           
                      if not FALP_UB_SIM == 0:  
                          if abs(1 - abs(FALP_LB_SIM/FALP_UB_SIM)) < Opt_gap or FALP_UB_SIM < FALP_LB_SIM:
                              if not IS_FALP_SIM_CONVERGED:
                                  self.numBasisDist[trial,1] = numBasis
                                  
                              IS_FALP_SIM_CONVERGED = True   
                              
                      """
                          FGLP Model With Uniform Initial Distribution 
                      """  
                      #--------------------------------------------------------
                      # FALP = FGLP at the very first iteration.
                      if basisItr == 0:
                          FGLP_UB_SIM       = FALP_UB_SIM
                          FGLP_UB_SE_SIM    = FALP_UB_SE_SIM
                          FGLP_LB_SIM       = FALP_LB_SIM
                          FGLP_LB_SE_SIM    = FALP_LB_SE_SIM
                          FGLPObj_SIM       = FALPObj_SIM
                      
                      #--------------------------------------------------------
                      # If FGLP is not converged and we are not in the very 
                      # first iteration, then
                      if basisItr > 0 and IS_FGLP_SIM_CONVERGED == False:
                            #----------------------------------------------------
                            # Consider constraints:
                            #     V_cur <= c(s,a) + gamma*E[V_cur]  ---> VFA_UB
                            #     V_cur >= V_prev                   ---> FGLP_LB_INI
                            # Then, 
                            #     1) FGLP_LB_INI: previous VFA used in self-guiding 
                            #        constraints
                            #     2) VFA_UB: upper bound on the VFA w.r.t. the current VFA
                            #     
                            curVFAEval_SIM = np.asarray([bf.evalBasisList(state) for state in FG_States])
                            VFA_UB          = [self.RHSVector[c] + np.inner(exp_VFA[c][:],ALPSln_SIM)
                                                for c in range(len(self.constraintMatrix))]
                            
                            #--------------------------------------------------------
                            # Solve FGLP to obtain
                            #     1) Optimal value: FGLPObj_SIM
                            #     2) Optimal solution: SG_ALPSln
                            FGLPObj_SIM , SG_ALPSln = solve_Feature_Guided_ALP( FALP_MODEL_SIM,
                                                                               len(self.constraintMatrix),
                                                                               curVFAEval_SIM,
                                                                               selfGuidedLB_SIM,
                                                                               VFA_UB,
                                                                               objVec)                                                  
                            bf.setOptimalCoef(SG_ALPSln)  
                            
                            #--------------------------------------------------------
                            # Compute upper and lower bounds.
                            FGLP_UB_SIM , FGLP_UB_SE_SIM = simulateGreedyPolicy(mdp, bf, splPathsForGreedySimulation)
                            LB, SE = get_LNS_LB( mdp, bf, inistates, self.LB_Setup, FGLPObj_SIM)
                            if LB >= FGLP_LB_SIM:
                                FGLP_LB_SIM  = LB
                                FGLP_LB_SE_SIM = SE

                      #--------------------------------------------------------
                      # Check convergence.                             
                      if not FGLP_UB_SIM == 0:
                          if abs(1 - abs(FGLP_LB_SIM/FGLP_UB_SIM)) < Opt_gap or FGLP_UB_SIM<FGLP_LB_SIM:
                              if not IS_FGLP_SIM_CONVERGED:
                                  self.numBasisDist[trial,2] = numBasis
                              IS_FGLP_SIM_CONVERGED = True  
                              
                      #--------------------------------------------------------
                      # Compute the value of "prev" VFA on the sampled states 
                      # such that is can be used with the self-guiding constraint
                      # in the next iteration
                      if not IS_FGLP_SIM_CONVERGED:
                          selfGuidedLB_SIM = [bf.getVFA(state) for state in FG_States]                         
                      

                      """
                          Get Ready For Next Iteration & Save Per Iteration Log
                      """
                      self.setJumpsBehavior(trial)
                      TIME = float((time.time() - start_time)/60)
                      
                      
                      self.setOutputLog(trial, basisItr,
                                        FALPObj_INI,FALP_UB_INI,FALP_UB_SE_INI,FALP_LB_INI,FALP_LB_SE_INI,
                                        FALPObj_SIM,FALP_UB_SIM,FALP_UB_SE_SIM,FALP_LB_SIM,FALP_LB_SE_SIM,
                                        FGLPObj_SIM,FGLP_UB_SIM,FGLP_UB_SE_SIM,FGLP_LB_SIM,
                                        FGLP_LB_SE_SIM,
                                        FGLPObj_INI,FGLP_UB_INI,FGLP_UB_SE_INI,FGLP_LB_INI,
                                        FGLP_LB_SE_INI,TIME)
                      
                      
                      self.printIterationResult(trial,basisItr,row=trial*len(self.BF_Setup['BF_List']) + basisItr)
                      
                      #--------------------------------------------------------
                      # If all models are converged, then stop!
                      if IS_FALP_INI_CONVERGED is True    and \
                             IS_FALP_SIM_CONVERGED is True    and \
                                 IS_FGLP_SIM_CONVERGED is True and \
                                     IS_FGLP_INI_CONVERGED is True:
                                             break
                  #------------------------------------------------------------
                  # Increment trial, and collect garbaes.
                  trial +=1
                  gc.collect()
              
              #----------------------------------------------------------------
              # Catch exceptions here (a naive implementation)
               except:
                   type,value,traceback = exc_info()
                   print("\n*** AN EXCEPTION IS HAPPENED ***")
                   print(type,value)
                   print("\n")
                   sys.exit()
          
          #--------------------------------------------------------------------
          # Make sure that policy cost fluctuation is computed and write the
          # log into a file.                          
          self.setFluctuation()
          self.writeOutput()       
          return True
    
  
    """
    This function prints lower bound, upper bound, optimality gap, and other 
    statistics obtained while solving a PIC instance. Particularly, it shows 
    this information for an iteration of Algorithm 1 in Pakiman et al. 2019.
    """
    def printIterationResult(self,trial,basisItr,row):
        # When iteration =0, print output header
        if basisItr==0 and not trial==0:
            print('{:{fill}^{w}}'.format('-',fill='-',w=len(self.outPutHeader)))
        
        # Construct the string that should be shown to a user
        outPutString  = '{:>3d} | {:>3d} | '.format(int(self.ALG_OUTPUT[row,0]),int(self.ALG_OUTPUT[row,1]))
        outPutString += '{:>7.1f} | {:>7.1f}({:>1.2f}) | {:>7.1f}({:>1.2f}) | {:>5.1f} | '.format(
                         self.ALG_OUTPUT[row,2], self.ALG_OUTPUT[row,3], self.ALG_OUTPUT[row,4],
                         self.ALG_OUTPUT[row,5], self.ALG_OUTPUT[row,6], self.ALG_OUTPUT[row,7])
        
        outPutString += '{:>7.1f} | {:>7.1f}({:>1.2f}) | {:>7.1f}({:>1.2f}) | {:>5.1f} | '.format(
                         self.ALG_OUTPUT[row,8], self.ALG_OUTPUT[row,9], self.ALG_OUTPUT[row,10],
                         self.ALG_OUTPUT[row,11], self.ALG_OUTPUT[row,12], self.ALG_OUTPUT[row,13])
        
        outPutString += '{:>7.1f} | {:>7.1f}({:>1.2f}) | {:>7.1f}({:>1.2f}) | {:>5.1f} | '.format(
                         self.ALG_OUTPUT[row,14], self.ALG_OUTPUT[row,15], self.ALG_OUTPUT[row,16],
                         self.ALG_OUTPUT[row,17], self.ALG_OUTPUT[row,18], self.ALG_OUTPUT[row,19])
        
        outPutString += '{:>7.1f} | {:>7.1f}({:>1.2f}) | {:>7.1f}({:>1.2f}) | {:>5.1f} | '.format(
                         self.ALG_OUTPUT[row,20], self.ALG_OUTPUT[row,21], self.ALG_OUTPUT[row,22],
                         self.ALG_OUTPUT[row,23], self.ALG_OUTPUT[row,24], self.ALG_OUTPUT[row,25])
        
        outPutString += '{:>4.1f}'.format(self.ALG_OUTPUT[row,26])        
        print(outPutString)

    """
    The following function saves stat of self-guided ALPs including obtained 
    gap, runtime, policy fluctuation percentage, and etc.
    """     
    def writeOutput(self):
        dataFrameUB=pd.DataFrame(self.UBFluct, columns = [ str('Trial_'+str(_+1)) for _ in range(self.BF_Setup['BF_Trials'])] + ['FLUCT'])
        
        rowName=['FALP_Nu(# basis = '+str(_)+')' for _ in self.BF_Setup['BF_List']]+ \
                    ['FALP_Unif(# basis = '+str(_)+')' for _ in self.BF_Setup['BF_List']] +\
                        ['FGLP_Nu(# basis = '+str(_)+')' for _ in self.BF_Setup['BF_List']] +\
                            ['FGLP_Unif(# basis = '+str(_)+')' for _ in self.BF_Setup['BF_List']]
        rowName = {_:rowName[_] for _ in range(len(rowName))}
        dataFrameUB.rename(index=rowName,inplace=True)   
        dataFrameUB.to_csv (self.iPath + "/UB_Fluctuation_"+str(self.mdpSetup['mdp_name'])+'.csv')
        
        dataFrameLB = pd.DataFrame(self.LBFluct, columns = [ str('Trial_'+str(_+1)) for _ in range(self.BF_Setup['BF_Trials'])] + ['FLUCT'])
        dataFrameLB.rename(index=rowName, inplace=True)
        dataFrameLB.to_csv (self.iPath + "/LB_Fluctuation_"+str(self.mdpSetup['mdp_name'])+'.csv')
        
        header  = ['Trial','Basis']
        header += ['FALP_Nu[OBJ]','FALP_Nu[UB]','FALP_Nu[SE of UB]','FALP_Nu[LB]', 'FALP_Nu[SE of LB]', 'FALP_Nu[GAP]']   # Header to FALP with nu init distribution
        header += ['FALP_Unif[OBJ]','FALP_Unif[UB]','FALP_Unif[SE of UB]','FALP_Unif[LB]', 'FALP_Unif[SE of LB]', 'FALP_Unif[GAP]']   # Header to FALP with uniform init distribution
        header += ['FGLP_Nu[OBJ]','FGLP_Nu[UB]','FGLP_Nu[SE of UB]','FGLP_Nu[LB]', 'FGLP_Nu[SE of LB]', 'FGLP_Nu[GAP]']   # Header to FGLP with nu init distribution
        header += ['FGLP_Unif[OBJ]','FGLP_Unif[UB]','FGLP_Unif[SE of UB]','FGLP_Unif[LB]', 'FGLP_Unif[SE of LB]', 'FGLP_Unif[GAP]']   # Header to FGLP with uniform init distribution
        header += ['Time (m)']
        dataFrameOut = pd.DataFrame(self.ALG_OUTPUT,columns=header)  
        dataFrameOut.to_csv (self.iPath + "/performanceTable_"+str(self.mdpSetup['mdp_name'])+'.csv')  
        
        dataFrameNumBasis = pd.DataFrame( self.numBasisDist, columns = ['FALP_unif', 'FALP_nu', 'FGLP_nu','FGLP_unif'])
        dataFrameNumBasis.to_csv(self.iPath + "/distributionOfNumberOfBases_"+str(self.mdpSetup['mdp_name'])+'.csv')  
        
        dataFrameNumJumps = pd.DataFrame( self.numJumps, columns = ['FALP_unif', 'FALP_nu', 'FGLP_nu','FGLP_unif'])
        dataFrameNumJumps.to_csv(self.iPath + "/numberOfJumps_"+str(self.mdpSetup['mdp_name'])+'.csv')  
        
        dataFrameJumpsMagnitude = pd.DataFrame( self.jumpsMagnitude, columns = ['FALP_unif', 'FALP_nu', 'FGLP_nu','FGLP_unif'])
        dataFrameJumpsMagnitude.to_csv(self.iPath + "/jumpsMagnitude_"+str(self.mdpSetup['mdp_name'])+'.csv')  

    """
    The following function computes percentage and magnitude of fluctuation for 
    4 models FAL_nu, FALP_unif, FGLP_nu, FGLP_unif.
    """                      
    def setJumpsBehavior(self,trial):
        FALP_JUMPS_INI = 0      # Jumps of FALP with uniform distribution
        FALP_JUMPS_SIM = 0      # Jumps of FALP with (5,5,5) distribution
        FGLP_JUMPS_SIM = 0      # Jumps of FGLP with (5,5,5) distribution
        FGLP_JUMPS_INI = 0      # Jumps of FGLP with uniform distribution
        FALP_MAG_JUMPS_INI = 0  # Magnitude of Jumps of FALP with uniform distribution
        FALP_MAG_JUMPS_SIM = 0  # Magnitude of Jumps of FALP with (5,5,5) distribution
        FGLP_MAG_JUMPS_SIM = 0  # Magnitude of Jumps of FGLP with (5,5,5) distribution
        FGLP_MAG_JUMPS_INI = 0  # Magnitude of Jumps of FGLP with uniform distribution
        
        for i in range(4, len(self.BF_Setup['BF_List'])):
            
            row=trial*len(self.BF_Setup['BF_List']) + i
            ins = trial*len(self.BF_Setup['BF_List'])
            if self.ALG_OUTPUT[row,3] < self.ALG_OUTPUT[row+1,3]:
                FALP_JUMPS_INI +=1
                FALP_MAG_JUMPS_INI += (self.ALG_OUTPUT[row+1,3]/np.min(self.ALG_OUTPUT[range(ins,row+1),3]))*100
                
            if self.ALG_OUTPUT[row,9] < self.ALG_OUTPUT[row+1,9]:
                FALP_JUMPS_SIM +=1
                FALP_MAG_JUMPS_SIM += (self.ALG_OUTPUT[row+1,9]/np.min(self.ALG_OUTPUT[range(ins,row+1),9]))*100
                
            if self.ALG_OUTPUT[row,15] < self.ALG_OUTPUT[row+1,15]:
                FGLP_JUMPS_SIM +=1
                FGLP_MAG_JUMPS_SIM += (self.ALG_OUTPUT[row+1,15]/np.min(self.ALG_OUTPUT[range(ins,row+1),15]))*100

            if self.ALG_OUTPUT[row,21] < self.ALG_OUTPUT[row+1,21]:
                FGLP_JUMPS_INI +=1
                FGLP_MAG_JUMPS_INI += (self.ALG_OUTPUT[row+1,21]/np.min(self.ALG_OUTPUT[range(ins,row+1),21]))*100

        self.numJumps[trial,0] = FALP_JUMPS_INI
        self.numJumps[trial,1] = FALP_JUMPS_SIM
        self.numJumps[trial,2] = FGLP_JUMPS_SIM
        self.numJumps[trial,3] = FGLP_JUMPS_INI
        
        if not FALP_JUMPS_INI == 0:
            self.jumpsMagnitude[trial,0] = FALP_MAG_JUMPS_INI / FALP_JUMPS_INI
            
        if not FALP_JUMPS_SIM == 0:    
            self.jumpsMagnitude[trial,1] = FALP_MAG_JUMPS_SIM / FALP_JUMPS_SIM
            
        if not FGLP_JUMPS_SIM == 0:     
            self.jumpsMagnitude[trial,2] = FGLP_MAG_JUMPS_SIM / FGLP_JUMPS_SIM
  
        if not FGLP_JUMPS_INI == 0:     
            self.jumpsMagnitude[trial,3] = FGLP_MAG_JUMPS_INI / FGLP_JUMPS_INI
            
    """
    We compute the fluctuation percentage and magnitude of fluctuation in the
    following function.
    """             
    def setFluctuation(self):
        BF_Len=len(self.BF_Setup['BF_List'])
        for i in range(BF_Len):
              for j in range(self.BF_Setup['BF_Trials']):
                      self.UBFluct[i,j] = self.ALG_OUTPUT[j*BF_Len+i,3] 
                      self.LBFluct[i,j] = self.ALG_OUTPUT[j*BF_Len+i,5]
          
        for i in range(BF_Len):
              for j in range(self.BF_Setup['BF_Trials']):
                      self.UBFluct[i+ 1*BF_Len,j] = self.ALG_OUTPUT[j*BF_Len+i,9]  
                      self.LBFluct[i+ 1*BF_Len,j] = self.ALG_OUTPUT[j*BF_Len+i,11]
                      
        for i in range(BF_Len):
              for j in range(self.BF_Setup['BF_Trials']):
                      self.UBFluct[i+ 2*BF_Len,j] = self.ALG_OUTPUT[j*BF_Len+i,15]
                      self.LBFluct[i+ 2*BF_Len,j] = self.ALG_OUTPUT[j*BF_Len+i,17]

        for i in range(BF_Len):
              for j in range(self.BF_Setup['BF_Trials']):
                      self.UBFluct[i+ 3*BF_Len,j] = self.ALG_OUTPUT[j*BF_Len+i,21]
                      self.LBFluct[i+ 3*BF_Len,j] = self.ALG_OUTPUT[j*BF_Len+i,23]
                      
        for i in range(len(self.BF_Setup['BF_List'])*4):
              j = self.BF_Setup['BF_Trials']
              UB_mx=np.max(self.UBFluct[i,range(j)])
              UB_mn=np.min(self.UBFluct[i,range(j)])
              LB_mx=np.max(self.LBFluct[i,range(j)])
              LB_mn=np.min(self.LBFluct[i,range(j)])
              if LB_mx == 0:
                  self.LBFluct[i,j] = 0
                  self.UBFluct[i,j] = 0
              else:
                  self.LBFluct[i,j] = (1 - LB_mn/LB_mx)*100
                  self.UBFluct[i,j] = (1 - UB_mn/UB_mx)*100
                 
    """
    The following function collects per iteration statistics of 4 models FAL_nu,
    FALP_unif, FGLP_nu, and FGLP_unif.
    """                
    def setOutputLog(self,
                     trial,
                     basisItr,
                     FALPObj_INI,
                     FALP_UB_INI,
                     FALP_UB_SE_INI,
                     FALP_LB_INI,
                     FALP_LB_SE_INI,
                     FALPObj_SIM,
                     FALP_UB_SIM,
                     FALP_UB_SE_SIM,
                     FALP_LB_SIM,
                     FALP_LB_SE_SIM,
                     FGLPObj_SIM,
                     FGLP_UB_SIM,
                     FGLP_UB_SE_SIM,
                     FGLP_LB_SIM,
                     FGLP_LB_SE_SIM,
                     FGLPObj_INI,
                     FGLP_UB_INI,
                     FGLP_UB_SE_INI,
                     FGLP_LB_INI,
                     FGLP_LB_SE_INI,
                     TIME):
      
      FALP_GAP_INI  = 0.0
      FALP_GAP_SIM  = 0.0
      FGLP_GAP_SIM  = 0.0
      FGLP_GAP_INI  = 0.0
      
      if not FALP_UB_INI == 0:
          FALP_GAP_INI = 100*(1-FALP_LB_INI/FALP_UB_INI)
      
      if not FALP_UB_SIM == 0:
          FALP_GAP_SIM = 100*(1-FALP_LB_SIM/FALP_UB_SIM)
            
      if not FGLP_UB_SIM == 0:
              FGLP_GAP_SIM =100*(1-FGLP_LB_SIM/FGLP_UB_SIM)

      if not FGLP_UB_INI == 0:
              FGLP_GAP_INI =100*(1-FGLP_LB_INI/FGLP_UB_INI)
      
      row=trial*len(self.BF_Setup['BF_List']) + basisItr
      self.ALG_OUTPUT[row,0]    = int(trial+1)
      self.ALG_OUTPUT[row,1]    = int(self.BF_Setup['BF_List'][basisItr])
      self.ALG_OUTPUT[row,2]    = FALPObj_INI
      self.ALG_OUTPUT[row,3]    = FALP_UB_INI
      self.ALG_OUTPUT[row,4]    = FALP_UB_SE_INI
      self.ALG_OUTPUT[row,5]    = FALP_LB_INI
      self.ALG_OUTPUT[row,6]    = FALP_LB_SE_INI
      self.ALG_OUTPUT[row,7]    = FALP_GAP_INI
      self.ALG_OUTPUT[row,8]    = FALPObj_SIM
      self.ALG_OUTPUT[row,9]    = FALP_UB_SIM
      self.ALG_OUTPUT[row,10]   = FALP_UB_SE_SIM
      self.ALG_OUTPUT[row,11]   = FALP_LB_SIM
      self.ALG_OUTPUT[row,12]   = FALP_LB_SE_SIM
      self.ALG_OUTPUT[row,13]   = FALP_GAP_SIM
      self.ALG_OUTPUT[row,14]   = FGLPObj_SIM
      self.ALG_OUTPUT[row,15]   = FGLP_UB_SIM
      self.ALG_OUTPUT[row,16]   = FGLP_UB_SE_SIM
      self.ALG_OUTPUT[row,17]   = FGLP_LB_SIM
      self.ALG_OUTPUT[row,18]   = FGLP_LB_SE_SIM
      self.ALG_OUTPUT[row,19]   = FGLP_GAP_SIM
      self.ALG_OUTPUT[row,20]   = FGLPObj_INI
      self.ALG_OUTPUT[row,21]   = FGLP_UB_INI
      self.ALG_OUTPUT[row,22]   = FGLP_UB_SE_INI
      self.ALG_OUTPUT[row,23]   = FGLP_LB_INI
      self.ALG_OUTPUT[row,24]   = FGLP_LB_SE_INI
      self.ALG_OUTPUT[row,25]   = FGLP_GAP_INI
      self.ALG_OUTPUT[row,26]   = TIME
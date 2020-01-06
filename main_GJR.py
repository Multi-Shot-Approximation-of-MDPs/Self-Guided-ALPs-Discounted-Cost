"""
-------------------------------------------------------------------------------

    Authors:    Parshan Pakiman  | https://parshanpakiman.github.io/homepage/
                Selva Nadarajah  | https://selvan.people.uic.edu/
                         
    Licensing Information: The MIT License
-------------------------------------------------------------------------------
"""
import importlib,sys,os,time,textwrap
from Algorithms.selfGuidedALP_GJR import SelfGuidedALP
from BasisFunction.stumpBasisFunctions import StumpBasis
from utils import getExperimentSetupSummary,selfGuidedHeader
from ALP.constraintSampler import sampleConstraints
from Algorithms.adaptiveBasisGenerationBenchmark_AK12 import GJR_Benchmark


"""
The following main function solves an instance of GJR by running different 
components of self-guided approximate linear programs.
"""
if __name__== "__main__":
    
    #--------------------------------------------------------------------------
    # Receive the trial of GJR and load the MDP and the basis function setup
    trial   = int(sys.argv[3])
    expInfo = importlib.import_module(sys.argv[1]).experimentSetup(trial)
    mdp     = expInfo['mdp']
    bf      = expInfo['bf']
    misc    = expInfo['misc']
    
    #--------------------------------------------------------------------------
    # Print the algorithm header for a user
    selfGuidedHeader(mdp['mdp_name'],trial,"Generalized Joint Replenishment",124)    
    
    #--------------------------------------------------------------------------
    # Construct random stump basis functions
    BF              = StumpBasis(bf)
    BF.setRandBasisCoefList()
    indexList       = BF.index_list         #random index of random stumps
    thresholdList   = BF.threshold_list     #random threshold of random stumps
   
    #--------------------------------------------------------------------------
    # Creating needed folders for outputs
    if not os.path.exists('Output/GJR/'+mdp['mdp_name']+'/'):
            os.makedirs('Output/GJR/'+mdp['mdp_name']+'/')
    
    if not os.path.exists('Output/GJR/'+mdp['mdp_name']+'/FG'):
            os.makedirs('Output/GJR/'+mdp['mdp_name']+'/FG')            
           
    if not os.path.exists('Output/GJR/'+mdp['mdp_name']+'/AK'):
            os.makedirs('Output/GJR/'+mdp['mdp_name']+'/AK')              

    if not os.path.exists('Output/GJR/'+mdp['mdp_name']+'/SampleConstraints/TRIAL_'+str(trial)):
            os.makedirs('Output/GJR/'+mdp['mdp_name']+'/SampleConstraints/TRIAL_'+str(trial))  
   
    with open('Output/GJR/'+mdp['mdp_name'] + '/'+mdp['mdp_name']+'.txt', 'w') as file:
            file.write(getExperimentSetupSummary(expInfo))    
   
    
    #--------------------------------------------------------------------------
    # Updated the path associated with a GJR instance
    mdp.update({'Adrr':'Output/GJR/'+mdp['mdp_name']+'/FG/TRIAL_'+str(trial)})
    mdp.update({'trial':trial})
    if not os.path.exists(mdp['Adrr']+'/'):
        os.makedirs(mdp['Adrr'])
    
    #--------------------------------------------------------------------------
    # Generating the initial state-action pairs needed for performing row generation
    print("Generating the initial state-action pairs needed for performing row generation ...")    
    intialContrSamples              = misc['intialContrSamples']
    start_time                      = time.time()
    MVC_StateList,MVC_ActionList,MVC_NStateList = \
            sampleConstraints(mdp,intialContrSamples,trial,mdp['Threads'])
    
    
    ALG = SelfGuidedALP(expInfo)

    #--------------------------------------------------------------------------
    # Do we get an unbounded ALP to start row generation?    
    flag = ALG.isSampledALPFeasible(MVC_StateList,MVC_ActionList,MVC_NStateList)
    
    # If the ALP is bounded with the sampled state-action pairs, then
    if flag:
        print("ALP components are generated in {:>5.4f} (m).".format((time.time() - start_time)/60)) # To minutes
    else:
        msg = 'WARNING! The number of initial state-action pairs used for solving ALP before performing constraint (row) generation  is low. ALP is unbounded. Please increase the number of samples, e.g., change the parameter intialContrSamples.'
        print('\n\n'+textwrap.TextWrapper(width=50).fill(text=msg)+'\n\n')
        sys.exit()

   
    """
        =======================================================================
                                    Self-guided ALP
        =======================================================================
    """    
    mdp.update({'Adrr':'Output/GJR/'+mdp['mdp_name']+'/FG/TRIAL_'+str(trial)})
    flag, resTable = ALG.SG_ALP(expInfo,BF,bf,indexList,thresholdList,trial,mdp['Threads'])


   
    """
        =======================================================================
                                    Set Up AK 2012
        =======================================================================
    """    
    expInfo     = importlib.import_module(sys.argv[1]).experimentSetup(trial)
    mdp         = expInfo['mdp']
    bf          = expInfo['bf']
    expInfo['mdp'].update({'Adrr':'Output/GJR/'+expInfo['mdp']['mdp_name']+'/AK/TRIAL_'+str(trial)})
    
    if not os.path.exists(mdp['Adrr']+'/'):
        os.makedirs(mdp['Adrr'])
    
    
    GJR_Benchmark(expInfo).AK_GJR_Algorithm(trial)     
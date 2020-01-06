"""
-------------------------------------------------------------------------------

    Authors:    Parshan Pakiman  | https://parshanpakiman.github.io/homepage/
                Selva Nadarajah  | https://selvan.people.uic.edu/
                         
    Licensing Information: The MIT License
-------------------------------------------------------------------------------
"""
import time,sys,importlib
import shutil
from Algorithms.selfGuidedALP_PIC import SelfGuidedALP
from ALP.constraintSampler import getStateActionSamples
from utils import getExperimentOutputPath,getExperimentSetupSummary, selfGuidedHeader
from ALP.ALPMaker import makeALP

"""
The following main function solves an instance of PIC by running different 
components of self-guided approximate linear programs.
"""
if __name__== "__main__":
    #--------------------------------------------------------------------------
    # Given 'argv' to the main function, which indicates which instance of PIC 
    # should be solved, the following code block sets up MDP and basis functions
    # objects.
    expInfo     = importlib.import_module(sys.argv[1]).experimentSetup()
    outputPath  = getExperimentOutputPath(expInfo)
    mdp         = expInfo['mdp']['type']( expInfo['mdp'])
    bf          = expInfo['bf']['type']( expInfo['bf'])
    

    #--------------------------------------------------------------------------
    # Storing a copy of instance parameters in a subfolder of the output folder.
    iPath = './Output/PIC/'+ str(expInfo['mdp']['mdp_name']).upper()
    shutil.copy2('./MDP/PerishableInventory/Instances/'+str(expInfo['mdp']['mdp_name']).upper()+'.py',iPath)
    with open(iPath+'/'+str(expInfo['mdp']['mdp_name']).upper()+'.txt', 'w') as file:
         file.write(getExperimentSetupSummary(expInfo))
    
    #--------------------------------------------------------------------------
    # Print the algorithm header for a user
    selfGuidedHeader(mdp.mdpName,False,"Perishable Inventory Control", 216)     
    #--------------------------------------------------------------------------
    # The following code runs the constraint sampling algorithm and creates a 
    # huge matrix of ALP constraints. 
    """
        *** If you already saved your ALP constraints matrix as a '.npy' array,
            then you can comment out the following part. 
    """
    print('Performing constraint sampling ...')
    start_time = time.time()    
    getStateActionSamples(mdp,expInfo['cs'],iPath)
    print("ALP constraints are sampled in {:>5.4f} (s).".format(time.time() - start_time))    
    start_time = time.time()
    makeALP(mdp,bf,iPath)
    print("ALP components are generated in {:>5.4f} (m).".format((time.time() - start_time)/60)) # To minutes

    #--------------------------------------------------------------------------
    # The following code creates object ALG that essentially allows us to call
    # the self-guided ALPs algorithm.
    start_time  = time.time()          
    ALG         = SelfGuidedALP(outputPath,expInfo)
    ALG.SG_ALP()    
    print('{:{fill}^{w}}'.format('-',fill='-',w=216))
    print("The whole process has been done in {:>5.4f} (h).".format((time.time() - start_time)/3600)) # To hours
    
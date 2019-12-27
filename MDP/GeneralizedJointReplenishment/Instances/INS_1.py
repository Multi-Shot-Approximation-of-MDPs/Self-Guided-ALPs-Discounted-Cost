"""
###############################################################################
# Created: Parshan Pakiman  | https://parshanpakiman.github.io/homepage/
#                           | http://business.uic.edu/faculty/parshan-pakiman
#                          
# Licensing Information: The MIT License
###############################################################################
"""
import numpy as np
import os
from math import floor 
from MDP.GeneralizedJointReplenishment.generalizedJointReplenishment import GeneralizedJointReplenishment

"""
Set up an experiment by configuring MDP, basis function, and sampling properties.
"""
def experimentSetup(trial):

    #--------------------------------------------------------------------------
    # Get current path
    dir_path            = os.path.dirname(os.path.realpath(__file__))

    #--------------------------------------------------------------------------
    # Load parameters of a GJR instance
    minorCost     =   np.genfromtxt(dir_path+'/sampledInstances/minorCost.csv',delimiter=',')        
    consRate      =   np.genfromtxt(dir_path+'/sampledInstances/consumptionRate.csv',delimiter=',')   
    #upBound      =   np.genfromtxt(dir_path+'/sampledInstances/upBound.csv',delimiter=',')    #For future use
    rndCapacity   =   np.genfromtxt(dir_path+'/sampledInstances/rndCapacity.csv',delimiter=',')   
    discProb      =   np.genfromtxt(dir_path+'/sampledInstances/discProb.csv',delimiter=',') 
    
    #--------------------------------------------------------------------------
    # The setting of basis functions, MDP, upper bound, and miscellaneous
    bf      = {}
    mdp     = {}
    ub      = {}
    misc    = {}
     
    #--------------------------------------------------------------------------
    # A discrete probability defined in the 
    #   Adelman, Daniel, and Diego Klabjan. "Computing near-optimal policies
    #   in generalized joint replenishment." INFORMS Journal on Computing 24,
    #   no. 1 (2012): 148-164.
    def DiscProb(trial,i):
        r=discProb[trial,i]
        if r<1/3:
            return 2
        elif r<2/3:
            return 4
        else:
            return 8
    
    #--------------------------------------------------------------------------
    # MDP setting
    mdp.update({ 'mdp_name'             :    'INS_1'})                          
    mdp.update({ 'type'                 :    GeneralizedJointReplenishment})
    mdp.update({ 'Adrr'                 :    ''})
    mdp.update({ 'trial'                :    ''})
    mdp.update({ 'dimX'                 :    6}) 
    mdp.update({ 'dimU'                 :    None}) 
    mdp.update({ 'discount'             :    None}) 
    mdp.update({ 'SA_numSamples'        :    None}) 
    mdp.update({ 'actionDiscFactor'     :    None}) 
    mdp.update({ 'majorFixCost'         :    100}) 
    mdp.update({ 'minorFixCost'         :    minorCost[trial,range(mdp['dimX'])]}) 
    mdp.update({ 'consumptionRate'      :    consRate[trial,range(mdp['dimX'])]})
    mdp.update({ 'holdingCost'          :    np.asarray([0 for _ in range(mdp['dimX'])])})
    mdp.update({ 'Threads'              :    8})
    mdp.update({ 'invPerc'              :    1.00})
    mdp.update({ 'round'                :    5}) 
    
    #--------------------------------------------------------------------------
    # Left for future use. Some GJR instances require using the following constant
    # uppBound =sum(mdp['consumptionRate'][_]*upBound[trial,_] for _ in \
    #               range(mdp['dimX'])) + sum(mdp['consumptionRate'])/mdp['dimX']
    
    #--------------------------------------------------------------------------
    # Three ways to set up invUppBounds
    
    # 1) Constant 
    #mdp.update({ 'invUppBounds'         :   np.asarray([ uppBound for i in range(mdp['dimX'])])})
    
    # 2) Random 
    # mdp.update({ 'invUppBounds'         :   [mdp['consumptionRate'][_]*10*rndCapacity[trial,_] + \
    #                                              mdp['consumptionRate'][_] for _ in range(mdp['dimX'])]}) 
         
    # 3) Discerete     
    # mdp.update({ 'invUppBounds'         :   np.asarray([uppBound*DiscProb(trial,_) for _ in range(mdp['dimX'])])}) 
     
    mdp.update({ 'invUppBounds'         :   [mdp['consumptionRate'][_]*10*rndCapacity[trial,_] + \
                                                 mdp['consumptionRate'][_] for _ in range(mdp['dimX'])]}) 
         
    #--------------------------------------------------------------------------
    # Total capacity (maxOrder)
    x = mdp['invUppBounds']
    x = np.sort(x)
    x = x[range(floor((mdp['dimX'])*mdp['invPerc']))]
    mdp.update({ 'maxOrder'             :     sum(x)}) 
    
    #--------------------------------------------------------------------------
    # Bases function setting
    bf.update({ 'dimX'                  :     mdp['dimX']})
    bf.update({ 'BF_List'               :     [0,50,100,150,200,250]})
    bf.update({ 'Margin'                :     0.001})
    bf.update({ 'BF_num'                :     5001})
    bf.update({ 'bandwidth'             :     None})
    bf.update({ 'isStationary'          :     False})
    bf.update({ 'bandWidth_LB'          :     np.divide(mdp['invUppBounds'],-1) })
    bf.update({ 'bandWidth_UB'          :     np.divide(mdp['invUppBounds'], 1) })#
    bf.update({ 'BF_Trials'             :     5})    
    
    #--------------------------------------------------------------------------
    # Upper bound setting
    ub.update({ 'oftenUpdateUB'         :     150})
    ub.update({ 'roleHorizon'           :     5})
    ub.update({ 'trajLen'               :     1000})    
    
    #--------------------------------------------------------------------------                                                    
    # Miscellaneous setting
    misc['runTime']                  = 60*60 # runtime limit: Minutes * Seconds
    misc['intialContrSamples']       = 5000
         
    #--------------------------------------------------------------------------
    # Round     
    mdp['consumptionRate']              = np.round(mdp['consumptionRate'],mdp['round'])
    mdp['minorFixCost']                 = np.round(mdp['minorFixCost'],mdp['round'])
    mdp['invUppBounds']                 = np.round(mdp['invUppBounds'],mdp['round'])
    mdp['maxOrder']                     = np.round(mdp['maxOrder'],mdp['round'])
    
    #--------------------------------------------------------------------------
    # Wrap up all settings.
    info = {}
    info['mdp']         = mdp
    info['bf']          = bf
    info['ub']          = ub
    info['misc']        = misc
    return info


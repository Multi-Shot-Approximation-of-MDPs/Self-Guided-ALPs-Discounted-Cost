"""
-------------------------------------------------------------------------------

    Authors:    Parshan Pakiman  | https://parshanpakiman.github.io/homepage/
                Selva Nadarajah  | https://selvan.people.uic.edu/
                         
    Licensing Information: The MIT License
-------------------------------------------------------------------------------
"""
from BasisFunction.fourierBasisFunctions import FourierBasis
from MDP.PerishableInventory.perishableInventory import PerishableInventoryPartialBacklogLeadTime 
from scipy.stats import uniform
from math import log,gamma,pi
from numpy.linalg import norm
from numpy import arange

def experimentSetup():
    
    mdp              = {}
    cs               = {}
    bf               = {}
    lb               = {}
    misc             = {}    
    
    #--------------------------------------------------------------------------
    # Constraint sampling setting
    cs.update( {'CS_num'                :     10000,  
                'SA_numSamples'         :     1000 }) 
    
    #--------------------------------------------------------------------------
    # MDP configuration
    mdp.update({'mdp_name'              :     'INS_0'})
    mdp.update({'type'                  :     PerishableInventoryPartialBacklogLeadTime})
    mdp.update({'dimX'                  :     3})
    mdp.update({'dimU'                  :     1})
    mdp.update({'discount'              :     .95})
    mdp.update({'leadTime'              :     2,
                'lifeTime'              :     2,
                'purchase_cost'         :     20*pow(mdp.get('discount'),2),
                'holding_cost'          :     2,
                'disposal_cost'         :     5,
                'backlogg_cost'         :     10,
                'lostsale_cost'         :     100,
                'maxOrder'              :     10,     
                'maxBacklog'            :    -10,
                'distMean'              :     5,
                'distStd'               :     2,
                'distMin'               :     0,
                'distMax'               :     10,
                'actionDiscFactor'      :     1,        #MUST BE LESS THAN OR EQUAL 1 
                'trajLen'               :     1000,    
                'trajNum'               :     240,       
                'LB'                    :     None,     # Optional; in this case LNS_BOUND
                'CPU_CORE'              :     8 })
    
    mdp.update({'SA_numSamples'         :     cs.get('SA_numSamples')})
    mdp.update({'stateSamplingRvs'      :     uniform( loc = [ mdp.get('maxBacklog'),0,0],
                                                        scale=[-mdp.get('maxBacklog')+mdp.get('maxOrder'),
                                                                   mdp.get('maxOrder'),
                                                                   mdp.get('maxOrder')])})

    mdp.update({'actionSamplingRvs'     :     uniform(  loc = 0, scale=mdp.get('maxOrder'))})
    mdp.update({'initDistRV_GreedyPol'  :     uniform(loc=[5,5,5],scale=0)}) 
    mdp.update({'initDistRV_ALP'        :     mdp.get('stateSamplingRvs')})
    
    #--------------------------------------------------------------------------
    # Basis functions setting
    bf.update({ 'type'                  :     FourierBasis})
    bf.update({ 'dimX'                  :     mdp['dimX']})
    bf.update({ 'BF_num'                :     150})                 # *** BF_num > max(BF_List)
    bf.update({ 'BF_List'               :     arange(10,110,10)})
    bf.update({ 'bandwidth'             :     0.001})
    bf.update({ 'isStationary'          :     False})
    bf.update({ 'bandWidth_LB'          :     0.001})
    bf.update({ 'bandWidth_UB'          :     0.01})
    bf.update({ 'BF_Trials'             :     1})
    
    #--------------------------------------------------------------------------
    # Lower bound parameter based on the following paper:
    # Lin, Qihang, Selvaprabu Nadarajah, and Negar Soheili. "Revisiting
    # approximate linear programming: Constraint-violation learning with
    # applications to inventory control and energy storage." Management 
    # Science (2019).
    lb.update({'dimXU'                  :    mdp['dimX'] + mdp['dimU'] ,
               'R'                      :    mdp['maxOrder']/2          }) # R=mdp['maxOrder']/2 
    
    lb = {
          'LNS_const_log_prob'          :    -log((mdp['maxOrder'] - mdp['maxBacklog'])*pow(mdp['maxOrder'],3)),      
          'LNS_const_log_gamma'         :    -log(gamma((lb['dimXU']/2) + 1) / ((pow(pi,lb['dimXU']/2))*pow(lb['R'],lb['dimXU']))), 
          'LNS_R_Q'                     :     lb['R'] + norm([ (mdp['maxOrder'] - mdp['maxBacklog']),
                                                                mdp['maxOrder'],
                                                                mdp['maxOrder']])  # R+Q
          }

    #--------------------------------------------------------------------------
    # Miscellaneous setting
    misc.update({'optGap'                :     0.02})

    #--------------------------------------------------------------------------
    #Wrap up the setting and return
    info           = {}
    info['mdp']    = mdp
    info['bf']     = bf
    info['cs']     = cs
    info['lb']     = lb
    info['misc']   = misc
    return info
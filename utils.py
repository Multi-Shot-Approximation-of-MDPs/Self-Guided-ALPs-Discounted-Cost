"""
-------------------------------------------------------------------------------

    Authors:    Parshan Pakiman  | [https://parshanpakiman.github.io/homepage]
                Selva Nadarajah  | https://selvan.people.uic.edu/
                         
    Licensing Information: The MIT License
-------------------------------------------------------------------------------
"""
import os,itertools
import numpy as np
from itertools import chain, combinations 

"""
For a given MDP setup (expInfo; see, e.g., MDP/Instances/INS_1.py), the 
following function makes the following directories:
    1) Output folder
    2) SAMPLED_STATE_ACTION_BASIS
    3) ALP_COMPONENTS
"""
def getExperimentOutputPath(expInfo):
      outputPath = './Output/PIC/' + str(expInfo['mdp']['mdp_name']).upper() 
      if not os.path.exists(outputPath):
        os.makedirs(outputPath)    

      if not os.path.exists(outputPath+'/SAMPLED_STATE_ACTION_BASIS'):
        os.makedirs(outputPath+'/SAMPLED_STATE_ACTION_BASIS') 
        
      if not os.path.exists(outputPath+'/ALP_COMPONENTS'):
        os.makedirs(outputPath+'/ALP_COMPONENTS') 

      return outputPath
   
"""
This function generates a string that summarizes the parameters of an 
instance of PIC.
"""
def getExperimentSetupSummary(dictionary, indent=0):
       out = str("")
       for key, value in dictionary.items():
          if isinstance(value, dict):
             out+=getExperimentSetupSummary(value, indent+1)
          else:
             out+= '{:{width}.{prec}} = {:{width}.{prec}} \n'.format(key, str(value),width=30, prec=60)        
       return str(out)
   
""" Print a string and write the string into a logger if it is needed. """
def printWithPickle(string, logger, isReturn=False):
    if not isReturn:
        print(string)
    else:
        print(string,end='',flush=True)
    logger.write(string + '\n')

"""
All subsets of a set for a given size of subsets
Please see https://www.geeksforgeeks.org/python-program-to-get-all-subsets-of-given-size-of-a-set/             
"""
def findsubsets(s, n): 
    return list(map(list, itertools.combinations(s, n))) 

"""
Returns true if an array is in a list, and false otherwise. 
Please see https://stackoverflow.com/questions/23979146/check-if-numpy-array-is-in-list-of-numpy-arrays          
"""
def arreqclose_in_list(myarr, list_arrays):
    return next((True for elem in list_arrays if elem.size == myarr.size and np.allclose(elem, myarr)), False)

"""
Drops elements in a list that are 'small'
"""
def dropZeros(x):
             if abs(x) < 1e-4:
                 x = 0.0
             return x

"""
Generate powerset
"""
def powerset(iterable):
    xs = list(iterable)
    return list(chain.from_iterable(combinations(xs,n) for n in range(len(xs)+1)))

"""
Find location of element in a pSet
"""
def whereIsElementInPowerSet(pSet,element):
    idx = []
    itr = 0
    for subset in pSet:
        if element in subset:
           idx.append(itr)
        itr+=1  
    return idx

"""
Showing a message (i.e., instance number) to a user in a terminal.
"""
def selfGuidedHeader(mdpName,trial,applicationName,width):
    header      = '{:{fill}^{w}}'.format('Self-guided Approximate Linear Programs',fill=' ',w=width)
    title       = '{:{fill}^{w}}'.format(applicationName,fill=' ',w=width)
    ins         = '{:{fill}^{w}}'.format("Solving Instance " + str(mdpName),fill=' ',w=width)
    trialMsg    = '{:{fill}^{w}}'.format("Trial  " + str(trial+1),fill=' ',w=width)
    
    print('{:{fill}^{w}}'.format('-',fill='-',w=len(header)))
    print(header.upper())
    print('{:{fill}^{w}}'.format(' ',fill=' ',w=len(header)))
    print(title.upper())
    print('{:{fill}^{w}}'.format(' ',fill=' ',w=len(header)))
    print(ins.upper())
    
    if trial:
        print('{:{fill}^{w}}'.format(' ',fill=' ',w=len(header)))
        print(trialMsg.upper())
    print('{:{fill}^{w}}'.format('-',fill='-',w=len(header)))
    
    
    
"""
Showing a message (i.e., instance number) to a user in a terminal.
"""
def AK_Header(mdpName,trial,applicationName,width):
    header      = '{:{fill}^{w}}'.format('Adaptive basis function generation for GJR: Adelman and Klabjan 2012',fill=' ',w=width)
    title       = '{:{fill}^{w}}'.format(applicationName,fill=' ',w=width)
    ins         = '{:{fill}^{w}}'.format("Solving Instance " + str(mdpName),fill=' ',w=width)
    trialMsg    = '{:{fill}^{w}}'.format("Trial  " + str(trial+1),fill=' ',w=width)
    
    print('{:{fill}^{w}}'.format('-',fill='-',w=len(header)))
    print(header.upper())
    print('{:{fill}^{w}}'.format(' ',fill=' ',w=len(header)))
    print(title.upper())
    print('{:{fill}^{w}}'.format(' ',fill=' ',w=len(header)))
    print(ins.upper())
    print('{:{fill}^{w}}'.format(' ',fill=' ',w=len(header)))
    print(trialMsg.upper())
    print('{:{fill}^{w}}'.format('-',fill='-',w=len(header)))    

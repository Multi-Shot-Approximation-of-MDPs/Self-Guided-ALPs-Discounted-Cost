###############################################################################
# Created: Parshan Pakiman  | https://parshanpakiman.github.io/homepage/
#                           | http://business.uic.edu/faculty/parshan-pakiman
#                          
# Licensing Information: The MIT License
###############################################################################

import imp

listLibs = ['gurobipy',
            'numpy',
            'itertools',
            'time',
            'pandas',
            'os',
            'gc',
            'multiprocessing',
            'functools',
            'scipy',
            'math',
            'sampyl',
            'sys',
            'textwrap']
notFound = []
found    = []

for lib in listLibs:
    try:
        imp.find_module(lib)
        found.append(lib)
    except ImportError:
        notFound.append(lib)
   
print("{:<15}\t{:>5}".format('Library','Found'))
print('{:{fill}^{w}}'.format('-',fill='-',w=21))     
for lib in found:
    print("{:<15}\t{:>5}".format(lib,'YES'))
for lib in notFound:
    print("{:<15}\t{:>5}".format(lib,'NO'))
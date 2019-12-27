###############################################################################
# Created: Parshan Pakiman  | https://parshanpakiman.github.io/homepage/
#                           | http://business.uic.edu/faculty/parshan-pakiman
#                          
# Licensing Information: The MIT License
###############################################################################


#------------------------------------------------------------------------------
# The following loop calls the main function for PIC application for instance I.
# Examples:
#   1) for I in {1,2,3}
#   2) for I in {1..1}  
for I in {1..1}
do
    python main_PIC.py "MDP.PerishableInventory.Instances.INS_$I" $I
done  


#------------------------------------------------------------------------------
#
#    Authors:    Parshan Pakiman  | https://parshanpakiman.github.io/homepage/
#                Selva Nadarajah  | https://selvan.people.uic.edu/
#                         
#    Licensing Information: The MIT License
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# The following loop calls the main function for PIC application for instance I.
# Examples:
#   1) for I in {1,2,3}
#   2) for I in {1..1}  
for I in {0..0}
do
    python main_PIC.py "MDP.PerishableInventory.Instances.INS_$I" $I
done  


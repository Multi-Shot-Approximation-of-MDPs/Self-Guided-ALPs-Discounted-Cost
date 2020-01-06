#------------------------------------------------------------------------------
#
#	AOUTHORS:    Parshan Pakiman, Selva Nadarajah 
#
#	HOMEPAGES:   https://parshanpakiman.github.io/homepage/ 
#                https://selvan.people.uic.edu/
#                         
#	LICENSE: The MIT License
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# The outer loop iterates over instances (I) of GJR and the inner loop specifies
# which trial (t) of the instance (I) should be solved. In the inner loop, we 
# call the main function of the GJR application.
# Examples:
#   1) for I in {1,2,3}
#   2) for I in {1..1} 
for I in {0..0}
    do
    for t in {0..0}
    do
        python main_GJR.py "MDP.GeneralizedJointReplenishment.Instances.INS_$I" $I $t
    done
done


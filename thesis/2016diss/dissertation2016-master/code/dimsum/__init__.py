#!/usr/bin/env python

###############################################################################
# GLOBAL DATA COLUMNS OVERVIEW
###############################################################################
# columns for DIMSUM data
# need to predict 5,6,8
# 6 can be deterministically filled in given the full sequence of 5
###############################################################################
# 1. token offset
# 2. word
# 3. lowercase lemma
# 4. POS
# 5. MWE tag
# 6. offset of parent token (i.e. previous token in the same MWE), if applicable
# 7. strength level encoded in the tag, if applicable. Currently not used
# 8. supersense label, if applicable
# 9. sentence ID
###############################################################################
DATA_COLUMNS=["tokenoffset","word","lowerlemma","POStag","MWEtag","parenttokenoffset","strengthlevel","supersense","ID"]

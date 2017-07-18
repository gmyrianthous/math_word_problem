#!/bin/bash

function usage(){
    echo "Usage for $0"
    echo "   First argument is DiMSUM style file with Sentences separated by newlines"
    echo "   and columns separated by tabs. Creates two files for training and testing."
    echo "   Appends the suffix .train and .dev to the original file given in \$1"
    echo "Example:"
    echo "   \$ $0 foo"
    echo "   ----------------------------------------------------------------------------"
    echo "   Total sentences in file foo: 4799"
    echo "   Total split for dev (40% of total sentences): 1919"
    echo "   Total split train                           : 2880"
    echo "   ----------------------------------------------------------------------------"
    echo "   Split file counts for foo.train and foo.dev"
    echo "   foo.train : 2880"
    echo "   foo.dev   : 1919"
    echo "   ----------------------------------------------------------------------------"

}

# must provide argument to split training and dev data
if [ -z $1 ]
then
    echo
    echo "ERROR: First argument is empty, must supply file to split for train/dev"
    echo
    usage
    exit
fi

###############################################################################
# take 40% of train and distribute in dev
# assume first argument is training file
###############################################################################
trainf=$1.train
devf=$1.dev
tmpf=$1.tmp
tab='-_T_-'
newline='-_N_-'

# make sentences one per line with special tab and newline sequences
sed "s/\t/$tab/g" $1 | sed "s/$/$newline/g" | tr -d '\n' | sed "s/""$newline""$newline""/\n/g" > $tmpf

# count the quantity of sentences
count=($(wc -l $tmpf))
# extract total for training
total=${count[0]}

# calculate 40% of data for dev
devtotal=$(expr \( "$total" \* 50 \) / 100)
# train quantity is the beginning 60%
traintotal=$(expr "$total" - "$devtotal") 

echo "-------------------------------------------------------------------------------"
# print out the quantities to verify 
echo "Total sentences in file $1: $total"
echo "Total split for dev "'(40% of total sentences):'" $devtotal"
echo "Total split train                           : $traintotal"
echo "-------------------------------------------------------------------------------"

# split training as first portion of file
head -n $traintotal $tmpf | sed 's/$/\n/g' | sed "s/$newline/\n/g" | sed "s/$tab/\t/g"  > $trainf
# dev is what is left, tail 40% of $1
tail -n $devtotal $tmpf | sed 's/$/\n/g' | sed "s/$newline/\n/g" | sed "s/$tab/\t/g" > $devf

# print out the file line counts
echo "Split file counts for $trainf and $devf"
# grep empty line/newlines only
echo "$trainf : $(grep '^$' $trainf | wc -l)"
echo "$devf   : $(grep '^$' $devf | wc -l)"
echo "-------------------------------------------------------------------------------"
rm $tmpf

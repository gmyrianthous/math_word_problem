#!/bin/bash

###############################################################################
# extract sentences from DiMSUM data files
###############################################################################

function usage(){
    echo
    echo "Usage $0"
    echo "   Take dimsum file and extract sentences"
    echo "   Reads words from each sentence and places each sentence on a single line"
    echo "Example:"
    echo "   $0 dimsum16.train"
    echo
}

if [ -z $1 ] 
then
    echo "ERROR: No argument provided to script for DiMSUM sentence extraction."
    usage
    exit
fi

newline="-_N_-"
awk '{print $2}' $1 | sed "s/$/$newline/g" | tr -d '\n' | sed "s/$newline$newline/\n/g" | sed "s/$newline/ /g"

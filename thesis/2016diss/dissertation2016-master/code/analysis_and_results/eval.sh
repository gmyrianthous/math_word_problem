#!/bin/bash

if [ -z $1 ];
then
    exit
fi

~/data/DIMSUM/dimsum-data/scripts/dimsumeval.py -C ~/data/DIMSUM/dimsum-data/dimsum16.test $1 $2 $3

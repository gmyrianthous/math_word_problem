#!/usr/bin/env python

# simple script to average results from fold output evaluation

import sys

# initial item in line with evaluation results for fold
eval_items = ["MWEs:","Supersenses:","Combined:"]

# print average results for accumulated data list of folds
# for the specific eval item name
def print_avg_eval(data, name):
    ACC = None
    if len(data[0]) > 3:
        ACC = []
    P = []
    R = []
    F = []
    for evalresults in data:
        for i in range(0, len(evalresults)):
            result = evalresults[i].split('=')
            if ACC != None and result[0] == 'Acc':
                ACC.append(float(result[2]))
            elif result[0] == 'P':
                P.append(float(result[2]))
            elif result[0] == 'R':
                R.append(float(result[2]))
            elif result[0] == 'F':
                F.append(float(result[1].replace('%','')))
    pavg = sum(P)/len(P)
    ravg = sum(R)/len(R)
    favg = sum(F)/len(F)
    if ACC != None:
        accavg = sum(ACC)/len(ACC)
        print(name, "Acc=%.2f"%accavg,"P=%.2F"%pavg,"R=%.2f"%ravg,"F=%.2f"%favg,"%")
    else:
        print(name, "P=%.2F"%pavg,"R=%.2f"%ravg,"F=%.2f"%favg,"%")

def usage():
    use = '''
    Averages k-fold data from file with dimsumeval.py results
    Provide filename for evaluation results and argument
    for the quantity of folds (k) in results file.

    Example: baseline 1 evaluations results file with 10-folds
    {0} {1} {2}

    Arguments provided {3}
    '''.format(sys.argv[0], "b1_fold_evals.txt", 10, str(sys.argv))
    print(use)

try:
    # open fold file, assume first arg as filename
    with open(sys.argv[1]) as foldfile:
        k = int(sys.argv[2])
        results = {}
        fold_data = {}
        for line in foldfile:
            line = line.strip()
            line = line.split()
            if len(line) > 0:
                key = line[0]
                value = line[1:]
                if key in eval_items:
                    if not key in fold_data:
                        fold_data[key] = []
                    fold_data[key].append(value)

        for key in eval_items:
            if len(fold_data[key]) != k:
                print("K-folds should have",k,"data points but only",len(fold_data[key]),"found. Quitting.")
                exit()

        for key in eval_items:
            print_avg_eval(fold_data[key], key)

except IOError:
    print("IO Exception, inexistent file?")
    usage()
except IndexError:
    print("Index Error, k not provided?")
    usage()

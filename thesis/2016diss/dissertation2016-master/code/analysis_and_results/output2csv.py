#!/usr/bin/env python

import sys

with open(sys.argv[1]) as of:
    lines = of.readlines()

epoch_scores = {}
for line in lines:
    split = line.strip().split()
    if len(split) == 3:
        if split[1] == 'epoch':
            epoch = int(split[2].strip('...'))
            if not epoch in epoch_scores:
                epoch_scores[epoch] = {}
    elif len(split) == 4:
        dataset = split[2]
        if dataset == 'dev:' or dataset == 'test:':
            score = float(split[3])
            if not dataset in epoch_scores[epoch]:
                epoch_scores[epoch][dataset] = []
            epoch_scores[epoch][dataset].append(score)
line="epoch,"
for dataset in epoch_scores[0].items():
    line+=str(dataset[0].strip(':'))+","
print(line)
for epoch,dataset in epoch_scores.items():
    line = str(epoch)+","
    for dataset,scores in epoch_scores[epoch].items():
        try:
            avg = sum(scores)/len(scores)
        except:
            avg = float('nan')
        line+="%0.2f"%avg+","
    print(line)

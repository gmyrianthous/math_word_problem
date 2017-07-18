# Code

This readme contains information about code usage and incremental results during training and testing of the code for the dissertation.

## CRF Baselines

The baselines for the task use Conditional Random Fields (CRF) to predict the MWE tag sequences and the supersense tag sequences.

There are three baselines which use different features and labels. 

All contexts windows are +/- 2 words for feature extraction.

### Algorithms

#### Baseline 1 

Uses 2 CRFs for the MWE and Supersense sequences, then combines results.

1.  Extract features from training data with +/-2 word context (Lemma+POS Tags)
2.  Train MWE CRF to predict MWE (BIO) Tags using features in (1)
3.  Predict MWE tag sequences using (2)
4.  Extract features from training data with +/-2 word context (Lemma+POS Tags+MWE tags)
5.  Train Supersense CRF to predict Supersense Tags using features in (4)
6.  Predict Supersense tag sequences using testing data+predictions from (3)
7.  Systematically create parent offset sequences for MWE sequences
8.  Combine MWE+Supersense predictions in (6) with parent offset sequences in (7) for final predictions

#### Baseline 2 

Uses 2 CRFs for the MWE and Supersense sequences, then combines results.

It is identical to Baseline 1, but does not use MWE tags for the Supersense prediction CRF

1.  Extract features from training data with +/-2 word context (Lemma+POS Tags)
2.  Train MWE CRF to predict MWE (BIO) Tags using features in (1)
3.  Predict MWE tag sequences using (2)
4.  Extract features from training data with +/-2 word context (Lemma+POS Tags)
5.  Train Supersense CRF to predict Supersense Tags using features in (4)
6.  Predict Supersense tag sequences using testing data+predictions from (3)
7.  Systematically create parent offset sequences for MWE sequences
8.  Combine MWE+Supersense predictions in (6) with parent offset sequences in (7) for final predictions

#### Baseline 3

Uses a single CRF and concatenates labels, then splits them for final prediction.

1.  Extract features from training data with +/-2 word context (Lemma+POS Tags)
2.  Train CRF to predict concatenated MWE Tags+Supersense tags using features in (1)
3.  Predict Concatenated sequences using (2)
4.  Split MWE and Supersense sequences from (3)
5.  Systematically create parent offset sequences for MWE sequences
6.  Combine MWE and Supersense predictions in (4) with parent offset sequences in (5) for final predictions

### Evaluation

Evaluation Results are obtained using the `dimsumeval.py` script provided with the shared task data.

#### Baselines comparison results

The evaluation results for `dimsum16.test.blind` for the three baselines can be seen below. 

Baseline 1

    SUMMARY SCORES
    ==============
    MWEs: P=290/437=0.6636 R=290/1115=0.2601 F=37.37%
    Supersenses: P=2247/5060=0.4441 R=2247/4745=0.4736 F=45.83%
    Combined: Acc=12588/16500=0.7629 P=2537/5497=0.4615 R=2537/5860=0.4329 F=44.68%

Baseline 2

    SUMMARY SCORES
    ==============
    MWEs: P=290/437=0.6636 R=290/1115=0.2601 F=37.37%
    Supersenses: P=2119/4278=0.4953 R=2119/4745=0.4466 F=46.97%
    Combined: Acc=12471/16500=0.7558 P=2409/4715=0.5109 R=2409/5860=0.4111 F=45.56%

Baseline 3

    SUMMARY SCORES
    ==============
    MWEs: P=541/981=0.5515 R=543/1115=0.4870 F=51.72%
    Supersenses: P=2218/4691=0.4728 R=2218/4745=0.4674 F=47.01%
    Combined: Acc=12728/16500=0.7714 P=2759/5672=0.4864 R=2761/5860=0.4712 F=47.87%

#### 10-fold cross-evaluation

In addition, as a verification mechanism, a 10-fold cross evaluation was performed using the training data with the following results.
    
Baseline 1

    Running evaluations for 10 folds to file b1_fold_evals.txt
    Average results for 10 folds
    MWEs: P=0.74 R=0.36 F=48.04 %
    Supersenses: P=0.58 R=0.60 F=59.22 %
    Combined: Acc=0.79 P=0.60 R=0.56 F=57.59 %

Baseline 2

    Running evaluations for 10 folds to file b2_fold_evals.txt
    Average results for 10 folds
    MWEs: P=0.74 R=0.36 F=48.04 %
    Supersenses: P=0.63 R=0.57 F=60.02 %
    Combined: Acc=0.78 P=0.64 R=0.53 F=58.18 %

Baseline 3

    Running evaluations for 10 folds to file b3_fold_evals.txt
    Average results for 10 folds
    MWEs: P=0.58 R=0.54 F=55.53 %
    Supersenses: P=0.60 R=0.58 F=59.37 %
    Combined: Acc=0.80 P=0.60 R=0.58 F=58.86 %

#### Comparison

In relation to the other system's scores submitted for the original task:

|Place #|System|Team|F1-Score|Resources|
|-------|------|----|--------|---------|
|1|S214|ICL-HD|57.77|++|
||S249|UW-CSE|57.71|++|
||S248|UW-CSE|57.1||
|2|S106|UFRGS&LIF|50.27||
|3|S227|VectorWeavers|49.94|++|
||**Baseline 3**||**47.87**||
|4|S225|UTU|47.13|++|
|5|S211|UTU|46.17|+|
||S254|UTU|45.79||
||**Baseline 2**||**45.56**||
||**Baseline 1**||**44.68**||
|6|S108|WHUNlp|25.71||

The results column explains the [data conditions](http://dimsum16.github.io/) for the shared task:

    The final column indicates the resource condition: 
    systems entered in the open condition (all resources allowed) are designated “++”; 
    “+” indicates the more restricted semi-supervised closed condition, 
    while the remaining systems are in the closed condition (most restrictive).

Baseline 3 works best, there is something about the 'jointness' of tags that works better such as misalignment issues with supersense and [Bb] tags.

#### Split train 80/20

Running the same baselines by splitting the `dimsum16.train` into 80% train (`dimsum16.train.80.train`), 20% test (`dimsum16.train.20.dev`), we get the following results.

Baseline 1

    SUMMARY SCORES
    ==============
    MWEs: P=332/445=0.7461 R=328/929=0.3531 F=47.93%
    Supersenses: P=2640/4163=0.6342 R=2640/4026=0.6557 F=64.48%
    Combined: Acc=10724/13196=0.8127 P=2972/4608=0.6450 R=2968/4955=0.5990 F=62.11%

Baseline 2

    SUMMARY SCORES
    ==============
    MWEs: P=332/445=0.7461 R=328/929=0.3531 F=47.93%
    Supersenses: P=2538/3694=0.6871 R=2538/4026=0.6304 F=65.75%
    Combined: Acc=10603/13196=0.8035 P=2870/4139=0.6934 R=2866/4955=0.5784 F=63.07%

Baseline 3

    SUMMARY SCORES
    ==============
    MWEs: P=459/803=0.5716 R=456/929=0.4909 F=52.82%
    Supersenses: P=2620/3927=0.6672 R=2620/4026=0.6508 F=65.89%
    Combined: Acc=10824/13196=0.8202 P=3079/4730=0.6510 R=3076/4955=0.6208 F=63.55%

### Joint tag data statistics

As it can be seen in multiple tests, splitting the training data and using it as train/dev/test for multiple systems seems to perform better. This would lead me to believe that the data in the original `dimsum16.test` sentences is **more difficult** to learn in someway. 

In general, not just due to the size, some joint tags are represented up to 36x higher in `dimsum16.train` then `dimsum16.test` even though the `train` set is only 4.7 times larger than then `test` set (4799 train/1000 test). 

Here is a table with the joint tag counts in each file sorted by `train`/`test` factors.

|data/dimsum16.train.tagger | data/dimsum16.test.tagger | Jointtag|train/test (Times jointtag more train then test)|test/train (Percent jointtag of test in train)|
|---------------------------|---------------------------|---------|------------------------------------------------|----------------------------------------------|
|36|1| B__v.possession|36|0.0277777778|
|35|1| O__n.motive|35|0.0285714286|
|719|25| O__n.food|28.76|0.0347705146|
|291|11| B__v.social|26.4545454545|0.0378006873|
|55|3| B__v.body|18.3333333333|0.0545454545|
|32|2| i__|16|0.0625|
|107|7| B__n.food|15.2857142857|0.0654205607|
|561|37| o__|15.1621621622|0.0659536542|
|780|64| O__v.social|12.1875|0.0820512821|
|157|15| B__v.stative|10.4666666667|0.0955414013|
|352|36| O__n.possession|9.7777777778|0.1022727273|
|682|76| O__n.location|8.9736842105|0.1114369501|
|107|12| O__v.consumption|8.9166666667|0.1121495327|
|461|53| B__n.group|8.6981132075|0.114967462|
|129|15| B__n.event|8.6|0.1162790698|
|1274|152| O__n.group|8.3815789474|0.1193092622|
|105|13| O__n.animal|8.0769230769|0.1238095238|
|192|24| B__v.cognition|8|0.125|
|8|1| o__n.body|8|0.125|
|518|69| O__n.event|7.5072463768|0.1332046332|
|1124|154| O__n.time|7.2987012987|0.1370106762|
|138|19| B__v.motion|7.2631578947|0.1376811594|
|727|101| O__n.cognition|7.198019802|0.1389270977|
|114|16| B__v.communication|7.125|0.1403508772|
|162|23| B__n.location|7.0434782609|0.1419753086|
|1050|158| B__|6.6455696203|0.1504761905|
|627|103| O__v.motion|6.0873786408|0.1642743222|
|24|4| o__n.artifact|6|0.1666666667|
|47|8| O__v.competition|5.875|0.170212766|
|31|6| B__n.possession|5.1666666667|0.1935483871|
|5623|1113| I__|5.0521114106|0.1979370443|
|25|5| O__n.process|5|0.2|
|1117|226| O__v.cognition|4.9424778761|0.2023276634|
|34|7| B__v.emotion|4.8571428571|0.2058823529|
|1571|329| O__n.person|4.7750759878|0.2094207511|
|19|4| B__n.quantity|4.75|0.2105263158|
|361|78| O__v.possession|4.6282051282|0.216066482|
|46|10| O__n.substance|4.6|0.2173913043|
|98|23| O__v.body|4.2608695652|0.2346938776|
|3196|754| O__v.stative|4.2387267905|0.2359198999|
|44099|10445| O__|4.2220201053|0.2368534434|
|1110|264| O__v.communication|4.2045454545|0.2378378378|
|1018|247| O__n.artifact|4.1214574899|0.242632613|
|8|2| o__n.person|4|0.25|
|58|16| O__n.natural_object|3.625|0.275862069|
|138|39| B__n.act|3.5384615385|0.2826086957|
|119|34| B__v.change|3.5|0.2857142857|
|755|216| O__n.act|3.4953703704|0.2860927152|
|111|32| O__v.creation|3.46875|0.2882882883|
|284|82| B__n.person|3.4634146341|0.2887323944|
|100|29| B__n.cognition|3.4482758621|0.29|
|129|41| O__n.body|3.1463414634|0.3178294574|
|229|73| B__n.communication|3.1369863014|0.3187772926|
|653|210| O__n.communication|3.1095238095|0.3215926493|
|3|1| o__n.communication|3|0.3333333333|
|135|48| O__n.quantity|2.8125|0.3555555556|
|379|136| O__v.emotion|2.7867647059|0.3588390501|
|36|13| O__n.relation|2.7692307692|0.3611111111|
|286|108| O__n.attribute|2.6481481481|0.3776223776|
|420|176| O__v.change|2.3863636364|0.419047619|
|230|98| O__v.perception|2.3469387755|0.4260869565|
|9|4| B__n.animal|2.25|0.4444444444|
|17|8| B__n.state|2.125|0.4705882353|
|180|86| B__n.artifact|2.0930232558|0.4777777778|
|14|7| B__n.feeling|2|0.5|
|6|3| B__v.creation|2|0.5|
|43|23| O__n.phenomenon|1.8695652174|0.5348837209|
|7|4| O__n.shape|1.75|0.5714285714|
|44|26| B__n.time|1.6923076923|0.5909090909|
|21|13| O__n.other|1.6153846154|0.619047619|
|16|10| B__n.body|1.6|0.625|
|105|68| O__n.state|1.5441176471|0.6476190476|
|54|35| O__n.feeling|1.5428571429|0.6481481481|
|94|61| O__v.contact|1.5409836066|0.6489361702|
|3|2| B__n.process|1.5|0.6666666667|
|4|3| B__n.relation|1.3333333333|0.75|
|8|7| O__n.plant|1.1428571429|0.875|
|10|9| B__v.contact|1.1111111111|0.9|
|24|22| B__n.attribute|1.0909090909|0.9166666667|
|11|11| B__v.perception|1|1|
|2|2| O__v.weather|1|1|
|1|1| b__n.act|1|1|
|5|6| B__n.substance|0.8333333333|1.2|
|2|3| B__n.plant|0.6666666667|1.5|
|3|7| B__n.natural_object|0.4285714286|2.3333333333|
|3|8| B__n.phenomenon|0.375|2.6666666667|
|14|0| o__n.group|0|0|
|8|0| o__n.act|0|0|
|8|0| o__n.cognition|0|0|
|7|0| B__v.consumption|0|0|
|7|0| b__|0|0|
|7|0| o__n.animal|0|0|
|5|0| o__n.quantity|0|0|
|5|0| o__n.time|0|0|
|4|0| B__v.competition|0|0|
|4|0| b__n.person|0|0|
|4|0| o__n.food|0|0|
|4|0| o__v.stative|0|0|
|4|0| b__n.artifact|0|0|
|3|0| o__n.event|0|0|
|2|0| o__v.social|0|0|
|2|0| o__n.other|0|0|
|2|0| o__n.possession|0|0|
|2|0| b__n.quantity|0|0|
|1|0| o__v.emotion|0|0|
|1|0| b__n.possession|0|0|
|1|0| o__n.attribute|0|0|
|1|0| b__n.location|0|0|
|1|0| o__n.substance|0|0|
|1|0| o__v.cognition|0|0|
|1|0| b__v.possession|0|0|
|1|0| o__v.communication|0|0|
|1|0| b__n.animal|0|0|
|1|0| b__n.group|0|0|
|1|0| b__n.communication|0|0|
|1|0| o__n.process|0|0|
|1|0| o__n.natural_object|0|0|
|0|1| B__n.motive|0|0|
|0|1| B__n.other|0|0|
|0|1| o__n.state|0|0|

## LSTM-CRF (Tagger)

Run all new trained models with dimsum test data

Get permission/license information for tagger code

### Preliminary results and Epoch cutoff

Initial training results for of LSTM-CRF ["tagger"](https://github.com/natemccoy/tagger) adaption

Using 60% of `dimsum16.train` for training and 20% for each {dev,test}

Initial parameters for Tagger adaption

    tag_scheme=generic,
    lower=False,
    zeros=False,
    char_dim=25,
    char_lstm_dim=25,
    char_bidirect=True,
    word_dim=100,
    word_lstm_dim=100,
    word_bidirect=True,
    pre_emb=,
    all_emb=False,
    cap_dim=0,
    crf=True,
    dropout=0.5,
    lr_method=sgd-lr_.005

Epoch results during initial training 
Uses Word column from dimsum data

Epoch 1

    SUMMARY SCORES
    ==============
    MWEs: P=422/1425=0.2961 R=426/1115=0.3821 F=33.37%
    Supersenses: P=477/2703=0.1765 R=477/4745=0.1005 F=12.81%
    Combined: Acc=10738/16500=0.6508 P=899/4128=0.2178 R=903/5860=0.1541 F=18.05%

Epoch 25

    SUMMARY SCORES
    ==============
    MWEs: P=579/1636=0.3539 R=581/1115=0.5211 F=42.15%
    Supersenses: P=2058/4625=0.4450 R=2058/4745=0.4337 F=43.93%
    Combined: Acc=11873/16500=0.7196 P=2637/6261=0.4212 R=2639/5860=0.4503 F=43.53%

Epoch 41

    SUMMARY SCORES
    ==============
    MWEs: P=436/801=0.5443 R=436/1115=0.3910 F=45.51%
    Supersenses: P=2017/4267=0.4727 R=2017/4745=0.4251 F=44.76%
    Combined: Acc=12403/16500=0.7517 P=2453/5068=0.4840 R=2453/5860=0.4186 F=44.89%

Epoch 99

    SUMMARY SCORES
    ==============
    MWEs: P=436/801=0.5443 R=436/1115=0.3910 F=45.51%
    Supersenses: P=2017/4267=0.4727 R=2017/4745=0.4251 F=44.76%
    Combined: Acc=12403/16500=0.7517 P=2453/5068=0.4840 R=2453/5860=0.4186 F=44.89%

The previous example used no word embeddings and a word dimension of 100, so Additional tests were performed and verified that after about 25 Epochs, there is little to no improvement.

This image was taken from preliminary results using google news word embeddings and 300 word dimension.

![Initial Epoch Cutoff Tests](http://i.imgur.com/wdLc0L8.png)

In both the cases of 100/300 word dimension with/without word embeddings, after about 25 Epochs, the results did not improve.

Therefore the Epoch limit is set to 25.

### Results for LSTM-CRF

#### Split train 60/20/20

##### Models with 100 Epochs

Pre-embeddings GoogleNews-vectors-negative300.txt using 300 word dimension

Parameters

     tag_scheme=generic,
     lower=False,
     zeros=False,
     char_dim=25,
     char_lstm_dim=25,
     char_bidirect=True,
     word_dim=300,
     word_lstm_dim=100,
     word_bidirect=True,
     pre_emb=GoogleNews-vectors-negative300.txt,
     all_emb=False,
     cap_dim=0,
     crf=True,
     dropout=0.5,
     lr_method=sgd-lr_.005

Commands 

    ./tagger.py --model=./models/epochs100/tag_scheme=generic\,lower=False\,zeros=False\,char_dim=25\,char_lstm_dim=25\,char_bidirect=True\,word_dim=300\,word_lstm_dim=100\,word_bidirect=True\,pre_emb=GoogleNews-vectors-negative300.txt\,all_emb=False\,cap_dim=0\,crf=True\,dropout=0.5\,lr_method=sgd-lr_.005/ --input=../data/dimsum16.train.20.test.sentences --output=dimsum16.train.20.test.tagged
    ../data/conversion/tagger2dimsum.py ../data/dimsum16.train.20.test.blind dimsum16.train.20.test.tagged > dimsum16.train.20.test.tagged.preds
    evaluation/dimsumeval.py ../data/dimsum16.train.20.test dimsum16.train.20.test.tagged.preds 

Score

    SUMMARY SCORES
    ==============
    MWEs: P=476/682=0.6979 R=474/1203=0.3940 F=50.37%
    Supersenses: P=2928/4389=0.6671 R=2928/4356=0.6722 F=66.96%
    Combined: Acc=12008/14816=0.8105 P=3404/5071=0.6713 R=3402/5559=0.6120 F=64.03%

No pre-embeddings 100 using word dimension

Parameters

    tag_scheme=generic,
    lower=False,
    zeros=False,
    char_dim=25,
    char_lstm_dim=25,
    char_bidirect=True,
    word_dim=100,
    word_lstm_dim=100,
    word_bidirect=True,
    pre_emb=,
    all_emb=False,
    cap_dim=0,
    crf=True,
    dropout=0.5,
    lr_method=sgd-lr_.005/

Commands

    ./tagger.py --model=./models/epochs100/tag_scheme=generic\,lower=False\,zeros=False\,char_dim=25\,char_lstm_dim=25\,char_bidirect=True\,word_dim=100\,word_lstm_dim=100\,word_bidirect=True\,pre_emb=\,all_emb=False\,cap_dim=0\,crf=True\,dropout=0.5\,lr_method=sgd-lr_.005/ --input=../data/dimsum16.train.20.test.sentences --output=dimsum16.train.20.test.nopreemb.tagged
    ../data/conversion/tagger2dimsum.py ../data/dimsum16.train.20.test.blind dimsum16.train.20.test.nopreemb.tagged > dimsum16.train.20.test.nopreemb.tagged.preds
    evaluation/dimsumeval.py ../data/dimsum16.train.20.test dimsum16.train.20.test.nopreemb.tagged.preds 

Score

    SUMMARY SCORES
    ==============
    MWEs: P=477/762=0.6260 R=480/1203=0.3990 F=48.74%
    Supersenses: P=2555/4204=0.6078 R=2555/4356=0.5865 F=59.70%
    Combined: Acc=11583/14816=0.7818 P=3032/4966=0.6106 R=3035/5559=0.5460 F=57.65%

##### Models with 25 Epochs

Sorted By F1-Combined Score

|tag_scheme|lower|zeros|char_dim|char_lstm_dim|char_bidirect|word_dim|word_lstm_dim|word_bidirect|pre_emb|all_emb|cap_dim|crf|dropout|lr_method|MWE F1| Supersense F1| Combined F1|
|----------|-----|-----|--------|-------------|-------------|--------|-------------|-------------|-------|-------|-------|---|-------|---------|------|--------------|------------|
|generic|False|False|25|25|True|300|300|True|GoogleNews-vectors-negative300.txt|False|0|True|0.5|sgd-lr_.005|49.86|67.92|64.71|
|generic|False|False|25|25|True|300|100|True|GoogleNews-vectors-negative300.txt|False|0|True|0.5|sgd-lr_.005|52|67.57|64.66|
|generic|False|False|5|5|True|300|300|True|GoogleNews-vectors-negative300.txt|False|0|True|0.5|sgd-lr_.005|50.46|67.62|64.5|
|generic|False|False|5|5|True|300|200|True|GoogleNews-vectors-negative300.txt|False|0|True|0.5|sgd-lr_.005|50.89|66.96|64.04|
|generic|False|False|5|5|True|300|100|True|GoogleNews-vectors-negative300.txt|False|0|True|0.5|sgd-lr_.005|53.48|66.54|64|
|generic|False|False|25|25|True|300|100|True|GoogleNews-vectors-negative300.txt|False|0|True|0.25|sgd-lr_.005|49.29|67.23|63.63|
|generic|False|False|25|25|True|300|200|True|GoogleNews-vectors-negative300.txt|False|0|True|0.5|sgd-lr_.005|49.62|66.6|63.55|
|generic|False|False|5|5|True|300|100|True|GoogleNews-vectors-negative300.txt|False|0|True|0.25|sgd-lr_.005|50.16|nan|63.54|
|generic|False|False|5|5|True|300|100|True|GoogleNews-vectors-negative300.txt|False|0|True|0|sgd-lr_.005|49.47|66.6|63.53|
|generic|False|False|5|5|True|300|300|True|GoogleNews-vectors-negative300.txt|False|0|True|0.25|sgd-lr_.005|49.46|66.5|63.46|
|generic|False|False|25|25|True|300|300|True|GoogleNews-vectors-negative300.txt|False|0|True|0.25|sgd-lr_.005|49.18|66.39|63.26|
|generic|False|False|25|25|True|300|300|True|GoogleNews-vectors-negative300.txt|False|0|True|0|sgd-lr_.005|48.19|66.57|63.21|
|generic|False|False|5|5|True|300|200|True|GoogleNews-vectors-negative300.txt|False|0|True|0.25|sgd-lr_.005|48.2|66.12|63.03|
|generic|False|False|25|25|True|300|200|True|GoogleNews-vectors-negative300.txt|False|0|True|0.25|sgd-lr_.005|43.4|66.72|62.82|
|generic|False|False|5|5|True|300|300|True|GoogleNews-vectors-negative300.txt|False|0|True|0|sgd-lr_.005|45.41|66.28|62.76|
|generic|False|False|5|5|True|300|200|True|GoogleNews-vectors-negative300.txt|False|0|True|0|sgd-lr_.005|48.05|65.67|62.6|
|generic|False|False|25|25|True|300|200|True|GoogleNews-vectors-negative300.txt|False|0|True|0|sgd-lr_.005|41.49|66.14|62.15|
|generic|False|False|25|25|True|300|300|True||False|0|True|0.25|sgd-lr_.005|50.52|58.69|57.04|
|generic|False|False|5|5|True|300|300|True||False|0|True|0.5|sgd-lr_.005|46.23|59.57|56.99|
|generic|False|False|25|25|True|300|200|True||False|0|True|0.5|sgd-lr_.005|49.21|58.45|56.7|
|generic|False|False|5|5|True|200|200|True||False|0|True|0.5|sgd-lr_.005|48.71|58.6|56.61|
|generic|False|False|25|25|True|100|200|True||False|0|True|0.25|sgd-lr_.005|49.91|58.14|56.57|
|generic|False|False|25|25|True|200|200|True||False|0|True|0.5|sgd-lr_.005|48.7|58.38|56.5|
|generic|False|False|25|25|True|100|300|True||False|0|True|0.25|sgd-lr_.005|44.83|58.96|56.37|
|generic|False|False|25|25|True|200|100|True||False|0|True|0.25|sgd-lr_.005|49.54|57.91|56.33|
|generic|False|False|25|25|True|100|100|True||False|0|True|0|sgd-lr_.005|49.7|57.91|56.32|
|generic|False|False|5|5|True|200|300|True||False|0|True|0.25|sgd-lr_.005|48.33|58.23|56.3|
|generic|False|False|25|25|True|200|300|True||False|0|True|0|sgd-lr_.005|49.36|57.96|56.24|
|generic|False|False|25|25|True|100|100|True||False|0|True|0.5|sgd-lr_.005|49.71|57.85|56.21|
|generic|False|False|25|25|True|100|200|True||False|0|True|0.5|sgd-lr_.005|47.5|58.09|56.14|
|generic|False|False|5|5|True|100|100|True||False|0|True|0.25|sgd-lr_.005|47.32|58.15|56.1|
|generic|False|False|25|25|True|300|300|True||False|0|True|0.5|sgd-lr_.005|44.73|58.45|56.08|
|generic|False|False|5|5|True|300|200|True||False|0|True|0.25|sgd-lr_.005|46.19|58.31|56.07|
|generic|False|False|25|25|True|200|100|True||False|0|True|0.5|sgd-lr_.005|49.85|57.5|55.96|
|generic|False|False|25|25|True|300|200|True||False|0|True|0|sgd-lr_.005|48.38|57.65|55.95|
|generic|False|False|25|25|True|200|300|True||False|0|True|0.5|sgd-lr_.005|42.97|58.72|55.93|
|generic|False|False|25|25|True|100|300|True||False|0|True|0|sgd-lr_.005|46.65|58.1|55.91|
|generic|False|False|5|5|True|200|300|True||False|0|True|0.5|sgd-lr_.005|45.39|58.48|55.89|
|generic|False|False|25|25|True|100|300|True||False|0|True|0.5|sgd-lr_.005|46.6|58.03|55.87|
|generic|False|False|25|25|True|100|100|True||False|0|True|0.25|sgd-lr_.005|49.75|57.59|55.85|
|generic|False|False|25|25|True|300|200|True||False|0|True|0.25|sgd-lr_.005|49.98|57.22|55.73|
|generic|False|False|25|25|True|300|100|True||False|0|True|0.25|sgd-lr_.005|44.12|58|55.6|
|generic|False|False|25|25|True|200|300|True||False|0|True|0.25|sgd-lr_.005|46.99|57.47|55.59|
|generic|False|False|25|25|True|200|200|True||False|0|True|0.25|sgd-lr_.005|50.22|56.78|55.55|
|generic|False|False|5|5|True|300|100|True||False|0|True|0.5|sgd-lr_.005|46.36|57.92|55.55|
|generic|False|False|25|25|True|100|200|True||False|0|True|0|sgd-lr_.005|47.29|57.38|55.54|
|generic|False|False|5|5|True|100|300|True||False|0|True|0|sgd-lr_.005|46.07|57.75|55.54|
|generic|False|False|5|5|True|200|100|True||False|0|True|0.25|sgd-lr_.005|49.14|57.25|55.52|
|generic|False|False|25|25|True|300|100|True||False|0|True|0.5|sgd-lr_.005|43.5|58.01|55.52|
|generic|False|False|5|5|True|100|300|True||False|0|True|0.25|sgd-lr_.005|48.2|57.21|55.5|
|generic|False|False|5|5|True|100|200|True||False|0|True|0.5|sgd-lr_.005|47.39|57.59|55.49|
|generic|False|False|5|5|True|300|200|True||False|0|True|0.5|sgd-lr_.005|46.94|57.42|55.3|
|generic|False|False|5|5|True|200|100|True||False|0|True|0.5|sgd-lr_.005|46.56|57.13|55.16|
|generic|False|False|5|5|True|300|200|True||False|0|True|0|sgd-lr_.005|43.79|57.78|55.13|
|generic|False|False|5|5|True|200|200|True||False|0|True|0.25|sgd-lr_.005|47.23|56.92|55.12|
|generic|False|False|5|5|True|200|100|True||False|0|True|0|sgd-lr_.005|47.3|56.97|55.01|
|generic|False|False|5|5|True|300|300|True||False|0|True|0.25|sgd-lr_.005|44.62|57.31|54.98|
|generic|False|False|25|25|True|200|100|True||False|0|True|0|sgd-lr_.005|47.76|56.54|54.9|
|generic|False|False|5|5|True|100|200|True||False|0|True|0.25|sgd-lr_.005|47.58|56.53|54.86|
|generic|False|False|5|5|True|100|100|True||False|0|True|0.5|sgd-lr_.005|40.86|57.78|54.84|
|generic|False|False|25|25|True|300|100|True||False|0|True|0|sgd-lr_.005|46.35|56.73|54.82|
|generic|False|False|5|5|True|100|300|True||False|0|True|0.5|sgd-lr_.005|48.36|56.32|54.74|
|generic|False|False|25|25|True|200|200|True||False|0|True|0|sgd-lr_.005|47.09|56.51|54.71|
|generic|False|False|5|5|True|300|300|True||False|0|True|0|sgd-lr_.005|45.87|56.51|54.59|
|generic|False|False|5|5|True|200|200|True||False|0|True|0|sgd-lr_.005|45.01|57|54.59|
|generic|False|False|5|5|True|200|300|True||False|0|True|0|sgd-lr_.005|46.77|56.13|54.44|
|generic|False|False|5|5|True|300|100|True||False|0|True|0.25|sgd-lr_.005|41.85|57.14|54.42|
|generic|False|False|5|5|True|100|100|True||False|0|True|0|sgd-lr_.005|46.67|56.04|54.27|
|generic|False|False|5|5|True|100|200|True||False|0|True|0|sgd-lr_.005|42.18|56.51|54.14|
|generic|False|False|5|5|True|300|100|True||False|0|True|0|sgd-lr_.005|47.45|55.44|53.87|
|generic|False|False|25|25|True|300|300|True||False|0|True|0|sgd-lr_.005|42.68|55.98|53.79|
|generic|False|False|25|25|True|300|100|True|GoogleNews-vectors-negative300.txt|False|0|True|0|sgd-lr_.005|9.84|46.42|41.3|

Observations and relationships on data

* Top 10 Highest Combined F1 Score use Preembeddings
* Top 8 Use Dropout
* Top 5 Have Dropout at 0.5
* LSTM dimension effects MWE tags more then Supersense tags. 
* No correlation with char/char LSTM dims and MWE/SS F1-Scores
* "lowering" words causes a major drop in combined F1-Score 
* Bidirection char/word `True` default value has highest F1-Score
* Disabling CRF decreases F1-Score drastically
* Default optimization stochastic gradient descent proved best

A word LSTM dimension of 100 increases MWE F1-Score by an average of 2%, with a variation of less than 0.1% differance in combined F1 scores.  

|tag_scheme|lower|zeros|char_dim|char_lstm_dim|char_bidirect|word_dim|word_lstm_dim|word_bidirect|pre_emb|all_emb|cap_dim|crf|dropout|lr_method|MWE F1| Supersense F1| Combined F1|
|----------|-----|-----|--------|-------------|-------------|--------|-------------|-------------|-------|-------|-------|---|-------|---------|------|--------------|------------|
|generic|False|False|25|25|True|300|100|True|GoogleNews-vectors-negative300.txt|False|0|True|0.5|sgd-lr_.005|*52*|67.57|**64.66**|
|generic|False|False|25|25|True|300|300|True|GoogleNews-vectors-negative300.txt|False|0|True|0.5|sgd-lr_.005|*49.86*|67.92|**64.71**|
|generic|False|False|25|25|True|300|400|True|GoogleNews-vectors-negative300.txt|False|0|True|0.5|sgd-lr_.005|51.14|67.75|64.59|
|generic|False|False|25|25|True|300|50|True|GoogleNews-vectors-negative300.txt|False|0|True|0.5|sgd-lr_.005|50.64|66.69|63.85|
|generic|False|False|25|25|True|300|75|True|GoogleNews-vectors-negative300.txt|False|0|True|0.5|sgd-lr_.005|50.62|66.63|63.8|
|generic|False|False|25|25|True|300|200|True|GoogleNews-vectors-negative300.txt|False|0|True|0.5|sgd-lr_.005|49.62|66.6|63.55|
|generic|False|False|25|25|True|300|500|True|GoogleNews-vectors-negative300.txt|False|0|True|0.5|sgd-lr_.005|49.41|66.6|63.5|
|generic|False|False|25|25|True|300|25|True|GoogleNews-vectors-negative300.txt|False|0|True|0.5|sgd-lr_.005|50.69|65.2|62.52|
|generic|False|False|25|25|True|300|600|True|GoogleNews-vectors-negative300.txt|False|0|True|0.5|sgd-lr_.005|44.52|64.8|61.21|

There does not seem to be a correlation with char/char LSTM dims and MWE/SS F1-Scores

![Char Dim Chart](http://i.imgur.com/4ZKQx0N.jpg)

Therefore the highest score is picked from tests of 10 as seen in the table below:

|tag_scheme|lower|zeros|char_dim|char_lstm_dim|char_bidirect|word_dim|word_lstm_dim|word_bidirect|pre_emb|all_emb|cap_dim|crf|dropout|lr_method|MWE F1| Supersense F1| Combined F1|
|----------|-----|-----|--------|-------------|-------------|--------|-------------|-------------|-------|-------|-------|---|-------|---------|------|--------------|------------|
|generic|False|False|**10**|**20**|True|300|100|True|GoogleNews-vectors-negative300.txt|False|0|True|0.5|sgd-lr_.005|54.39|67.78|**65.14**|
|generic|False|False|20|20|True|300|100|True|GoogleNews-vectors-negative300.txt|False|0|True|0.5|sgd-lr_.005|53.96|67.55|64.88|
|generic|False|False|25|25|True|300|100|True|GoogleNews-vectors-negative300.txt|False|0|True|0.5|sgd-lr_.005|52|67.57|64.66|
|generic|False|False|30|20|True|300|100|True|GoogleNews-vectors-negative300.txt|False|0|True|0.5|sgd-lr_.005|54.1|67.02|64.6|
|generic|False|False|10|30|True|300|100|True|GoogleNews-vectors-negative300.txt|False|0|True|0.5|sgd-lr_.005|51.54|67.57|64.58|
|generic|False|False|40|40|True|300|100|True|GoogleNews-vectors-negative300.txt|False|0|True|0.5|sgd-lr_.005|52.8|67.16|64.48|
|generic|False|False|30|40|True|300|100|True|GoogleNews-vectors-negative300.txt|False|0|True|0.5|sgd-lr_.005|51.51|67.36|64.48|
|generic|False|False|20|40|True|300|100|True|GoogleNews-vectors-negative300.txt|False|0|True|0.5|sgd-lr_.005|52.18|67.1|64.39|
|generic|False|False|40|10|True|300|100|True|GoogleNews-vectors-negative300.txt|False|0|True|0.5|sgd-lr_.005|50.93|67.08|64.16|
|generic|False|False|5|5|True|300|100|True|GoogleNews-vectors-negative300.txt|False|0|True|0.5|sgd-lr_.005|53.48|66.54|64|
|generic|False|False|30|30|True|300|100|True|GoogleNews-vectors-negative300.txt|False|0|True|0.5|sgd-lr_.005|50.43|66.91|63.97|
|generic|False|False|20|10|True|300|100|True|GoogleNews-vectors-negative300.txt|False|0|True|0.5|sgd-lr_.005|52.22|66.76|63.97|
|generic|False|False|10|10|True|300|100|True|GoogleNews-vectors-negative300.txt|False|0|True|0.5|sgd-lr_.005|47.92|67.23|63.9|
|generic|False|False|10|40|True|300|100|True|GoogleNews-vectors-negative300.txt|False|0|True|0.5|sgd-lr_.005|50.11|66.75|63.86|
|generic|False|False|30|10|True|300|100|True|GoogleNews-vectors-negative300.txt|False|0|True|0.5|sgd-lr_.005|50.39|66.61|63.77|
|generic|False|False|20|30|True|300|100|True|GoogleNews-vectors-negative300.txt|False|0|True|0.5|sgd-lr_.005|48.93|66.51|63.51|
|generic|False|False|40|20|True|300|100|True|GoogleNews-vectors-negative300.txt|False|0|True|0.5|sgd-lr_.005|50.74|66.16|63.47|
|generic|False|False|40|30|True|300|100|True|GoogleNews-vectors-negative300.txt|False|0|True|0.5|sgd-lr_.005|50.54|65.78|63.09|

Cannot modify `zeros` parameter as it changes important values to one or more `0` characters. It was changing MWE offset numbers to a sequence of `0`s. It was removed from the program. 

When the `lower` bool is set to `True` there is a 5.7% decrease in F1-Score.

|tag_scheme|lower|zeros|char_dim|char_lstm_dim|char_bidirect|word_dim|word_lstm_dim|word_bidirect|pre_emb|all_emb|cap_dim|crf|dropout|lr_method|MWE F1| Supersense F1| Combined F1|
|----------|-----|-----|--------|-------------|-------------|--------|-------------|-------------|-------|-------|-------|---|-------|---------|------|--------------|------------|
|generic|False|False|10|20|True|300|100|True|GoogleNews-vectors-negative300.txt|False|0|True|0.5|sgd-lr_.005|49.87|66.47|63.65|
|generic|True|False|10|20|True|300|100|True|GoogleNews-vectors-negative300.txt|False|0|True|0.5|sgd-lr_.005|33.00|62.38|57.94|


Character bidirection and word bidirection default values of `True` proved to be the best results.

|tag_scheme|lower|zeros|char_dim|char_lstm_dim|char_bidirect|word_dim|word_lstm_dim|word_bidirect|pre_emb|all_emb|cap_dim|crf|dropout|lr_method|MWE F1| Supersense F1| Combined F1|
|----------|-----|-----|--------|-------------|-------------|--------|-------------|-------------|-------|-------|-------|---|-------|---------|------|--------------|------------|
|generic|False|False|10|20|True|300|100|True|GoogleNews-vectors-negative300.txt|False|0|True|0.5|sgd-lr_.005|51.07|66.95|64.1|
|generic|False|False|10|20|False|300|100|True|GoogleNews-vectors-negative300.txt|False|0|True|0.5|sgd-lr_.005|51.26|66.74|63.92|
|generic|False|False|10|20|True|300|100|False|GoogleNews-vectors-negative300.txt|False|0|True|0.5|sgd-lr_.005|49.94|65.51|62.64|
|generic|False|False|10|20|False|300|100|False|GoogleNews-vectors-negative300.txt|False|0|True|0.5|sgd-lr_.005|48.23|65.63|62.42|

Disabling CRF useage in the LSTM-CRF decreases performance by 26.8% F1-Score. `True` is the original default value.

|tag_scheme|lower|zeros|char_dim|char_lstm_dim|char_bidirect|word_dim|word_lstm_dim|word_bidirect|pre_emb|all_emb|cap_dim|crf|dropout|lr_method|MWE F1| Supersense F1| Combined F1|
|----------|-----|-----|--------|-------------|-------------|--------|-------------|-------------|-------|-------|-------|---|-------|---------|------|--------------|------------|
|generic|False|False|10|20|True|300|100|True|GoogleNews-vectors-negative300.txt|False|0|False|0.5|sgd-lr_.005|nan|43.25|37.27|
|generic|False|False|10|20|True|300|100|True|GoogleNews-vectors-negative300.txt|False|0|True|0.5|sgd-lr_.005|51.07|66.95|64.10|

Original default model with `lr_model` optimization parameter set to `sgd-lr_.005` proved most effective with `sgdmomentum` in second.

|tag_scheme|lower|zeros|char_dim|char_lstm_dim|char_bidirect|word_dim|word_lstm_dim|word_bidirect|pre_emb|all_emb|cap_dim|crf|dropout|lr_method|MWE F1| Supersense F1| Combined F1|
|----------|-----|-----|--------|-------------|-------------|--------|-------------|-------------|-------|-------|-------|---|-------|---------|------|--------------|------------|
|generic|False|False|10|20|True|300|100|True|gnvn300.txt|False|0|True|0.5|sgdmomentum-lr_.001|49.96|67.1|64.04|
|generic|False|False|10|20|True|300|100|True|gnvn300.txt|False|0|True|0.5|sgd-lr_.005|52.19|66.29|63.77|
|generic|False|False|10|20|True|300|100|True|gnvn300.txt|False|0|True|0.5|sgd-lr_.002|49.8|66.67|63.65|
|generic|False|False|10|20|True|300|100|True|gnvn300.txt|False|0|True|0.5|adadelta|49.94|66.1|63.3|
|generic|False|False|10|20|True|300|100|True|gnvn300.txt|False|0|True|0.5|adadelta-lr_.001|49.94|66.1|63.3|
|generic|False|False|10|20|True|300|100|True|gnvn300.txt|False|0|True|0.5|adadelta-lr_.002|49.94|66.1|63.3|
|generic|False|False|10|20|True|300|100|True|gnvn300.txt|False|0|True|0.5|adadelta-lr_.005|49.94|66.1|63.3|
|generic|False|False|10|20|True|300|100|True|gnvn300.txt|False|0|True|0.5|rmsprop-lr_.002|50.81|65.7|62.78|
|generic|False|False|10|20|True|300|100|True|gnvn300.txt|False|0|True|0.5|sgdmomentum-lr_.002|47.76|65.8|62.68|
|generic|False|False|10|20|True|300|100|True|gnvn300.txt|False|0|True|0.5|rmsprop-lr_.001|50.38|65.68|62.56|
|generic|False|False|10|20|True|300|100|True|gnvn300.txt|False|0|True|0.5|sgdmomentum-lr_.005|52.71|64.91|62.47|
|generic|False|False|10|20|True|300|100|True|gnvn300.txt|False|0|True|0.5|sgd-lr_.001|41.14|63.78|59.96|
|generic|False|False|10|20|True|300|100|True|gnvn300.txt|False|0|True|0.5|rmsprop-lr_.005|41.66|60.94|57.5|
|generic|False|False|10|20|True|300|100|True|gnvn300.txt|False|0|True|0.5|adagrad-lr_.005|11.24|6.44|7.45|
|generic|False|False|10|20|True|300|100|True|gnvn300.txt|False|0|True|0.5|rmsprop|11.24|6.44|7.45|
|generic|False|False|10|20|True|300|100|True|gnvn300.txt|False|0|True|0.5|adagrad-lr_.001|nan|7.24|5.93|
|generic|False|False|10|20|True|300|100|True|gnvn300.txt|False|0|True|0.5|adagrad-lr_.002|nan|2.06|1.75|

Using `all_emb` did not provide any improvements. Default provided better results

Therefore, Fixed values:

     pre_emb=GoogleNews-vectors-negative300.txt
     word_dim=300
     word_lstm_dim=100
     char_dim=10
     char_lstm_dim=20
     dropout=0.5		(default)
     lower=False		(default)
     char_bidirect=True		(default)
     word_bidirect=True		(default)
     crf=True			(default)
     lr_method=sgd-lr_.005 	(default)
     all_emb=False 		(default)
     cap_dim=0 			(default)

#### Split train 80/20

#### Final results

These are the final results used with the hyper parmeters chosen by the 60/20/20 split. They represent the highest scoring model for the LSTM-CRF for the `dimsum16.test` sentences. 

The model was trained on `dimsum16.train.80.train` with dev file `dimsum16.train.20.dev` and test file `dimsum16.test.blind`

|tag_scheme|lower|zeros|char_dim|char_lstm_dim|char_bidirect|word_dim|word_lstm_dim|word_bidirect|pre_emb|all_emb|cap_dim|crf|dropout|lr_method|MWE F1| Supersense F1| Combined F1|
|----------|-----|-----|--------|-------------|-------------|--------|-------------|-------------|-------|-------|-------|---|-------|---------|------|--------------|------------|
|generic|False|False|10|20|True|300|100|True|gnvn300.txt|False|0|True|0.5|sgd-lr_.005|46.71|54.59|53.27|

Output from `dimsumeval.py` script:

    SUMMARY SCORES
    ==============
    MWEs: P=435/755=0.5762 R=438/1115=0.3928 F=46.71%
    Supersenses: P=2541/4564=0.5567 R=2541/4745=0.5355 F=54.59%
    Combined: Acc=12917/16500=0.7828 P=2976/5319=0.5595 R=2979/5860=0.5084 F=53.27%

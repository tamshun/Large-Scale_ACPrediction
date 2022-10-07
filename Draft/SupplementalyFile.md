## 1.5. Model measurements
To evaluate performance of the models, balanced accuracy (BA), recall, precision, and matthew's correlation coefficient (MCC) were calculated. The four measures are defined below,
$$ \tag{1} BA = \frac{1}{2}(TPR + TNR) $$
$$ \tag{2} recall = \frac{TP}{TP+FN} $$
$$ \tag{3} precision = \frac{TP}{TP+FP} $$
$$ \tag{4} MCC = \frac{TP × TN − FP × FN}{\sqrt{(TP + FP)(TP + FN)(TN + FP)(TN + FN)}} $$
where TP, TN, FP, and FN stand for true positives, true negatives, false positives, and false negatives, respectively.


| Part  |    Attribute    |                 Descriptor                 | Dimensions |
| :---: | :-------------: | :----------------------------------------: | :--------: |
| Atom  |    Atom type    |               Atomic number                |    118     |
|   ^   |     Degree      |     Number of neighboring heavy atoms      |     6      |
|   ^   |  Formal Charge  |     Formal charge (− 2, − 1, 0, 1, 2)      |     5      |
|   ^   | Chirality label |       Nothing, R, S, or unrecognized       |     4      |
|   ^   |  Hybridization  | sp, sp^2^ , sp^3^ , sp^3^ d, or sp^3^ d^2^ |     5      |
|   ^   |   Aromaticity   |          Aromatic or not aromatic          |     1      |
| Bond  |    Bond type    |  Single, double, triple, or aromatic ring  |     4      |
|   ^   |      Ring       |         Ring bond or non-ring bond         |     1      |
|   ^   |   Bond stereo   |       None, any, Z, E, cis, or trans       |     6      |
Table: __Table X Initial invariants on atoms and bonds__


|  Method  |                     Hyperparameter                      |  Range(interval)   | log scale |
| :------: | :-----------------------------------------------------: | :----------------: | :-------: |
|    RF    |                number of decision trees                 |   50 - 1000 (1)    |   False   |
|    ^     |                 maximum depth of a tree                 |     4 - 50 (1)     |   False   |
|    ^     |             minimum samples to split a node             |    10^-8^ - 1.0    |   True    |
|   XGB    |                 alpha for L1 regulation                 |    10^-8^ - 1.0    |   True    |
|    ^     |                lambda for L2 regulation                 |    10^-8^ - 1.0    |   True    |
|    ^     |                     subsample ratio                     |     0.2 - 1.0      |   False   |
|    ^     |                 colsample rate by tree                  |     0.2 - 1.0      |   False   |
|    ^     |                 maximum depth of a tree                 |     3 - 9 (2)      |   False   |
|    ^     |         minimum sum of weights in splited nodes         |     2 - 10 (1)     |   False   |
|    ^     |                           eta                           |    10^-8^ - 1.0    |   True    |
|    ^     |                          gamma                          |    10^-8^ - 1.0    |   True    |
|   FCNN   |                 number of hidden layers                 |     2 - 3 (1)      |   False   |
|    ^     |                      dropout rate                       | 0.25 - 0.50 (0.25) |   False   |
|    ^     |             learning rate of adam optimizer             |  10^-4^ - 10^-2^   |   False   |
|    ^     |            number of steps for the scheduler            |     1 - 3 (1)      |   False   |
| FCNN_sep |            number of hidden layers for core             |     1 - 4 (1)      |   False   |
|    ^     |        number of hidden layers for substituents         |     1 - 4 (1)      |   False   |
|    ^     | number of hidden layers for concatenated feature vector |     2 - 8 (1)      |   False   |
|    ^     |          dimension of feature vector for core           |   70 - 100 (10)    |   False   |
|    ^     |      dimension of feature vector for substituents       |   70 - 100 (10)    |   False   |
|    ^     |                      dropout rate                       |  0.1 - 0.3 (0.05)  |   False   |
|    ^     |             learning rate of adam optimizer             |  10^-4^ - 10^-2^   |   False   |
|    ^     |            number of steps for the scheduler            |     1 - 3 (1)      |   False   |
|   MPNN   |                  number of convolution                  |     2 - 4 (1)      |   False   |
|    ^     |               dimension of feature vector               |   70 - 100 (10)    |   False   |
|    ^     |             number of hidden layers of FCNN             |     2 - 8 (1)      |   False   |
|    ^     |                      dropout rate                       |  0.1 - 0.3 (0.05)  |   False   |
|    ^     |             learning rate of adam optimizer             |  10^-4^ - 10^-2^   |   False   |
|    ^     |            number of steps for the scheduler            |     1 - 3 (1)      |   False   |
| MPNN_sep |             number of convolution for core              |     2 - 4 (1)      |   False   |
|    ^     |         number of convolution for substituents          |     2 - 4 (1)      |   False   |
|    ^     |          dimension of feature vector for core           |   70 - 100 (10)    |   False   |
|    ^     |      dimension of feature vector for substituents       |   70 - 100 (10)    |   False   |
|    ^     |             number of hidden layers of FCNN             |     2 - 8 (1)      |   False   |
|    ^     |                      dropout rate                       |  0.1 - 0.3 (0.05)  |   False   |
|    ^     |             learning rate of adam optimizer             |  10^-4^ - 10^-2^   |   False   |
|    ^     |            number of step for the scheduler             |     1 - 3 (1)      |   False   |

Table: __Table X__ Hyperparameters of models
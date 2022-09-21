---
output:
    word_document:
        path: ./Draft/Method.docx
        reference_doc: ./Draft/styles.docx
---

# 1. Method

## 1.1. Dataset
Compounds active against the activity classes were extracted from ChEMBL29 based on the following criteria; less than 1000Da, target confidence score of 9, interaction relationship type 'D', an exact potency value given as _K~i~_ or _K~d~_ values. In this study, MMP-cliffs were used as the definition of ACs with target independent criteria. 

MMPs with the potency difference of greater than $\mu + 2\sigma$, where $\mu$ and $\sigma$ represent mean and standerard deviation of the _pK~i~_ values in the corresponding activity classes, were defined as ACs and those of less than one were defined as non-ACs. The other MMPs with the potency difference of greater than one and less than the criteria were discarded.     

MMPs were generated with computationally efficient algorithm using Husian-Lee approach implemented in our previous study. For MMP generation, substructure exchange were limited in no more than 13 heavy atoms and maximum difference between substituent of a compound was limited in no more than eight heavy atoms.

Activity classes with greater than 50 MMPs were used for the further analysis, which resulted in 100 activity classes.

## 1.2. MMP fingerprints
Extended connectivity fingerprints of bond diameter 4 (ECFP4) were used as the molecular representation in present study. the process to calculate an MMP fingerprints is followed. Fingerprints for the core and two substituents were individually calculated. During these fingerprints calculation, features within bond diameter of one were eliminated from ECFP feature collection to clarify the contributions of features over bond diameter of two. For each part, identifiers corresponding to the features were sorted in ascending order and assigned to bits in the fingerprints vectors in the same order to prevent feature collision and make features contribute to AC prediction as many as possible. After the individual fingerprints calculation for the three parts, the XOR operation and AND operation were applied to the two substituent fingerprints to represent unique and common features in the two substituents separately and focus on the transformation. Finally, three fingerprints representing features of core, unique features between substituents, and common features between substituents were concatenated into one vector. MMP fingerprints calculation was conducted with inhouse Java and Python scripts based on _OEChem toolkit_   

## 1.3. Condenced graph of reaction representation
To apply graph neural networks approach to AC prediction, MMPs were represented in a single graph using the condensed graph of reaction (CGR) approach. The CGR formalism was originally conceived to combine reactants and products graphs based upon a superposition of invariant parts. The resulting CGR is a completely connected graph in which each node represents an atom and each edge a bond. In a CGR, the shared core of an MMP and the two exchanged substituent fragments form a single pseudo-molecule. The subgraph coming from the low potent compound was connected with the core via a single bond and the high potent compound a hypothetical zero-order bond. An MMP was converted into a pseudo-molecule using an in-house Python script with the RDKit API. 

## 1.4. Machine learning
To find best approach for AC prediction, four fingerprints-based approach which were support vector machine (SVM), extreme gradient boosting (XGboost), random forest, and fully connected neural network (FCNN), and a graph-based approach which was message passing neural network (MPNN) were used. For FCNN and MPNN, two distinct models which differ at molecular representation for the input were build. As control calculations, nearest neighbor (1NN) and 5NN were also compared. All models except the two neural networks were implemented with scikit-learn and the neural newtworks pytorch.


### 1.4.1. Support vector machine
SVM is a supervised learning method that aims to define the hyperplane separating given training instances with two class labels with maximising the mergin from the hyperplane. SVM was originally develpoed as linear classification method. If linear classification is not possible, SVM can be easily extend to nonlinear classification using kernel functions. In present study, MMPkernel which is a product of two individual Tanimoto kernel calculating core-wise and substituent-wise similarity.  

### 1.4.2. Random forest
RF is a supervised learning method that is an ensamble of multiple decision trees generated from randomly chosen training instances using bootstrapping. RF predict for test instances based on the majority of the indivisual prediction of decision trees.

### 1.4.3. Extreme gradient boosting
XGboost is a supervised learing method that is also emsamble of decision trees using gradient boosting. Gradient boosting iteratively generates decison trees so that each decision trees minimize the resudual error from previous models. XGBoost is a computationally efficient and accurate extention of gradient boosting, which is achieved by pararelizing the decision tree construction.

### 1.4.4. Fully-connected neural network
A FCNN consists of a series of connected perceptrons stored in several layers. Each perceptron receives signals from previous layer transform into scaler value with activation function and send it to next layer as a signal. In this study, two distinct FCNNs were implemented based on their input; FCNN with single feature vector represented by MMPfingerprints (simply called as FCNN), FCNN with three separated ECFP4 representing core and two substituents (FCNN_sep). For FCNN, input MMPfingerprints were mapped into probability indicating how likely the input MMP forms AC. In FCNN_sep, the individual fingerprints were input to several hidden layers and the outputs were concatenated into a vector which is sent to following hidden layers and mapped into the probability.  

### 1.4.5. Message passing neural network
MPNN is a graph convolutional neural network approach which accepts molecular graph as the input and can learn the way to convert input molecular graphs to its optimal feature vectors. During MPNN training step, feature vector on each atom iteratively merged with informations from its neighbor atoms and bonds so as to minimize the loss function. The initial features on each atom and bonds are listed in __Table X__. The transformed feature vectors of each atom are merged into single vector and connected to fully-connected neural network. Same as FCNN and FCNN_sep, two distinct MPNN were implemented corresponding to their input; MPNN with single CGR (simply called as MPNN), MPNN with three separated subgraphs representing core and two substituents (MPNN_sep). In MPNN_sep, feature vectors for each substructure were calculated with individual MPNN and concatenated into single vector which connected to fully-connected neural network.  

## 1.5. Model construction and hyperprameter optimization
The models were built for each activity class. When a data set randomly is separated into training and test set, one compound could be shared among several MMPs and this shared compound appears in both training and test set simaltaniously, which causes that a MMP in the training set having compounds shared with a test MMP would be highly focused on the similarity evaluation between training and test samples. This data leakage prevents to evaluate whether the models have learnt chemical features or not. To analyse influence from data leakage, two data splitting approach were conducted. In 'data leakage possibly included' splitting, data sets were randomly separated into training set (80%) and test set (20%). In 'data leakage excluded' splitting, advanced cross-validation (AXV) approach proposed by Horvath et al.[ref] was referenced. First, for each activity class, compounds used for making MMPs were collected and _n_ compounds were randomly chosen as 'kept-out' pool. Then, MMPs were recursively selected and assigned to training or test set. If neither compound of the MMP was shared with kept-out pool, the MMP was assigned to training set. If both compounds were in kept-out pool, the MMP was assigned to external test set. If one compound was in kept-out pool, the MMP was no longer used for the further analysis.  

random split, axv split

## 1.6. Model measurements
To evaluate performance of the models, balanced accuracy (BA), recall, precision, and matthew's correlation coefficient (MCC) were calculated. The four measures are defined below,
$$ \tag{1} BA = \frac{1}{2}(TPR + TNR) $$
$$ \tag{2} recall = \frac{TP}{TP+FN} $$
$$ \tag{3} precision = \frac{TP}{TP+FP} $$
$$ \tag{4} MCC = \frac{TP × TN − FP × FN}{\sqrt{(TP + FP)(TP + FN)(TN + FP)(TN + FN)}} $$
where TP, TN, FP, and FN stand for true positives, true negatives, false positives, and false negatives, respectively.

## Tables
--------------------------------------------------------------------------------------
 Part       Attribute         Descriptor                                  Dimensions
------- ------------------ -------------------------------------------- --------------
 Atom     Atom type          Atomic number                                   118         
          Degree             Number of neighboring heavy atoms                6          
          Formal Charge      Formal charge (− 2, − 1, 0, 1, 2)                5          
          Chirality label    Nothing, R, S, or unrecognized                   4          
          Hybridization      sp, sp^2^ , sp^3^ , sp^3^ d, or sp^3^ d^2^       5 
          Aromaticity        Aromatic or not aromatic                         1          
 Bond     Bond type          Single, double, triple, or aromatic ring         4          
          Ring               Ring bond or non-ring bond                       1          
          Bond stereo        None, any, Z, E, cis, or trans                   6        
--------------------------------------------------------------------------------------
Table: __Table X Initial invariants on atoms and bonds__

--------------------------------------------------------------------------------------
Method Hyperparameter Range(interval)
--------------------------------------------------------------------------------------
 SVM
 RF
 XGB
 FCNN
 MPNN
--------------------------------------------------------------------------------------
Table: __Table X__ Hyperparameters of models
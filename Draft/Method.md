# 3. Method

## 3.1. Dataset
Compounds active against the activity classes were extracted from ChEMBL29 based on the following criteria; less than 1000Da, target confidence score of 9, interaction relationship type 'D', an exact potency value given as Ki or Kd values. In this study, MMP-cliffs were used as the definition of ACs with target independent criteria. 

MMPs with the potency difference of greater than the mean value of the corresponding activity classes added by two times of its standard deviation were defined as ACs and those of less than 1 were defined as non-ACs. The other MMPs with the potency difference of greater than 1 and less than the criteria were discarded.     

MMPs were generated with computationally efficient algorithm using Husian-Lee approach implemented in our previous study. For MMP generation, substructure exchange were limited in no more than 13 heavy atoms and maximum difference between substituent of a compound was limited in no more than eight heavy atoms.

Activity classes with greater than 50 MMPs were used for the further analysis, which resulted in 100 activity classes.

## 3.2. MMP fingerprints
Extended connectivity fingerprints of bond diameter 4 (ECFP4) were used as the molecular representation in present study. the process to calculate an MMP fingerprints is followed. Fingerprints for the core and 2 substituents were individually calculated. During these fingerprints calculation, features within bond diameter 1 were eliminated from ECFP feature collection to clarify the contributions of features over bond diameter 2. For each part, identifiers corresponding to the features were sorted in ascending order and assigned to bits in the fingerprints vectors in the same order to prevent feature collision and make features contribute to AC prediction as many as possible. After the individual fingerprints calculation for the 3 parts, the XOR operation and AND operation were applied to the 2 substituent fingerprints to represent unique and common features in the two substituents separately and focus on the transformation. Finally, 3 fingerprints representing features of core, unique features between substituents, and common features between substituents were concatenated into one vector.  

## 3.3. Condenced graph of reaction representation
To apply graph neural networks approach to AC prediction, MMPs were represented in a single graph using the condensed graph of reaction (CGR) approach. The CGR formalism was originally conceived to combine reactants and products graphs based upon a superposition of invariant parts. The resulting CGR is a completely connected graph in which each node represents an atom and each edge a bond. In a CGR, the shared core of an MMP and the two exchanged substituent fragments are represented as a single pseudo-molecule. The fragment coming from the low potent compound was connected with the core via a single bond and the high potent compound a hypothetical zero-order bond. An MMP was converted into a pseudo-molecule using an in-house Python script with the RDKit API. 

## 3.4. Machine learning
To find best approach for AC prediction, 4 fingerprints-based approach which were support vector machine (SVM), extra tree gradient boosting (XGboost), random forest, and fully connected neural network (FCNN), and 1 graph-based approach which was message passing neural network (MPNN) were used. For FCNN and MPNN, 2 distinct models which differ at molecular representation for the input were build. As control calculations, nearest neighbor (1NN) and 5NN were also compared.

### 3.4.1 SVM

### 3.4.2 XGboost

### 3.4.3 Random forest

### 3.4.4 FCNN with single input

### 3.4.5 FCNN with multiple input 

### 3.4.6 MPNN with single input

### 3.4.7 MPNN with maltiple input


## 3.5. Prediction Scheme

random split, axv split
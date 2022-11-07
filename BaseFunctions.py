# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 14:09:59 2020

@author: Tamura
"""

#%%
import pandas as pd
import numpy as np
import random
from BaseClass                         import Initialize, MultipleAXVSplit, MultipleTrainTestSplit
from sklearn.metrics import pairwise_distances
import json

def ToJson(obj, path):
    
    f = open(path, 'w')
    json.dump(obj,f)
    
def LoadJson(path):
    f = open(path, 'r')
    obj = json.load(f)

    return obj
    
class Base_wodirection(Initialize):
    
    def __init__(self, modeltype, dir_log=None, dir_score=None, data_split_metric='trtssplit'):
        
        super().__init__(modeltype=modeltype, dir_log=dir_log, dir_score=dir_score)
        
        self.trtssplit    = data_split_metric
        self.col          = ["core", "sub1", "sub2", "overlap"]


    def _SetParams(self, target):

        self.nbits_c = FindBitLength(self.ecfp, [self.col[0]])
        self.nbits_s = FindBitLength(self.ecfp, self.col[1:] )
        
        if self.trtssplit == 'axv':
            # Leave One Core Out
            all_seeds = pd.read_csv('./Dataset/Stats/axv.tsv', sep='\t', index_col=0).index
            seeds = [int(i.split('-Seed')[1]) for i in all_seeds if target == i.split('-Seed')[0]]
            self.data_split_generator = MultipleAXVSplit(self.main, seeds=seeds)
            self.testsetidx           = self.data_split_generator.keys()
            self.predictable          = True
        
        elif self.trtssplit == 'trtssplit':
            # Stratified Shuffled split
            if self._IsPredictableSet():
                self.data_split_generator = MultipleTrainTestSplit(self.main, n_dataset=3)
                self.testsetidx           = self.data_split_generator.keys()
                self.predictable          = True
            else:
                self.predictable          = False

        
    def _ReadDataFile(self, target):
   
        self.main = pd.read_csv("./Dataset/Data/%s.tsv" %target, sep="\t", index_col=0) # Single class MMSs were removed in curation.py.
        self.ecfp = pd.read_csv("./Dataset/ECFP/%s.tsv" %target, sep="\t", index_col=0)


    def _GetMatrices(self, cid, separated_input=False, unbiased=False):
        '''
        Wrapper function of TrainTestSplit_main, TrainTestSplit_ecfp, Hash2Bits 
        '''
        
        if self.trtssplit == 'trtssplit':
            tr, ts = self._TrainTestSplit_main(cid)
            
            if unbiased:
                tr = self._MakeTrainUnbiased(tr, seed=cid)
                
            df_trX, df_tsX = self._TrainTestSplit_ecfp(tr, ts)
            
            if not separated_input:
                df_trY, df_tsY, trX, trY, tsX, tsY = self._Hash2Bits(tr, ts, df_trX, df_tsX)
                
            elif separated_input:
                df_trY, df_tsY, trX, trY, tsX, tsY = self._Hash2Bits_separated(tr, ts, df_trX, df_tsX)

            return tr, ts, df_trX, df_trY, df_tsX, df_tsY, trX, trY, tsX, tsY
        
        elif self.trtssplit == 'axv':
            tr, cpdout, bothout = self._TrainTestSplit_main_axv(cid)
            
            if unbiased:
                tr = self._MakeTrainUnbiased(tr, seed=cid)
                
            df_trX, df_cpdoutX, df_bothoutX = self._TrainTestSplit_ecfp_axv(tr, cpdout, bothout)
            
            if not separated_input:
                df_trY, df_cpdoutY, df_bothoutY, trX, trY, cpdoutX, cpdoutY, bothoutX, bothoutY = self._Hash2Bits_axv(tr, cpdout, bothout, df_trX, df_cpdoutX, df_bothoutX)
                
            elif separated_input:
                df_trY, df_cpdoutY, df_bothoutY, trX, trY, cpdoutX, cpdoutY, bothoutX, bothoutY = self._Hash2Bits_separated_axv(tr, cpdout, bothout, df_trX, df_cpdoutX, df_bothoutX)
            
            return tr, cpdout, bothout, df_trX, df_trY, df_cpdoutX, df_cpdoutY, df_bothoutX, df_bothoutY, trX, trY, cpdoutX, cpdoutY, bothoutX, bothoutY
    
        
    def _TrainTestSplit_main(self, cid):
        
        generator = self.data_split_generator[cid]
        tr        = self.main.loc[generator.tridx,:]
        ts        = self.main.loc[generator.tsidx,:]
            
        return tr, ts
    
    
    def _TrainTestSplit_ecfp(self, tr, ts):
        
        df_trX = self.ecfp.loc[tr.index, :]
        df_tsX = self.ecfp.loc[ts.index, :]
        
        return df_trX, df_tsX
    
    def _MakeTrainUnbiased(self, tr, seed):
        
        ac    = tr[tr['class']==1]
        nonac = tr[tr['class']!=1]
        nac   = ac.shape[0]
        
        # random sample so that #non-ac equals to #ac 
        random.seed(seed)
        idx   = random.sample(range(nonac.shape[0]), nac)
        nonac = nonac.iloc[idx]
        
        tr = pd.concat([ac, nonac])
        
        return tr

    def _Hash2Bits(self, tr, ts, df_trX, df_tsX):
        
        df_trY = tr["class"]
        df_tsY = ts["class"]

        forward  = Hash2Bits(subdiff=False, sub_reverse=False)
        trX, trY = forward.GetMMPfingerprints_DF_unfold(df=df_trX, cols=self.col, Y=df_trY, nbits=[self.nbits_c, self.nbits_s], overlap="concat")
        tsX, tsY = forward.GetMMPfingerprints_DF_unfold(df=df_tsX, cols=self.col, Y=df_tsY, nbits=[self.nbits_c, self.nbits_s], overlap="concat")

        return df_trY, df_tsY, trX, trY, tsX, tsY
    
    def _Hash2Bits_separated(self, tr, ts, df_trX, df_tsX):
        
        df_trY = tr["class"]
        df_tsY = ts["class"]

        forward  = Hash2Bits(subdiff=False, sub_reverse=False)
        trX, trY = forward.GetSeparatedfingerprints_DF_unfold(df=df_trX, cols=self.col, Y=df_trY, nbits=[self.nbits_c, self.nbits_s])
        tsX, tsY = forward.GetSeparatedfingerprints_DF_unfold(df=df_tsX, cols=self.col, Y=df_tsY, nbits=[self.nbits_c, self.nbits_s])

        return df_trY, df_tsY, trX, trY, tsX, tsY
    
    def _TrainTestSplit_main_axv(self, cid):
        
        generator = self.data_split_generator[cid]
        tr        = self.main.loc[generator.tridx,        :]
        cpdout    = self.main.loc[generator.compound_out, :]
        bothout   = self.main.loc[generator.both_out,     :]        
            
        return tr, cpdout, bothout
    
    
    def _TrainTestSplit_ecfp_axv(self, tr, cpdout, bothout):
        
        df_trX      = self.ecfp.loc[tr.index     , :]
        df_cpdoutX  = self.ecfp.loc[cpdout.index , :]
        df_bothoutX = self.ecfp.loc[bothout.index, :]
        
        return df_trX, df_cpdoutX, df_bothoutX
    
    
    def _Hash2Bits_axv(self, tr, cpdout, bothout, df_trX, df_cpdoutX, df_bothoutX):
        
        df_trY      = tr["class"]
        df_cpdoutY  = cpdout["class"]
        df_bothoutY = bothout["class"]

        forward            = Hash2Bits(subdiff=False, sub_reverse=False)
        trX     , trY      = forward.GetMMPfingerprints_DF_unfold(df=df_trX, cols=self.col, Y=df_trY, nbits=[self.nbits_c, self.nbits_s], overlap="concat")
        cpdoutX , cpdoutY  = forward.GetMMPfingerprints_DF_unfold(df=df_cpdoutX , cols=self.col, Y=df_cpdoutY , nbits=[self.nbits_c, self.nbits_s], overlap="concat")
        bothoutX, bothoutY = forward.GetMMPfingerprints_DF_unfold(df=df_bothoutX, cols=self.col, Y=df_bothoutY, nbits=[self.nbits_c, self.nbits_s], overlap="concat")

        return df_trY, df_cpdoutY, df_bothoutY, trX, trY, cpdoutX, cpdoutY, bothoutX, bothoutY
    
    
    def _Hash2Bits_separated_axv(self, tr, cpdout, bothout, df_trX, df_cpdoutX, df_bothoutX):
        
        df_trY      = tr["class"]
        df_cpdoutY  = cpdout["class"]
        df_bothoutY = bothout["class"]

        forward            = Hash2Bits(subdiff=False, sub_reverse=False)
        trX     , trY      = forward.GetSeparatedfingerprints_DF_unfold(df=df_trX, cols=self.col, Y=df_trY, nbits=[self.nbits_c, self.nbits_s])
        cpdoutX , cpdoutY  = forward.GetSeparatedfingerprints_DF_unfold(df=df_cpdoutX , cols=self.col, Y=df_cpdoutY , nbits=[self.nbits_c, self.nbits_s])
        bothoutX, bothoutY = forward.GetSeparatedfingerprints_DF_unfold(df=df_bothoutX, cols=self.col, Y=df_bothoutY, nbits=[self.nbits_c, self.nbits_s])

        return df_trY, df_cpdoutY, df_bothoutY, trX, trY, cpdoutX, cpdoutY, bothoutX, bothoutY
    
    def _GetMatrices_3parts(self, cid):
        '''
        Wrapper function of TrainTestSplit_main, TrainTestSplit_ecfp, Hash2Bits 
        '''
        tr, ts = self._TrainTestSplit_main(cid)
        
        df_trX, df_tsX = self._TrainTestSplit_ecfp(tr, ts)
        
        df_trY, df_tsY, trX, trY, tsX, tsY = self._Hash2Bits(tr, ts, df_trX, df_tsX)

        return tr, ts, df_trX, df_trY, df_tsX, df_tsY, trX, trY, tsX, tsY
    
    
    def _Hash2Bits_3parts(self, tr, ts, df_trX, df_tsX):
        
        df_trY = tr["class"]
        df_tsY = ts["class"]

        forward  = Hash2Bits(subdiff=False, sub_reverse=False)
        trX, trY = forward.GetSeparatedfingerprints_DF_unfold(df=df_trX, cols=self.col, Y=df_trY, nbits=[self.nbits_c, self.nbits_s], overlap="concat")
        tsX, tsY = forward.GetSeparatedfingerprints_DF_unfold(df=df_tsX, cols=self.col, Y=df_tsY, nbits=[self.nbits_c, self.nbits_s], overlap="concat")

        return df_trY, df_tsY, trX, trY, tsX, tsY


class Hash2Bits():
    
    def __init__(self, subdiff=True, sub_reverse=False):

        self.subdiff = subdiff
        self.sub_rev = sub_reverse

    def GetMMPfingerprints_DF(self, X, Y=None, nbits=4096):

        c, s1, s2 = X

        if isinstance(nbits, list):
            nbits_c,  nbits_s = nbits
        else:
            nbits_c = nbits_s = nbits

        Xcore = GenerateFpsArray( c, nbits_c)
        Xsub1 = GenerateFpsArray(s1, nbits_s)
        Xsub2 = GenerateFpsArray(s2, nbits_s)
        
        if self.subdiff:
            Xsub1, Xsub2 = SetOffCommonBit(Xsub1, Xsub2)
        
        if not self.sub_rev:
            Xsub = np.hstack((Xsub1, Xsub2))
        else:
            Xsub = np.hstack((Xsub2, Xsub1))
        
        flag       = np.full(Xcore.shape[0], True, dtype=bool)
        zero       = np.where(np.sum(Xsub, axis=1)==0)[0]
        flag[zero] = False

        Xcore      = Xcore[flag, :]
        Xsub       = Xsub[flag, :]
        
        X = np.hstack([Xcore, Xsub])

        if Y is not None:
            Y = Y.values.reshape(-1)
            Y = Y[flag]

            return X, Y, flag

        else:
            return X, flag


    def GetMMPfingerprints_DF_SubOnly(self, X, Y=None, nbits=4096):

        s1, s2 = X

        nbits_s = nbits

        Xsub1 = GenerateFpsArray(s1, nbits_s)
        Xsub2 = GenerateFpsArray(s2, nbits_s)
        
        if self.subdiff:
            Xsub1, Xsub2 = SetOffCommonBit(Xsub1, Xsub2)
        
        if not self.sub_rev:
            Xsub = np.hstack((Xsub1, Xsub2))
        else:
            Xsub = np.hstack((Xsub2, Xsub1))
        
        flag       = np.full(Xsub.shape[0], True, dtype=bool)
        zero       = np.where(np.sum(Xsub, axis=1)==0)[0]
        flag[zero] = False
        
        Xsub       = Xsub[flag, :]
        
        X = Xsub

        if Y is not None:
            Y = Y.values.reshape(-1)
            Y = Y[flag]

            return X, Y, flag

        else:
            return X, flag


    def GetMMPfingerprints_DF_unfold(self, df, cols, Y=None, nbits=None, overlap="delete"):
        """
        - Generate MMPfingerprints binary vectors from str hash in given pd.DataFrame
        - This is for unfolded fingerprints

        Args:
            df (pd.DataFrame)       : A dataframe stores str hash values
            cols (list)             : column names used for the fingerprints generation
            Y (array-like, optional): given Y values will become one dimentional vector. Defaults to None.
            nbits (int)             : This func needs specifing nbits. Find bit length with FindBitLength func before using.
            overlap (delete/concat) : An option to specify how overlaping sub features are treated.
                                         

        Note:
            Substructure difference should be applyed in data curation phase.
            This function do not do this operation

        Returns:
            X: fingerprints binary matrix for ML input
        """        
        
        if not (overlap == "delete") and not (overlap == "concat"):
            raise ValueError("Overlap option should be delete/concat, not %s." %overlap)

        print("    $ Overlap option is selected as %s" %overlap)

        c  = df[cols[0]]
        s1 = df[cols[1]]
        s2 = df[cols[2]]

        Xcore = GenerateFpsArray( c, nbits[0])
        Xsub1 = GenerateFpsArray(s1, nbits[1])
        Xsub2 = GenerateFpsArray(s2, nbits[1])
        
        Xsub  = (Xsub1 + Xsub2).astype(bool).astype(int)

        X = np.hstack([Xcore, Xsub])

        if overlap == "concat":
            o = df[cols[3]]
            Xoverlap = GenerateFpsArray(o, nbits[1])

            X = np.hstack([X, Xoverlap])

        if Y is not None:
            Y = Y.values.reshape(-1)

            return X, Y

        else:
            return X
        
        
    def GetSeparatedfingerprints_DF_unfold(self, df, cols, Y=None, nbits=None):
        """
        - Generate MMPfingerprints binary vectors for 3 parts from str hash in given pd.DataFrame
        - This is for unfolded fingerprints

        Args:
            df (pd.DataFrame)       : A dataframe stores str hash values
            cols (list)             : column names used for the fingerprints generation
            Y (array-like, optional): given Y values will become one dimentional vector. Defaults to None.
            nbits (int)             : This func needs specifing nbits. Find bit length with FindBitLength func before using.
                                         

        Note:
            Substructure difference should be applied in data curation phase.
            This function do not do this operation

        Returns:
            X: fingerprints binary matrix for ML input
        """        

        c  = df[cols[0]]
        s1 = df[cols[1]]
        s2 = df[cols[2]]

        Xcore = GenerateFpsArray( c, nbits[0])
        Xsub1 = GenerateFpsArray(s1, nbits[1])
        Xsub2 = GenerateFpsArray(s2, nbits[1])
        
        o = df[cols[3]]
        Xoverlap = GenerateFpsArray(o, nbits[1])

        Xsub1 = Xsub1 + Xoverlap
        Xsub2 = Xsub2 + Xoverlap

        Y = Y.values.reshape(-1)

        return {'core':Xcore, 'sub1':Xsub1, 'sub2':Xsub2}, Y


    

def FindBitLength(df, cols):

    df = df[cols]

    hashes_all = []

    for col in cols:
        df[col] = df[col].apply(lambda x:[int(h) for h in x.split(" ")] if isinstance(x, str) else [-1])
        hashes_all += df[col].sum()

    return np.max(hashes_all) + 1


def RestoreFPfromstr(fp, bits):
    
    if isinstance(fp, str):
        bit = [int(i) for i in fp.split(" ") if len(i) != 0]

    elif fp is None:
        bit = []
    
    elif np.isnan(fp):
        bit = []

    else:
        bit = [int(fp)]

    fp_vector  = np.zeros((bits), dtype=int)     
    for i in bit :
        fp_vector[i] = 1
    
    return fp_vector



def GenerateFpsArray(sr_fp, nbits):
    
    fps = np.array([RestoreFPfromstr(str_fp,bits=nbits) for str_fp in sr_fp])

    return fps



def SetOffCommonBit(Xsub1, Xsub2):
    
    Xsub = Xsub1 + Xsub2
    
    commonbit = np.where(Xsub>1)
    
    for row, col in zip(commonbit[0], commonbit[1]):
        
        Xsub1[row, col] = 0
        Xsub2[row, col] = 0
        
    return Xsub1, Xsub2

def funcTanimotoKernel_MMPKernel(x,y,len_c, weight=None):
    """
    - Tanimoto Kernel for svm
    - if you add weights to each similarity, you asign numbers to the argument "weight" with 1D seq.
    - the order of weights must be weight for core then that for sub.
    """
    x_core    = np.hsplit(x,[len_c])[0]
    x_subdiff = np.hsplit(x,[len_c])[1]
    
    y_core    = np.hsplit(y,[len_c])[0]
    y_subdiff = np.hsplit(y,[len_c])[1]
    
    core = funcTanimotoSklearn(x_core,y_core)
    sub  = funcTanimotoSklearn(x_subdiff,y_subdiff)
    
    if weight is None:
        return core * sub
    
    else:
        return weight[0]*core + weight[1]*sub
    
def funcTanimotoSklearn(x, y):
    """
    Jaccard similarity is used as a kernel (boolean transformation was conducted)
    """
    if 0:
        print('----')
        print(x.shape)
        print(y.shape)
        print('----')

    if (x.ndim == 1) and (y.ndim ==1):
        jdist = pairwise_distances(x.astype(bool, copy=False).reshape(1,-1), y.astype(bool, copy=False).reshape(1,-1), metric='jaccard')
    else:
        jdist = pairwise_distances(x.astype(bool, copy=False), y.astype(bool, copy=False), metric='jaccard')
    
    return 1 - jdist

import functools
from sklearn                 import svm
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_X_y
from sklearn                 import svm
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import GridSearchCV as gcv
from sklearn import preprocessing as pp
from sklearn.svm import SVC

class MMPKernelSVM_sklearn(BaseEstimator, ClassifierMixin):
    """
    - This is SVM with MMPKernel following sklearn's rule.
    - Sklearn BaseEstimator and ClassifierMixin are inherited.
    - Enabling to use this func with GridSearchCV is ongoing... 

    Args:
        BaseEstimator ([type]): [description]
        ClassifierMixin ([type]): [description]
    """    
    def __init__(self, C=1.0, len_c=4096, subonly=False, weight_kernel=False, decision_func=False, cv=False, nfold=5):
         
        self.len_c         = len_c
        self.subonly       = subonly
        self.decision_func = decision_func
        self.weight_kernel = weight_kernel
        self.cv            = cv

        if not self.cv:
            self.C             = C
        else:
            self.nfold = nfold
            print("    $Hyperpameter optimization: True")
            print("    $n_fold: %dfold\n" %self.nfold)
        


    def _set_kernel_func(self):
        if not self.subonly:
            kernel_func = functools.partial(funcTanimotoKernel_MMPKernel, len_c=self.len_c)
            
            if self.weight_kernel:
                kernel_func = functools.partial(kernel_func, weight=self.weight_kernel)

        else:
            kernel_func = funcTanimotoSklearn

        return kernel_func

    def _set_model(self, params=None):
        
        self.Kernel = self._set_kernel_func()

        if self.cv:
            # model = CSVM_CV(nf=self.nfold, paramset=params, selectedScore=make_scorer(matthews_corrcoef), kernelf="precomputed",
            #                 is_scaling=False, pos_label=1, neg_label=-1, precomputedkf=self.Kernel)
            model = CSVM_CV(nf=self.nfold, paramset=params, selectedScore='roc_auc', kernelf="precomputed",
                            is_scaling=False, pos_label=1, neg_label=-1, precomputedkf=self.Kernel)

        else:
            model = svm.SVC(C=self.C, class_weight="balanced", kernel=self.Kernel)

        return model


    def fit(self, X, y=None):

        if X.shape[0]==0:
            raise ValueError("No training were given.")
        
        if np.unique(y).shape[0]==1:
            raise ValueError("Given training has only one class.")

        X, y = check_X_y(X, y)

        if self.cv:
            optimizer = self._set_model()
            optimizer.fit(X, y)
            self.svc_ = optimizer.pmodel
        else:
            self.svc_   = self._set_model()
            self.svc_.fit(X, y)
            
        return self

    def predict(self, X):
        
        if X.shape[0]==0: 
            raise ValueError("No test data. it sometimes happen beacause of substructure differense operation.")
                
        predY = self.svc_.predict(X)
        
        if self.decision_func:
            self.dec_func_ = self.svc_.decision_function(X)
            
        return predY
    
    def get_params(self):
        return self.svc_.get_params()

class CSVM_CV():
    """
    cross validation version. wrapper class of Nu SVM

    """
    
    def __init__(self, rseed=0, nf=5, paramset=None, verbose=False, kernelf='rbf', is_scaling=False, selectedScore=None, pos_label='Active', neg_label='Decoy', precomputedkf=None):
        self.nf         = nf
        self.verbose    = verbose
        self.rng        = np.random.RandomState(rseed)
        
        self.metric, self.kernelf, self.paramset, self.is_scaling = self._set_conditions(selectedScore, kernelf, paramset, is_scaling, precomputedkf)

        if self.is_scaling: # if Tanimoto kernel is used this part is skipped 
            self.xscaler = pp.MinMaxScaler(feature_range=(-1, 1))
        
        self.pos_label = pos_label
        self.neg_label = neg_label

        self.model = self._set_model(self.kernelf, self.paramset, self.rng, self.nf, self.metric)    
        

    def _set_conditions(self, selectedScore, kernelf, paramset, is_scaling, precomputedkf):
        """
        Setting calculation conditions 
        """
        if selectedScore is None:
            metric = 'roc_auc'
        else:
            metric = selectedScore
        
        kernelf = kernelf.lower()
        if kernelf not in ['tanimoto', 'rbf', 'linear', "precomputed"]:
            ValueError('Kernel must be either RBF and Tanimoto')
            exit(1)

        if paramset is None:
            if kernelf == 'rbf':
                paramset = dict(gamma=np.logspace(-4, 2, num=5), C=np.logspace(-2, 2, num=5, endpoint=True, base=10))
            else:
                paramset = dict(C=np.logspace(-2, 2, num=10, endpoint=True, base=10))
    
        if kernelf == 'tanimoto':
            print('skip the scaling due to binary kernel')
            is_scaling = False

        if kernelf == "precomputed":
            self.precomputedkf = precomputedkf
            
        return metric, kernelf, paramset, is_scaling

    def _set_model(self, kernelf, paramset, rng, nf, selectedScore):
        """
        Setting the models with parameters
        """
            
        if kernelf == 'tanimoto':
            return gcv(SVC(kernel=funcTanimotoSklearn, random_state=0, class_weight='balanced'), param_grid=paramset,
                        cv=StratifiedKFold(n_splits=nf, shuffle=True, random_state=rng), 
                        scoring=selectedScore, n_jobs=-1)
        elif kernelf == 'rbf':
            return gcv(SVC(random_state=0, class_weight='balanced'), param_grid=paramset, 
                        cv=StratifiedKFold(n_splits=nf, shuffle=True, random_state=rng),
                        scoring=selectedScore, n_jobs=-1)
        elif kernelf == 'linear':
            return gcv(SVC(random_state=0, class_weight='balanced', kernel='linear'), param_grid=paramset, 
                        cv=StratifiedKFold(n_splits=nf, shuffle=True, random_state=rng),
                        scoring=selectedScore, n_jobs=-1)
        elif kernelf == 'precomputed':
            return gcv(SVC(kernel=self.precomputedkf, random_state=0, class_weight='balanced'), param_grid=paramset,
                        cv=StratifiedKFold(n_splits=nf, shuffle=True, random_state=rng), 
                        scoring=selectedScore, n_jobs=10)

    def fit(self, x, y, weights=None):
        """
        Fit the cv model with the x and y
        """
        if isinstance(y, pd.Series):
            y = y.values.reshape(-1,1).copy()

        if self.is_scaling:
            xs = self.xscaler.fit_transform(x)
        else:
            xs = x
        
#        y = self.label2category(y)
        
        if weights is None:
            self.model.fit(xs, y.ravel())
        else:
            self.model.fit(xs, y.ravel(), sample_weight=weights)

        # optimized parameters
        self.params = dict(kernelf=self.kernelf, C=self.model.best_estimator_.C)
        if self.kernelf == 'rbf':
            self.params['gamma'] = self.model.best_estimator_.gamma
            self.pmodel = SVC(C=self.params['C'], gamma=self.params['gamma'], class_weight='balanced')
            
        elif self.kernelf == 'tanimoto':
            self.pmodel = SVC(kernel=funcTanimotoSklearn, C=self.params['C'], class_weight='balanced')
        
        elif self.kernelf =='linear':
            self.pmodel = SVC(C=self.params['C'], class_weight='balanced', kernel='linear')
        
        elif self.kernelf == 'precomputed':
            self.pmodel = SVC(kernel=self.precomputedkf, C=self.params['C'], class_weight='balanced')

        # prediction by the best model
        self.pmodel.fit(xs, y.ravel())
        self.params.update(dict(nSVs_each=self.pmodel.n_support_, nSVs_all=len(self.pmodel.support_vectors_)))
        
        yptr = self.pmodel.predict(xs)
        Yptr = self.category2label(yptr)
        Yscore = self.pmodel.decision_function(xs)
        
#        self.evalscore = MakeAllStatistics(y.ravel(), Yscore, yp_label_org=Yptr, pos_label=self.pos_label, neg_label=self.neg_label)

    def label2category(self, y):
        sy = y.copy()
        sy[sy==self.pos_label] = 1
        sy[sy==self.neg_label] = -1
        sy = sy.astype(int)    
        return sy

    def category2label(self, y):
        sy = y.copy()
        if isinstance(self.pos_label,str):
            sy = sy.astype(object)
        sy[sy==1] = self.pos_label
        sy[sy==-1] = self.neg_label
        return sy

    def get_params(self):
        return self.params

    def predict(self, x):
        """
        Predict y values for x

        Note: there is no parallellization here (apply_with_batch function)
        """
        x = np.array(x)
        
        if self.is_scaling:
            sx = self.xscaler.transform(x)
        else:
            sx = x
        
        py = self.pmodel.predict(sx)

        py = self.category2label(py)
        return py

    def score(self, x):
        """
        Return the decision function values for positive and negatives
        """
        x = np.array(x)
        
        if self.is_scaling:
            sx = self.xscaler.transform(x)
        else:
            sx = x
        
        return self.pmodel.decision_function(sx)


    def predict_vals(self, xtrain, xtest):
        """
        predict multiple y values for mulple xs 
        """
        py1 = self.predict(xtrain)
        py2 = self.predict(xtest)
        return py1, py2
    
    
import optuna
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer
from sklearn.metrics import matthews_corrcoef as mcc
class RandomForest_cls_CV():
    
    """wrapper class of RF classifier based on sklearn.randomforest_classifier and optuna
    """    
    
    def __init__(self, ntrials=100, nfold=3, scoref='roc_auc', earlystop=True):
        
        self.nfold     = nfold
        self.ntrials   = ntrials
        self.earlystop = earlystop
        self.name_scoref, self.scoref = self._set_scoref(scoref)

    def _set_scoref(self, str_scoref):
        
        if str_scoref.lower() == 'mcc':
            return 'mcc', make_scorer(mcc) #self._MCC 
        else:
            return str_scoref, str_scoref
         
    def _MCC(self, preds, dtrain):
        '''
        Matthew's correlation coefficient func to be available in XGboost
        see details in:
        https://www.bitsdiscover.com/python-using-matthews-correlation-coefficient-mcc-as-evaluation-metric-in-xgboost/
        '''
        THRESHOLD = 0.5
        labels = dtrain.get_label()
        return 'mcc', mcc(labels, preds >= THRESHOLD)

    def _objective_sklearn(self, trial):        
        param = {
                'n_estimators'     : trial.suggest_int('n_estimators', 50, 1000),
                'max_depth'        : trial.suggest_int('max_depth', 4, 50),
                'min_samples_split': trial.suggest_float("min_samples_split", 1e-8, 1.0, log=True),
                'class_weight'     : 'balanced',
                }
        
        kfold = StratifiedKFold(n_splits=self.nfold, shuffle=True, random_state=42)
        ml    = RandomForestClassifier(**param)
        score_all = cross_val_score(ml, self.trX, self.trY, cv=kfold, scoring='roc_auc')
        score = score_all.mean()
        
        return score
    
    def fit(self, trX, trY):
        
        self.trX = trX
        self.trY = trY
        
        #change negative label to 0
        self.trY[np.where(trY==-1)[0]] = 0

        pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
        study = optuna.create_study(pruner=pruner, direction='maximize')
        study.optimize(self._objective_sklearn, n_trials=self.ntrials)
        
        self.rf_ = RandomForestClassifier(**study.best_params)
        self.rf_.fit(trX, trY)
        
    def predict(self, tsX):
        predY = self.rf_.predict(tsX)
        predY[np.where(predY==0)[0]] = -1
        return predY

    def score(self, tsX):
        prob = self.rf_.predict_proba(tsX)
        return prob

    def get_params(self):
        return self.rf_.get_params()
    
    
import xgboost as xgb
import optuna


class XGBoost_CV():
    
    """wrapper class of XGBoost classifier based on xgboost and optuna
       This is same as Optuna tutorial (https://github.com/optuna/optuna-examples/blob/main/xgboost/xgboost_simple.py).
       Scoring function is set to AUCROC
       MCC is not available because it is not supported in XGBoost early stopping callback
    """    
    
    def __init__(self, ntrials=100, nfold=3, scoref='auc', earlystop=True):
        
        self.nfold     = nfold
        self.ntrials   = ntrials
        self.earlystop = earlystop
        self.name_scoref, self.scoref = self._set_scoref(scoref)

    def _set_scoref(self, str_scoref):
        
        if str_scoref.lower() == 'mcc':
            return 'mcc', make_scorer(mcc) #self._MCC 
        else:
            return str_scoref, str_scoref
         
    def _MCC(self, preds, dtrain):
        '''
        Matthew's correlation coefficient func to be available in XGboost
        see details in:
        https://www.bitsdiscover.com/python-using-matthews-correlation-coefficient-mcc-as-evaluation-metric-in-xgboost/
        '''
        THRESHOLD = 0.5
        labels = dtrain.get_label()
        return 'mcc', mcc(labels, preds >= THRESHOLD)

    def _objective_sklearn(self, trial):        
        param = {
                "verbosity"             : 0,
                "use_label_encoder"     : False,
                "objective"             : "binary:logistic",
                # use exact for small dataset.
                "tree_method"           : "exact",
                # defines booster, gblinear for linear functions.
                "booster"               : "gbtree",
                # L2 regularization weight.
                "reg_lambda"            : trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
                #L1 regularization weight.
                "reg_alpha"             : trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
                # sampling ratio for training data.
                "subsample"             : trial.suggest_float("subsample", 0.2, 1.0),
                # sampling according to each tree.
                "colsample_bytree"      : trial.suggest_float("colsample_bytree", 0.2, 1.0),
                # maximum depth of the tree, signifies complexity of the tree.
                "max_depth"             : trial.suggest_int("max_depth", 3, 9, step=2),
                # minimum child weight, larger the term more conservative the tree.
                "min_child_weight"      : trial.suggest_int("min_child_weight", 2, 10),
                "eta"                   : trial.suggest_float("eta", 1e-8, 1.0, log=True),
                # defines how selective algorithm is.
                "gamma"                 : trial.suggest_float("gamma", 1e-8, 1.0, log=True),
                # weight for inbalanced data
                'scale_pos_weight'      : np.where(self.trY==0)[0].shape[0]/np.where(self.trY==1)[0].shape[0], # n_neg / n_pos
                # early stopping settings
                "eval_metric"           : self.scoref,
                'early_stopping_rounds' : 15 
                }
        
        kfold = StratifiedKFold(n_splits=self.nfold, shuffle=True, random_state=42)
        ml    = xgb.XGBClassifier(**param)
        score_all = cross_val_score(ml, self.trX, self.trY, cv=kfold, scoring=param['eval_metric'])
        score = score_all.mean()
    
        trial.report(score, 1)
    
        if trial.should_prune():
            print('    $ step%d is pruned')
            raise optuna.TrialPruned()
        
        return score

    def _objective(self, trial):        
        param = {
                "verbosity"             : 0,
                "use_label_encoder"     : False,
                "objective"             : "binary:logistic",
                # use exact for small dataset.
                "tree_method"           : "exact",
                # defines booster, gblinear for linear functions.
                "booster"               : "gbtree",
                # L2 regularization weight.
                "lambda"                : trial.suggest_float("lambda", 1e-8, 1.0, log=True),
                #L1 regularization weight.
                "alpha"                 : trial.suggest_float("alpha", 1e-8, 1.0, log=True),
                # sampling ratio for training data.
                "subsample"             : trial.suggest_float("subsample", 0.2, 1.0),
                # sampling according to each tree.
                "colsample_bytree"      : trial.suggest_float("colsample_bytree", 0.2, 1.0),
                # maximum depth of the tree, signifies complexity of the tree.
                "max_depth"             : trial.suggest_int("max_depth", 3, 9, step=2),
                # minimum child weight, larger the term more conservative the tree.
                "min_child_weight"      : trial.suggest_int("min_child_weight", 2, 10),
                "eta"                   : trial.suggest_float("eta", 1e-8, 1.0, log=True),
                # defines how selective algorithm is.
                "gamma"                 : trial.suggest_float("gamma", 1e-8, 1.0, log=True),
                # weight for inbalanced data
                'scale_pos_weight'      : np.where(self.trY==0)[0].shape[0]/np.where(self.trY==1)[0].shape[0], # n_neg / n_pos
                "eval_metric"           : self.scoref,
                }
        
        if self.earlystop:
            self.pruning_callback = [optuna.integration.XGBoostPruningCallback(trial, "test-%s"%self.name_scoref)]
        else:
            self.pruning_callback = [None]
          
        kfold   = StratifiedKFold(n_splits=self.nfold, shuffle=True, random_state=0)
        dtrain  = xgb.DMatrix(self.trX, label=self.trY)
        history = xgb.cv(param, dtrain, num_boost_round=1000, folds=kfold, early_stopping_rounds=15, callbacks=self.pruning_callback)

        mean_auc = history["test-auc-mean"].values[-1]
        return mean_auc
    
    def fit(self, trX, trY):
        
        self.trX = trX
        self.trY = trY
        
        #change negative label to 0
        self.trY[np.where(trY==-1)[0]] = 0

        pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
        study = optuna.create_study(pruner=pruner, direction='maximize')
        study.optimize(self._objective, n_trials=self.ntrials)
        
        self.xgb_ = xgb.XGBClassifier(**study.best_params)
        self.xgb_.fit(trX, trY)
        
    def predict(self, tsX):
        predY = self.xgb_.predict(tsX)
        predY[np.where(predY==0)[0]] = -1
        return predY

    def score(self, tsX):
        prob = self.xgb_.predict_proba(tsX)
        return prob

    def get_params(self):
        return self.xgb_.get_params()
    
import matplotlib.pyplot as plt
import seaborn

# global variables in this module
a4_dims = (11.7, 8.27) # inches
a4_square = (11.7, 11.7)
a3_dims = (16.54, 11.7)
margin  = 0.15 # bottom margin

def MakeBoxPlotsSeaborn(table, xname, yname, hue=None, fscale=2.0, save_fig_name=None, title='', 
                        xlab_name=None, ylab_name=None, yrange=None, legend_list=None, palette='husl',
                        manual_label=None, dpi=200, use_swarm=False, show_legend=True, rotate_x=False, hue_order=None, rotate_x_deg=30,
                        fsize=(11.7, 8.27), font=None, font_scale=1, context='paper', **kwargs):
    """
    Set the seaborn context and make a box plot
    """
    seaborn.set(font=font, font_scale=font_scale, context=context, style='whitegrid', palette=palette)
    
    margin  = 0.15 # bottom margin
    fig, ax = plt.subplots(figsize=fsize)
    if use_swarm:
        ax      = seaborn.swarmplot(x=xname, y=yname, hue=hue, data=table, ax=ax, size=fscale*5, **kwargs)
    else:
        ax      = seaborn.boxplot(x=xname, y=yname, hue=hue, data=table, ax=ax, hue_order=hue_order, **kwargs)
    if xlab_name is None:
        xlab_name = xname
    if ylab_name is None:
        ylab_name = yname
    
    seaborn.utils.axlabel(xlabel=xlab_name, ylabel=ylab_name)
    if yrange is not None:
        ax.set_ylim(yrange)
    ax.set_title(title)
    handles, _ = ax.get_legend_handles_labels()
    if legend_list is not None:
        ax.legend(handles, legend_list)
    if show_legend:
        ax.legend(bbox_to_anchor=(1.15, 1), borderaxespad=0.)
    else:
        ax.legend_.remove()
    
    #ax.grid(False)
    #ax.get_xaxis().set_visible(False)
    #ax.get_yaxis().set_visible(False)
    if manual_label is not None:
        plt.xticks(plt.xticks()[0], manual_label)
    if rotate_x:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=rotate_x_deg)

    if save_fig_name is not None:
        plt.savefig(save_fig_name, bbox_inches='tight', dpi=dpi)  
        plt.close() 
    else:
        return ax
    
def makeHistogramSeaborn(pd_table, xname, hue_col, save_fig_name=None, label_name=None,
                         legend=False, title="", font=None, font_scale=1, context='paper', **kwargs):
    seaborn.set_theme(font=font, font_scale=font_scale, context=context, style='whitegrid')
    fig, ax = plt.subplots(figsize=a4_square)
    #g = seaborn.FacetGrid(pd_table, hue=hue_col)
    if hue_col is not None:
        hue_list = pd_table[hue_col].unique().tolist()
        
        for category in hue_list:
            target_data = pd_table[pd_table[hue_col] == category]
            ax = seaborn.histplot(target_data[xname], ax=ax, **kwargs)
    else:
        ax = seaborn.histplot(pd_table[xname], ax=ax, **kwargs) 
    
    if label_name is not None:
        ax.set_xlabel(label_name)

    if legend:
        fig.legend(labels=hue_list)

    ax.xaxis.labelpad = 25
    ax.yaxis.labelpad = 25
    
    if save_fig_name is not None:
        plt.savefig(save_fig_name)  
        plt.close() 
    else:
        return ax
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 14:09:59 2020

@author: Tamura
"""

#%%
from operator import neg
import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
#import shap
from tqdm                              import tqdm
from SVM.svmwrappers                   import MMPKernelSVM_sklearn   as svm
from SVM.svrwrappers                   import MMPKernelSVR_sklearn   as svr
from SVM.ocsvmwrappers                 import MMPKernelOCSVM_sklearn as ocsvm
from RandomForest.RandomForestClassifier import RandomForestClassifier_CV as rf
from SVM.svmwrappers                   import MakeInputDict
from MMP.make_input                    import LeaveOneCoreOut, GetDiverseCore, DelTestCpdFromTrain, MultipleTrainTestSplit
from Tools.ReadWrite                   import ReadDataAndFingerprints, ToPickle, LoadPickle
from Tools.utils                       import BasicInfo
#from Metrix.scores                     import ScoreTable,RenderDF2Fig
from Evaluation.Labelling              import AddLabel
from Evaluation.Score                  import ScoreTable_wDefinedLabel
from collections                       import OrderedDict, defaultdict
from Plots.Barplot                     import MakeBarPlotsSeaborn as bar
from Plots.Lineplot                    import MakeLinePlotsSeaborn as line
from Fingerprint.Hash2BitManager       import Hash2Bits, FindBitLength
from functools                         import partial
#from svmviz.svmviz.mmpviz              import feature_contributions
from svmviz.svmviz.svmviz              import feature_contributions as fc_original
from sklearn.svm                       import SVC
from Kernels.Kernel                    import funcTanimotoSklearn
#from MKLpy.algorithms                  import EasyMKL, AverageMKL
from Kernels.Kernel                    import funcTanimotoSklearn
from collections                       import defaultdict
from sklearnex                         import patch_sklearn
from GradientBoost.XGBoost             import XGBoost_CV as xgb

patch_sklearn()

class Base_wodirection():
    
    def __init__(self, modeltype, model_name, dir_log=None, dir_score=None, interpreter=None, aconly=False, kernel_type="product", data_split_metric='LOCO'):
        #self.bd      = "/home/tamura/work/Interpretability"
        #os.chdir(self.bd)
        self.mtype        = modeltype
        self.mname        = model_name.lower()
        self.logdir, self.scoredir, self.modeldir = self._MakeLogDir(dir_log, dir_score)
        self.debug        = False
        self.interpreter  = interpreter
        self.aconly       = aconly
        self.kernel_type  = kernel_type
        self.trtssplit    = data_split_metric
        self.col  = ["core", "sub1", "sub2", "overlap"]
        
        print("$  Decision function will be applied.\n")    

    def _MakeLogDir(self, logdir, scoredir):
        
        if logdir is None:
            logdir   = "./Results/"
        
        if scoredir is None:
            scoredir = "./Scores/"
        
        modeldir = os.path.join(logdir, "Models")

        os.makedirs(logdir  , exist_ok=True)
        os.makedirs(scoredir, exist_ok=True)
        os.makedirs(modeldir, exist_ok=True)
        
        return logdir, scoredir, modeldir


    def _SetML(self, model_name, nu=0.05, kernel_type="product"):
    
        if model_name == "svm":

            if kernel_type == "product":
                print('    $ svm with MMPkernel is used.')
                weight  = False
            else: 
                print('    $ svm with weighted MMPkernel is used.')
                weight = kernel_type

            model = svm(len_c=self.nbits_c, decision_func=True, weight_kernel=weight, cv=True)

        elif model_name == "random_forest":
            print('    $ random forest is used.')
            model = rf(njobs=-1, pos_label=1, neg_label=-1)

        elif model_name == "ocsvm":
            print('    $ ocsvm with MMPkernel is used.')
            model = ocsvm(len_c=self.nbits_c, decision_func=True, nu=nu)
            
        elif model_name == 'xgboost':
            print('    $ XGboost is used.')
            model = xgb()
            
        else:
            NotImplementedError("%s has not been implemented. Please put the obj directly." %model_name)

        return model


    def _CalcFeatureImportance(self, model_name, ml, trdata, tsdata, method, link="identity", proportion="average"):
        
        if method == "shap":

            if model_name == "svm":
                shap_kernel = shap.KernelExplainer(
                                                    model = ml.svc_.decision_function,
                                                    data  = trdata,
                                                    link  = link)

                fcs = shap_kernel.shap_values(tsdata, nsamples=self.nsample)

            elif model_name == "random_forest":
                shap_kernel = shap.TreeExplainer(
                                                    model = ml.pmodel,
                                                    data  = trdata)

                fcs = shap_kernel.shap_values(tsdata)[1]

            elif model_name == "ocsvm":
                shap_kernel = shap.KernelExplainer(
                                                    model = ml.ocsvm_.decision_function,
                                                    data  = trdata,
                                                    link  = link)   

                fcs = shap_kernel.shap_values(tsdata, nsamples=self.nsample)                                  
                    
            else:
                NotImplementedError("%s has not been implemented. Please put the obj directly." %model_name)

        elif method == "tanimoto":
            
            svector = trdata[ml.svc_.support_,:]
            lambda_y = ml.svc_.dual_coef_

            for i, row in tqdm(enumerate(tsdata)):
                fc = feature_contributions(row, svector, lambda_y.ravel(), self.nbits_cf, proportion=proportion)

                if i == 0:
                    fcs = fc
                else:
                    fcs = np.vstack([fcs, fc])
            
        return fcs


    def _SetParams(self):

        self.nbits_c = FindBitLength(self.ecfp, [self.col[0]])
        self.nbits_s = FindBitLength(self.ecfp, self.col[1:] )
        
        if self.trtssplit == 'LOCO':
            # Leave One Core Out
            self.data_split_generator = LeaveOneCoreOut(self.main)
            self.testsetidx           = self.data_split_generator.keys()
        
        elif self.trtssplit == 'trtssplit':
            # Stratified Shuffled split
            self.data_split_generator = MultipleTrainTestSplit(self.main, n_dataset=3)
            self.testsetidx           = self.data_split_generator.keys()

        # Leave One Core Out
        if self.debug: 
            self.nsample = 1

        else:
            self.nsample = 50


    def _ReadDataFile(self, target, acbits=False):

        if acbits:
            main = pd.read_csv("./Dataset_ACbits/Data/%s.tsv" %target, sep="\t", index_col=0) # Single class MMSs were removed in curation.py.
            ecfp = pd.read_csv("./Dataset_ACbits/ECFP/%s.tsv" %target, sep="\t", index_col=0)
        else:    
            main = pd.read_csv("./Dataset/Data/%s.tsv" %target, sep="\t", index_col=0) # Single class MMSs were removed in curation.py.
            ecfp = pd.read_csv("./Dataset/ECFP/%s.tsv" %target, sep="\t", index_col=0)

        return main, ecfp


    def _GetMatrices(self, cid, aconly=False):

        generator = self.data_split_generator[cid]
        tr        = self.main.loc[generator.tridx,:]
        ts        = self.main.loc[generator.tsidx,:]
        
        # Check overlap
        tr = DelTestCpdFromTrain(ts, tr, deltype="both")
        
        # Assign pd_sr of fp; [core, sub1, sub2]
        if aconly:
            print(    "Only AC-MMPs are used for making training data.\n")
            tr = tr[tr["class"]==1]

        df_trX = self.ecfp.loc[tr.index, :]
        df_tsX = self.ecfp.loc[ts.index, :]
        df_trY = tr["class"]
        df_tsY = ts["class"]

        forward  = Hash2Bits(subdiff=False, sub_reverse=False)
        trX, trY = forward.GetMMPfingerprints_DF_unfold(df=df_trX, cols=self.col, Y=df_trY, nbits=[self.nbits_c, self.nbits_s], overlap="concat")
        tsX, tsY = forward.GetMMPfingerprints_DF_unfold(df=df_tsX, cols=self.col, Y=df_tsY, nbits=[self.nbits_c, self.nbits_s], overlap="concat")

        return tr, ts, df_trX, df_trY, df_tsX, df_tsY, trX, trY, tsX, tsY
        
    
    def run(self, target, debug=False, kernel_type=False, onlyfccalc=False, proportion="average"):
        
        print("\n----- %s is proceeding -----\n" %target)
        path_log = os.path.join(self.logdir, "%s_%s.tsv" %(target, self.mtype))
        
        if debug:
            self.debug=True
        
        self.main, self.ecfp = self._ReadDataFile(target, acbits=self.aconly)
        self._SetParams()

        if onlyfccalc:
            self._CalcFCs(target, path_log, proportion)
        
        else:

            if kernel_type == "simpleTanimoto":
                print("    $ %s is conducted\n" %kernel_type)
                self._AllMMSPred_SimpleTanimoto(target, path_log)
                
            elif kernel_type == "MKL":
                print("    $ %s is conducted\n" %kernel_type)
                self._AllMMSPred_MKL(target, path_log)
                
            else:
                self._AllMMSPred(target, path_log)

    
    def GetScore(self, t):

        scorer = ScoreTable_wDefinedLabel(pred_type=self.pred_type)

        path_log = os.path.join(self.logdir, "%s_%s.tsv" %(t, self.mtype))
        log      = pd.read_csv(path_log, sep="\t", index_col=0)
        df       = scorer.GetTable({t:log})

        if self.kernel_type==False:
            path_score = os.path.join(self.scoredir, "%s_%s.tsv" %(t, self.mtype))
        else:   
            path_score = os.path.join(self.scoredir, "%s_%s_%d_%d.tsv" %(t, self.mtype, self.kernel_type[0], self.kernel_type[1]))
        df.to_csv(path_score, sep="\t")
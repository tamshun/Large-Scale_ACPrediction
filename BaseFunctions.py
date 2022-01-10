# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 14:09:59 2020

@author: Tamura
"""

#%%
import pandas as pd
import numpy as np
import os
#import shap
from MMP.make_input                    import LeaveOneCoreOut, GetDiverseCore, DelTestCpdFromTrain, MultipleTrainTestSplit
from Tools.ReadWrite                   import ToJson
from Fingerprint.Hash2BitManager       import Hash2Bits, FindBitLength
from collections                       import defaultdict
from sklearnex                         import patch_sklearn

patch_sklearn()

class Base_wodirection():
    
    def __init__(self, modeltype, dir_log=None, dir_score=None, aconly=False, data_split_metric='LOCO'):
        #self.bd      = "/home/tamura/work/Interpretability"
        #os.chdir(self.bd)
        self.mtype        = modeltype
        self.debug        = False        
        self.aconly       = aconly
        self.trtssplit    = data_split_metric
        self.col          = ["core", "sub1", "sub2", "overlap"]
        self.logdir, self.scoredir, self.modeldir = self._MakeLogDir(dir_log, dir_score)  

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


    def _SetParams(self):

        self.nbits_c = FindBitLength(self.ecfp, [self.col[0]])
        self.nbits_s = FindBitLength(self.ecfp, self.col[1:] )
        
        if self.trtssplit == 'LOCO':
            # Leave One Core Out
            self.data_split_generator = LeaveOneCoreOut(self.main)
            self.testsetidx           = self.data_split_generator.keys()
            self.del_leak             = True
        
        elif self.trtssplit == 'trtssplit':
            # Stratified Shuffled split
            self.data_split_generator = MultipleTrainTestSplit(self.main, n_dataset=3)
            self.testsetidx           = self.data_split_generator.keys()
            self.del_leak             = False

        
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
        
        # Check leak cpds
        if self.del_leak:
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
        
    
    def run(self, target, debug=False, onlyfccalc=False):
        
        print("\n----- %s is proceeding -----\n" %target)
        path_log = os.path.join(self.logdir, "%s_%s.tsv" %(target, self.mtype))
        
        if debug:
            self.debug=True
        
        self.main, self.ecfp = self._ReadDataFile(target, acbits=self.aconly)
        self._SetParams()
        self._AllMMSPred(target, path_log)
            
                
    def _SetML(self):
        
        '''
        A model object is needed to run this code.
        You should set a model you use in your main class that inherits this class.
        '''
        pass
    
    
    def _AllMMSPred(self, target, path_log):
        
        '''
        main scripts
        '''
        pass

    
# class BaseWithFC(Base_wodirection):  
    
#     def __init__(self, interpreter):
        
#         super().__init__()
#         self.interpreter  = interpreter
        
#         # Leave One Core Out
#         if self.debug: 
#             self.nsample = 1

#         else:
#             self.nsample = 50
        
#     def _CalcFeatureImportance(self, model_name, ml, trdata, tsdata, method, link="identity", proportion="average"):
        
#         if method == "shap":

#             if model_name == "svm":
#                 shap_kernel = shap.KernelExplainer(
#                                                     model = ml.svc_.decision_function,
#                                                     data  = trdata,
#                                                     link  = link)

#                 fcs = shap_kernel.shap_values(tsdata, nsamples=self.nsample)

#             elif model_name == "random_forest":
#                 shap_kernel = shap.TreeExplainer(
#                                                     model = ml.pmodel,
#                                                     data  = trdata)

#                 fcs = shap_kernel.shap_values(tsdata)[1]

#             elif model_name == "ocsvm":
#                 shap_kernel = shap.KernelExplainer(
#                                                     model = ml.ocsvm_.decision_function,
#                                                     data  = trdata,
#                                                     link  = link)   

#                 fcs = shap_kernel.shap_values(tsdata, nsamples=self.nsample)                                  
                    
#             else:
#                 NotImplementedError("%s has not been implemented. Please put the obj directly." %model_name)

#         elif method == "tanimoto":
            
#             svector = trdata[ml.svc_.support_,:]
#             lambda_y = ml.svc_.dual_coef_

#             for i, row in tqdm(enumerate(tsdata)):
#                 fc = feature_contributions(row, svector, lambda_y.ravel(), self.nbits_cf, proportion=proportion)

#                 if i == 0:
#                     fcs = fc
#                 else:
#                     fcs = np.vstack([fcs, fc])
            
#         return fcs
    
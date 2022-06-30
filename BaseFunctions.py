# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 14:09:59 2020

@author: Tamura
"""

#%%
import pandas as pd
import numpy as np
import os
import joblib
import random
from MMP.make_input                    import LeaveOneCoreOut, GetDiverseCore, DelTestCpdFromTrain, MultipleTrainTestSplit
from Tools.ReadWrite                   import ToJson
from Fingerprint.Hash2BitManager       import Hash2Bits, FindBitLength
from collections                       import defaultdict

class AXV_generator():
    
    def __init__(self, data, keep_out_rate=0.2, seed=0, cols=['chembl_cid1', 'chembl_cid2']) -> None:
        self.data = data
        self.rate = keep_out_rate
        self.seed = seed
        self.cols = cols
        self.whole_cpds = np.union1d(np.unique(data[cols[0]]),np.unique(data[cols[1]])).tolist()
        
        self.keepout = self._get_keepout()
        self.identifier = self._set_identifier()
        
         
    def _get_keepout(self):
        random.seed(self.seed)
        sample_size = int(len(self.whole_cpds)*self.rate)
        keep_out = random.sample(self.whole_cpds, sample_size)
        
        return keep_out
    
    def _identifier(self, mmp:pd.Series):
    
        cpd1 = mmp[self.cols[0]]
        cpd2 = mmp[self.cols[1]]
        
        isin_cpd1 = cpd1 in self.keepout
        isin_cpd2 = cpd2 in self.keepout
        
        if (isin_cpd1==False) and (isin_cpd2==False):
            return 0
        
        elif (isin_cpd1==True) and (isin_cpd2==True):
            return 2
        
        elif (isin_cpd1==True) or (isin_cpd2==True):
            return 1
        
    def _set_identifier(self):
        return [self._identifier(sr) for i, sr in self.data.iterrows()]
    
    def get_subset(self, name):
        
        if name.lower() == 'train':   
            mask = [True if i==0 else False for i in self.identifier]
            
        elif name.lower() == 'compound_out':
            mask = [True if i==1 else False for i in self.identifier]
            
        elif name.lower() == 'both_out':
            mask = [True if i==2 else False for i in self.identifier]
            
        return self.data.loc[mask, :]    
    
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
            self.predictable          = True
        
        elif self.trtssplit == 'trtssplit':
            # Stratified Shuffled split
            if self._IsPredictableSet():
                self.data_split_generator = MultipleTrainTestSplit(self.main, n_dataset=3)
                self.testsetidx           = self.data_split_generator.keys()
                self.del_leak             = False
                self.predictable          = True
            else:
                self.predictable          = False

        
    def _ReadDataFile(self, target, acbits=False):

        if acbits:
            main = pd.read_csv("./Dataset_ACbits/Data/%s.tsv" %target, sep="\t", index_col=0) # Single class MMSs were removed in curation.py.
            ecfp = pd.read_csv("./Dataset_ACbits/ECFP/%s.tsv" %target, sep="\t", index_col=0)
        else:    
            main = pd.read_csv("./Dataset/Data/%s.tsv" %target, sep="\t", index_col=0) # Single class MMSs were removed in curation.py.
            ecfp = pd.read_csv("./Dataset/ECFP/%s.tsv" %target, sep="\t", index_col=0)

        return main, ecfp


    def _GetMatrices(self, cid, aconly=False):
        '''
        Wrapper function of TrainTestSplit_main, TrainTestSplit_ecfp, Hash2Bits 
        '''
        tr, ts = self._TrainTestSplit_main(cid, aconly)
        
        df_trX, df_tsX = self._TrainTestSplit_ecfp(tr, ts)
        
        df_trY, df_tsY, trX, trY, tsX, tsY = self._Hash2Bits(tr, ts, df_trX, df_tsX)

        return tr, ts, df_trX, df_trY, df_tsX, df_tsY, trX, trY, tsX, tsY
        
    
    def _TrainTestSplit_main(self, cid, aconly):
        
        generator = self.data_split_generator[cid]
        tr        = self.main.loc[generator.tridx,:]
        ts        = self.main.loc[generator.tsidx,:]
        
        # Check leak cpds
        if self.del_leak:
            tr = DelTestCpdFromTrain(ts, tr, id1='chembl_cid1', id2='chembl_cid2', deltype="both", biased_check=False)
        
        # Assign pd_sr of fp; [core, sub1, sub2]
        if aconly:
            print(    "Only AC-MMPs are used for making training data.\n")
            tr = tr[tr["class"]==1]
            
        return tr, ts
    
    def _TrainTestSplit_ecfp(self, tr, ts):
        
        df_trX = self.ecfp.loc[tr.index, :]
        df_tsX = self.ecfp.loc[ts.index, :]
        
        return df_trX, df_tsX
    
    def _Hash2Bits(self, tr, ts, df_trX, df_tsX):
        
        df_trY = tr["class"]
        df_tsY = ts["class"]

        forward  = Hash2Bits(subdiff=False, sub_reverse=False)
        trX, trY = forward.GetMMPfingerprints_DF_unfold(df=df_trX, cols=self.col, Y=df_trY, nbits=[self.nbits_c, self.nbits_s], overlap="concat")
        tsX, tsY = forward.GetMMPfingerprints_DF_unfold(df=df_tsX, cols=self.col, Y=df_tsY, nbits=[self.nbits_c, self.nbits_s], overlap="concat")

        return df_trY, df_tsY, trX, trY, tsX, tsY
    
    
    def _GetMatrices_3parts(self, cid, aconly=False):
        '''
        Wrapper function of TrainTestSplit_main, TrainTestSplit_ecfp, Hash2Bits 
        '''
        tr, ts = self._TrainTestSplit_main(cid, aconly)
        
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
    
    
    def run(self, target, debug=False, onlyfccalc=False):
        
        print("\n----- %s is proceeding -----\n" %target)
        
        if debug:
            self.debug=True
        
        self.main, self.ecfp = self._ReadDataFile(target, acbits=self.aconly)
        
        self._SetParams()
        
        if self._IsPredictableSet():
            self._AllMMSPred(target)
            
        else:
            print('    $ %s is skipped because of lack of the actives' %target)
        
        
    def run_parallel(self, target_list, njob=-1):
        result = joblib.Parallel(n_jobs=njob)(joblib.delayed(self.run)(target) for target in target_list)
        
        
    
    def _IsPredictableTarget(self):
        data = self.main
        
        bool_nsample = bool(data.shape[0] > 50)
        bool_nmms    = bool(np.unique(data['core_id']).shape[0] > 1)
        flag         = bool(bool_nsample * bool_nmms)
        
        if flag:
            gbr         = data.groupby(['core_id', 'class'])
            n_mms       = gbr.count().index.max()[0] + 1
            n_class     = gbr.count().index.shape[0]
            bool_nclass = bool((n_class - n_mms) >= 2)
            
            flag = bool(flag * bool_nclass)
        
        return flag
    
    
    def _IsPredictableSeries(self, tr, ts, min_npos):
        '''
        analyse if given series is predictable or not.
        
        Requirement
        - tr/ts have at least one data
        - tr has positive sample at least the number of cv so that each cv set has at least one positive sample  
        '''
        n_tr = bool(tr.shape[0] > 0)
        n_ts = bool(ts.shape[0] > 0) 
        n_pos_tr = bool(tr[tr['class']==1].shape[0] >= min_npos)
        
        return bool(n_tr * n_ts * n_pos_tr)         
    
    def _IsPredictableSet(self):
        bool_nac = np.where(self.main['class']==1)[0].shape[0] > 1
        return bool_nac
                    
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
    
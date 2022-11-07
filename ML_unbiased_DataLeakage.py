# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 14:09:59 2020

@author: Tamura
"""

import pandas as pd
import numpy as np
import os
import platform
from SVM.svmwrappers                     import MMPKernelSVM_sklearn   as svm
from RandomForest.RandomForestClassifier import RandomForest_cls_CV as rf
from GradientBoost.XGBoost               import XGBoost_CV as xgb
from Tools.ReadWrite                     import ReadDataAndFingerprints, ToPickle, ToJson
from collections                         import defaultdict
from sklearnex                           import patch_sklearn
from BaseFunctions                       import Base_wodirection
from sklearn.neighbors                   import KNeighborsClassifier
from Kernels.Kernel                      import funcTanimotoKernel_MMPKernel
from functools                           import partial

patch_sklearn()

class Classification(Base_wodirection):

    def __init__(self, modeltype, model, dir_log, dir_score, data_split_metric='trtssplit'):

        super().__init__(modeltype, dir_log=dir_log, dir_score=dir_score, data_split_metric=data_split_metric)
        self.model_name = model
        self.pred_type  = "classification"
        self.nfold      = 3
    
    
    def _SetML(self, kernel_type="product"):
        
        model_name = self.model_name.lower()
        
        if model_name == "svm":

            if kernel_type == "product":
                print('    $ svm with MMPkernel is used.')
                weight  = False
            else: 
                print('    $ svm with weighted MMPkernel is used.')
                weight = kernel_type

            model = svm(len_c=self.nbits_c, decision_func=True, weight_kernel=weight, cv=True, nfold=self.nfold)

        elif model_name == "random_forest":
            print('    $ random forest is used.')
            model = rf(nfold=self.nfold)
            
        elif model_name == 'xgboost':
            print('    $ XGboost is used.')
            model = xgb(nfold=self.nfold)
            
            
        else:
            NotImplementedError("%s has not been implemented. Please put the obj directly." %model_name)

        return model
    
            
    def _AllMMSPred(self, target):
        
        if self._IsPredictableTarget():    

            for trial in self.testsetidx:    
                
                
                if self.debug:
                    if trial>2:
                        break
                
                # Train test split
                print("    $ Prediction for cid%d is going on.\n" %trial)
                tr, ts, df_trX, df_trY, df_tsX, df_tsY, trX, trY, tsX, tsY = self._GetMatrices(trial, unbiased=True)
                
                if self._IsPredictableSeries(tr, ts, min_npos=self.nfold):
                    # Fit and Predict
                    ml = self._SetML()           
                    ml.fit(trX, trY)
                    predY = ml.predict(tsX)
                    print("    $ Prediction Done.\n")

                    # Write & save log
                    log = self._WriteLog(ml, trial, tr, ts, tsX, tsY, predY)           
                    self._Save(target, trial, log, ml)
                    print("    $  Log is out.\n")
                else:
                    print('    $ cid%d is not predictable' %trial)
                
            
            
    def _WriteLog(self, ml, cid, tr, ts, tsX, tsY, predY):
        
        log = defaultdict(list)
        
        # Write log
        tsids            = ts["id"].tolist()
        log["ids"]      += tsids
        log["cid"]      += [cid] * len(tsids)
        log['#tr']      += [tr.shape[0]] * len(tsids)
        log['#ac_tr']   += [tr[tr['class']==1].shape[0]] * len(tsids)
        log["trueY"]    += tsY.tolist()
        log["predY"]    += predY.tolist()

        mname = self.model_name.lower()
        if mname == "svm":
            log["prob"] += ml.svc_.decision_function(tsX).tolist()

        elif mname in ['xgboost', "random_forest"]:
            log["prob"] += [prob[1] for prob in ml.score(tsX).tolist()]
            
        else:
            raise NotImplementedError('%s is not available so far' %self.mname)
        
        return log
        
    def _Save(self, target, trial, log, ml):
        
        path_log = os.path.join(self.logdir, "%s_trial%d.tsv" %(target, trial))
        self.log = pd.DataFrame.from_dict(log)
        self.log.to_csv(path_log, sep="\t")

        params = {key: val for key, val in ml.get_params().items() if key!='kernel'}
        ToJson(params, self.modeldir+"/%s_trial%d.json"%(target, trial))
        

def main(bd, model):
    
    mtype = "unbiased_trtssplit"
    
    os.chdir(bd)
    os.makedirs("./Log_%s"%mtype  , exist_ok=True)
    os.makedirs("./Score_%s"%mtype, exist_ok=True)
    
    # top10 targets having lots of AC
    tlist = ['CHEMBL244', 'CHEMBL204', 'CHEMBL205', 'CHEMBL3594', 'CHEMBL261', 'CHEMBL264', 'CHEMBL3242', 'CHEMBL253', 'CHEMBL217', 'CHEMBL3837']
    
    p = Classification(modeltype   = mtype,
                       model       = model,
                       dir_log     = "./Log_%s/%s" %(mtype, model),
                       dir_score   = "./Score_%s/%s" %(mtype, model),
                       )
    
    print(' $ %s is selected as machine learning method'%model)    
    p.run_parallel(tlist, njob=-1)

def debug(bd, model):
    
    mtype = "unbiased_trtssplit"
    
    os.chdir(bd)
    os.makedirs("./Log_%s"%mtype  , exist_ok=True)
    os.makedirs("./Score_%s"%mtype, exist_ok=True)
    
    mtype += '_debug'
    
    p = Classification(modeltype   = mtype,
                       model       = model,
                       dir_log     = "./Log_%s/%s" %(mtype, model),
                       dir_score   = "./Score_%s/%s" %(mtype, model),
                       )
    p.run('CHEMBL204', debug=True)
    
if __name__ == '__main__':
    
    if platform.system() == 'Darwin':
        bd    = "/Users/tamura/work/ACPredCompare/"
    else:
        bd    = "/home/tamuras0/work/ACPredCompare/"
        
    #model = "Random_Forest"
    
    # debug(bd, model='SVM')
    main(bd, model='SVM')

    
 
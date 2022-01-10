# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 14:09:59 2020

@author: Tamura
"""

#%%
import pandas as pd
import numpy as np
import os
import platform
from SVM.svmwrappers                     import MMPKernelSVM_sklearn   as svm
from RandomForest.RandomForestClassifier import RandomForestClassifier_CV as rf
from GradientBoost.XGBoost               import XGBoost_CV as xgb
from Tools.ReadWrite                     import ReadDataAndFingerprints, ToPickle, ToJson
from collections                         import defaultdict
from sklearnex                           import patch_sklearn
from .BaseFunctions                      import Base_wodirection

patch_sklearn()

class Classification(Base_wodirection):

    def __init__(self, modeltype, model, dir_log, dir_score, data_split_metric='LOCO'):

        super().__init__(modeltype, dir_log=dir_log, dir_score=dir_score, data_split_metric=data_split_metric)
        self.model_name = model
        self.pred_type = "classification"
    
    
    def _SetML(self, kernel_type="product"):
        
        model_name = self.model_name.lower()
        
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
            
        elif model_name == 'xgboost':
            print('    $ XGboost is used.')
            model = xgb()
            
        else:
            NotImplementedError("%s has not been implemented. Please put the obj directly." %model_name)

        return model
        
        
    def _AllMMSPred(self, target, path_log):
            
        # Initialize
        log = defaultdict(list) #MakeLogDict(type=self.pred_type)

        for cid in self.testsetidx:    
            
            if self.debug:
                if cid>2:
                    break
            
            # Train test split
            print("    $ Prediction for cid%d is going on.\n" %cid)
            tr, ts, df_trX, df_trY, df_tsX, df_tsY, trX, trY, tsX, tsY = self._GetMatrices(cid)
            
            # Fit and Predict
            ml = self._SetML()           
            ml.fit(trX, trY)
            predY = ml.predict(tsX)
            print("    $ Prediction Done.\n")

            # Write & save log
            log = self._WriteLog(log, ml, cid, ts, tsX, tsY, predY)           
            self._Save(target, cid, log, ml)
            print("    $  Log is out.\n")
            
            
    def _WriteLog(self, log, ml, cid, ts, tsX, tsY, predY):
        
        # Write log
        tsids            = ts["id"].tolist()
        log["ids"]      += tsids
        log["cid"]      += [cid] * len(tsids)
        log["trueY"]    += tsY.tolist()
        log["predY"]    += predY.tolist()

        if self.mname == "svm":
            log["prob"] += ml.dec_func_.tolist()

        elif self.mname == "random_forest":
            log["prob"] += [prob[1] for prob in ml.score(tsX).tolist()]
            
        elif self.mname == 'xgboost':
            log["prob"] += [prob[1] for prob in ml.score(tsX).tolist()]
        
        else:
            raise NotImplementedError('%s is not available so far' %self.mname)
        
        
    def _Save(self, target, cid, log, ml):
        path_log = os.path.join(self.logdir, "%s_trial%d.tsv" %(target, cid))
        self.log = pd.DataFrame.from_dict(log)
        self.log.to_csv(path_log, sep="\t")

        params = {key: val for key, val in ml.get_params().items() if key!='kernel'}
        ToJson(params, self.modeldir+"/%s_trial%d.json"%(target, cid))
        
        
if __name__ == '__main__':
    
    if platform.system() == 'Darwin':
        bd    = "/Users/tamura/work/ACPredCompare/"
    else:
        bd    = "/home/tamuras0/work/ACPredCompare/"
        
    model = "XGBoost"
    mtype = "wodirection"
    os.chdir(bd)
    os.makedirs("./Log"  , exist_ok=True)
    os.makedirs("./Score", exist_ok=True)
    
    tlist = pd.read_csv('./Dataset/target_list.tsv', sep='\t', index_col=0)
    
    for i, sr in tlist.iterrows():
        
        target = sr['target']
        
        p = Classification(modeltype   = mtype,
                           model       = model,
                           dir_log     = "./Log_%s/%s" %(mtype, model),
                           dir_score   = "./Score_%s/%s" %(mtype, model)
                           )

        p.run(target=target, debug=False)
 
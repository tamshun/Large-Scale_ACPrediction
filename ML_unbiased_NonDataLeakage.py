# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 14:09:59 2020

@author: Tamura
"""

import pandas as pd
import numpy as np
import os
import platform
from collections     import defaultdict
from sklearnex       import patch_sklearn
from BaseFunctions   import MMPKernelSVM_sklearn   as svm
from BaseFunctions   import RandomForest_cls_CV as rf
from BaseFunctions   import XGBoost_CV as xgb
from BaseFunctions   import Base_wodirection, ToJson

patch_sklearn()

class Classification(Base_wodirection):

    def __init__(self, modeltype, model, dir_log, dir_score, data_split_metric='axv'):

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
                tr, cpdout, bothout, df_trX, df_trY, df_cpdoutX, df_cpdoutY, df_bothoutX, df_bothoutY, trX, trY, cpdoutX, cpdoutY, bothoutX, bothoutY = self._GetMatrices(trial, unbiased=True)
                
                flag_predictable = self._IsPredictableSeries(tr, cpdout, min_npos=self.nfold) * self._IsPredictableSeries(tr, bothout, min_npos=self.nfold)
                if flag_predictable:
                    # Fit and Predict
                    ml = self._SetML()           
                    ml.fit(trX, trY)
                    predY_tr      = ml.predict(trX)
                    predY_cpdout  = ml.predict(cpdoutX) 
                    predY_bothout = ml.predict(bothoutX) 
                    print("    $ Prediction Done.\n")
                    
                    # Write & save log
                    log_tr      = self._WriteLog(ml, trial, tr, tr , trX , trY , predY_tr)           
                    log_cpdout  = self._WriteLog(ml, trial, tr, cpdout , cpdoutX , cpdoutY , predY_cpdout)           
                    log_bothout = self._WriteLog(ml, trial, tr, bothout, bothoutX, bothoutY, predY_bothout)           
            
                    self._Save(target, trial, log_tr, log_cpdout, log_bothout, ml)
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
        
    def _Save(self, target, trial, log_tr, log_cpdout, log_bothout, ml):
        
        path_tr = os.path.join(self.logdir, "%s_tr_trial%d.tsv" %(target, trial))
        self.log_tr = pd.DataFrame.from_dict(log_tr)
        self.log_tr.to_csv(path_tr, sep="\t")
        
        path_cpdout = os.path.join(self.logdir, "%s_cpdout_trial%d.tsv" %(target, trial))
        self.log_cpdout = pd.DataFrame.from_dict(log_cpdout)
        self.log_cpdout.to_csv(path_cpdout, sep="\t")
        
        path_bothout = os.path.join(self.logdir, "%s_bothout_trial%d.tsv" %(target, trial))
        self.log_bothout = pd.DataFrame.from_dict(log_bothout)
        self.log_bothout.to_csv(path_bothout, sep="\t")

        params = {key: val for key, val in ml.get_params().items() if key!='kernel'}
        ToJson(params, self.modeldir+"/%s_trial%d.json"%(target, trial))
        

def main(bd, model):
    
    mtype = "unbiased_axv"
    
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
    
    mtype = "unbiased_axv"
    
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
    
 
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 14:09:59 2020

@author: Tamura
"""

import pandas as pd
import numpy as np
import os
import platform
from sklearn.neighbors                   import KNeighborsClassifier
from Tools.ReadWrite                     import ReadDataAndFingerprints, ToPickle, ToJson
from collections                         import defaultdict
from sklearnex                           import patch_sklearn
from BaseFunctions                       import Base_wodirection
from sklearn.neighbors                   import KNeighborsClassifier
from Kernels.Kernel                      import funcTanimotoKernel_MMPKernel
from functools                           import partial

patch_sklearn()

def distance_func(x, y, len_c):
    
    return 1 - funcTanimotoKernel_MMPKernel(x, y, len_c=len_c)

class Classification(Base_wodirection):

    def __init__(self, modeltype, model, dir_log, dir_score, data_split_metric='axv'):

        super().__init__(modeltype, dir_log=dir_log, dir_score=dir_score, data_split_metric=data_split_metric)
        self.model_name = model
        self.pred_type  = "classification"
        self.nfold      = 3
        
        if self.model_name == '1NN':
            self.dist = 1
        elif self.model_name == '5NN':
            self.dist = 5
        
        
    def _AllMMSPred(self, target):
        
        if self._IsPredictableTarget():    

            for trial in self.testsetidx:    
                
                if self.debug:
                    if trial>2:
                        break
                
                # Train test split
                print("    $ Prediction for cid%d is going on.\n" %trial)
                tr, cpdout, bothout, df_trX, df_trY, df_cpdoutX, df_cpdoutY, df_bothoutX, df_bothoutY, trX, trY, cpdoutX, cpdoutY, bothoutX, bothoutY = self._GetMatrices(trial)
                
                flag_predictable = self._IsPredictableSeries(tr, cpdout, min_npos=self.nfold) * self._IsPredictableSeries(tr, bothout, min_npos=self.nfold)
                if flag_predictable:
                    # Fit and Predict
                    dist_func = partial(distance_func, len_c=self.nbits_c)
                    ml = KNeighborsClassifier(n_neighbors=self.dist, metric=dist_func, n_jobs=-1)
                    ml.fit(trX, trY)
                    predY_cpdout  = ml.predict(cpdoutX) 
                    predY_bothout = ml.predict(bothoutX) 
                    log_cpdout  = self._WriteLog(ml, trial, tr, cpdout , cpdoutX , cpdoutY , predY_cpdout)           
                    log_bothout = self._WriteLog(ml, trial, tr, bothout, bothoutX, bothoutY, predY_bothout)           
                    self._Save(target, trial, log_cpdout, log_bothout)
                    print("    $ Prediction Done.\n")
                    
                    # Write & save log
                    #log_tr     = self._WriteLog(log_tr     , ml, trial, tr, tr , trX , trY , predY_tr)           
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
        log['prob']     += [p[1] for p in ml.predict_proba(tsX)]

        dist_neigh, idx_neigh = ml.kneighbors(tsX, return_distance=True)
        id_neigh   = ['; '.join([str(tr.index[i]) for i in idx]) for idx in idx_neigh]
        dist_neigh = ['; '.join(d.astype(str)) for d in dist_neigh] 
        
        log['neighbor'] += id_neigh
        log['distance'] += dist_neigh
        
        return log
        
    def _Save(self, target, trial, log_cpdout, log_bothout):
        
        # path_tr = os.path.join(self.logdir, "%s_tr_trial%d.tsv" %(target, trial))
        # self.log_tr = pd.DataFrame.from_dict(log_tr)
        # self.log_tr.to_csv(path_tr, sep="\t")
        
        path_cpdout = os.path.join(self.logdir, "%s_cpdout_trial%d.tsv" %(target, trial))
        self.log_cpdout = pd.DataFrame.from_dict(log_cpdout)
        self.log_cpdout.to_csv(path_cpdout, sep="\t")
        
        path_bothout = os.path.join(self.logdir, "%s_bothout_trial%d.tsv" %(target, trial))
        self.log_bothout = pd.DataFrame.from_dict(log_bothout)
        self.log_bothout.to_csv(path_bothout, sep="\t")
        

def main(bd, model, mtype):
    
    os.chdir(bd)
    os.makedirs("./Log_%s"%mtype  , exist_ok=True)
    os.makedirs("./Score_%s"%mtype, exist_ok=True)
    
    tlist = pd.read_csv('./Dataset/target_list.tsv', sep='\t', index_col=0)
    
    p = Classification(modeltype   = mtype,
                       model       = model,
                       dir_log     = "./Log_%s/%s" %(mtype, model),
                       dir_score   = "./Score_%s/%s" %(mtype, model),
                       )
    
    # for i, sr in tlist.iterrows():
        
    #     target = sr['chembl_tid']
    #     p.run(target=target, debug=True)
    
    print(' $ %s is selected as machine learning method'%model)    
    p.run_parallel(tlist['chembl_tid'], njob=-1)

def debug(bd, model, mtype):
    os.chdir(bd)
    os.makedirs("./Log_%s"%mtype  , exist_ok=True)
    os.makedirs("./Score_%s"%mtype, exist_ok=True)
    
    tlist = pd.read_csv('./Dataset/target_list.tsv', sep='\t', index_col=0)
    
    mtype += '_debug'
    
    p = Classification(modeltype   = mtype,
                       model       = model,
                       dir_log     = "./Log_%s/%s" %(mtype, model),
                       dir_score   = "./Score_%s/%s" %(mtype, model),
                       )
    p.run('CHEMBL1800', debug=True)
    
if __name__ == '__main__':
    
    if platform.system() == 'Darwin':
        bd    = "/Users/tamura/work/ACPredCompare/"
        
    elif platform.system() == 'Linux':
        bd    = "/home/tamura/work/ACPredCompare/"
        
    else:
        bd    = "/home/tamuras0/work/ACPredCompare/"
        
    #model = "Random_Forest"
    mtype = "axv"
    
    #debug(bd, model='1NN', mtype=mtype)
    main(bd, model='1NN', mtype=mtype)
    main(bd, model='5NN', mtype=mtype)
 
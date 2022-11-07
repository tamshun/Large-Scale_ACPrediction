# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 14:09:59 2020

@author: Tamura
"""

import pandas as pd
import numpy as np
import os
import platform
from Metrix.scores                     import CalcBasicScore
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,matthews_corrcoef, balanced_accuracy_score

class ScoreFuncsSklearn():
    
    def __init__(self):
        self.funcdict = dict(accuracy          = accuracy_score,
                             precision         = precision_score,
                             recall            = recall_score,
                             f1                = f1_score,
                             auc_roc           = roc_auc_score,
                             matthews_coeff    = matthews_corrcoef,
                             balanced_accuracy = balanced_accuracy_score,
                             )

class CalcBasicScore(ScoreFuncsSklearn):
    
    def __init__(self) -> None:
        super().__init__()
        scores = dict()
    
    def CalcScores(self, y_true, y_pred, y_score):
        d = dict()
        for metric_name in self.funcdict.keys():
            if metric_name != 'auc_roc':
                score = self.funcdict[metric_name](y_true=y_true, y_pred=y_pred)
            else:
                score = self.funcdict[metric_name](y_true=y_true, y_score=y_score)
                
            d[metric_name] = score
                
        return d
class MakeScoreTable(CalcBasicScore):
    
    def __init__(self, tlist, targets, model, modeltype, dir_log, dir_score):
        
        super().__init__()
        self.tlist   = tlist
        
        if targets!=None:
            self.targets = targets
        else:
            self.targets = np.unique([s.split('-')[0] for s in self.tlist.index])
            
        self.model     = model
        self.mtype     = modeltype
        self.dir_log   = dir_log
        self.dir_score = dir_score
        
        os.makedirs(self.dir_score, exist_ok=True)
        
    def _StoreStats(self, trials, target, score_mean, score_all):  
        mean   = trials.mean()
        var    = trials.var()
        var.index = [i+'_var' for i in var.index]
        score_mean[target] = pd.concat([mean, var])
        trials['target'] = target
        score_all = pd.concat([score_all, trials], ignore_index=True) 
        return  score_mean, score_all
    
    def _Save(self, score_mean, score_all, metric):
        score_mean = pd.DataFrame.from_dict(score_mean).T
        score_all  = pd.DataFrame.from_dict(score_all).T    
        
        score_mean = score_mean.applymap(lambda x: np.round(x,3))  
          
        score_mean.to_csv(self.dir_score + '/mean_%s.tsv' %metric, sep='\t')
        score_all.to_csv(self.dir_score + '/alltrial_%s.tsv' %metric , sep='\t')  
          
    def GetScores_TakeAverage(self):
        
        score_mean_cpdout  = dict()
        score_all_cpdout   = pd.DataFrame()
        score_mean_bothout = dict()
        score_all_bothout  = pd.DataFrame()
        
        for target in self.targets:
            trials_cpdout, trials_bothout = self._GetOneScore(target)
            score_mean_cpdout, score_all_cpdout = self._StoreStats(trials_cpdout, target, score_mean_cpdout, score_all_cpdout)
            score_mean_bothout, score_all_bothout = self._StoreStats(trials_bothout, target, score_mean_bothout, score_all_bothout)
        
        self._Save(score_mean_cpdout, score_all_cpdout, 'cpdout')
        self._Save(score_mean_bothout, score_all_bothout, 'bothout')
        
        print('$ score calculation done')        
        
    def _GetOneScore(self, target):
        
        d_cpdout = dict()
        d_bothout = dict()
        
        Seeds = [int(seed.split('-Seed')[1]) for seed in self.tlist.index if seed.split('-Seed')[0] == target]
        
        for i in Seeds:
            
            if self.model in ['FCNN', 'MPNN', 'FCNN_separated', 'MPNN_separated']:
                path_log_cpdout  = os.path.join(self.dir_log, "%s_Seed%d_cpdout.tsv" %(target, i))
                path_log_bothout = os.path.join(self.dir_log, "%s_Seed%d_bothout.tsv" %(target, i))
            elif 'NN' in self.model:
                path_log_cpdout  = os.path.join(self.dir_log, "%s_cpdout_trial%d.tsv" %(target, i))
                path_log_bothout = os.path.join(self.dir_log, "%s_bothout_trial%d.tsv" %(target, i))
            else:
                path_log_cpdout = os.path.join(self.dir_log, "%s_cpdout_trial%d.tsv" %(target, i))
                path_log_bothout = os.path.join(self.dir_log, "%s_bothout_trial%d.tsv" %(target, i))
                
            log_cpdout     = pd.read_csv(path_log_cpdout, sep="\t", index_col=0)
            score_cpdout    = self.CalcScores(log_cpdout['trueY'], log_cpdout['predY'], log_cpdout['prob'])
            score_cpdout['#tr']    = log_cpdout['#tr'].iloc[0]
            score_cpdout['#ac_tr'] = log_cpdout['#ac_tr'].iloc[0]
            d_cpdout[i] = score_cpdout
            
            log_bothout     = pd.read_csv(path_log_bothout, sep="\t", index_col=0)
            score_bothout    = self.CalcScores(log_bothout['trueY'], log_bothout['predY'], log_bothout['prob'])
            score_bothout['#tr']    = log_bothout['#tr'].iloc[0]
            score_bothout['#ac_tr'] = log_bothout['#ac_tr'].iloc[0]
            d_bothout[i] = score_bothout
        
        score_cpdout  = pd.DataFrame.from_dict(d_cpdout, orient='index')
        score_bothout = pd.DataFrame.from_dict(d_bothout, orient='index')
        
        return score_cpdout, score_bothout

if __name__ == '__main__':
    if platform.system() == 'Darwin':
        bd    = "/Users/tamura/work/ACPredCompare/"
    else:
        bd    = "/home/tamuras0/work/ACPredCompare/"
    
    tlist = pd.read_csv('./Dataset/Stats/axv.tsv', sep='\t', index_col=0)
    # top10 targets having lots of AC
    # targets = ['CHEMBL244', 'CHEMBL204', 'CHEMBL205', 'CHEMBL3594', 'CHEMBL261', 'CHEMBL264', 'CHEMBL3242', 'CHEMBL253', 'CHEMBL217', 'CHEMBL3837']
    
    for ml in ['FCNN', 'SVM', 'Random_Forest', 'XGBoost', 'FCNN_separated', 'MPNN', 'MPNN_separated', '1NN', '5NN']:
        # corr_ml = 'Random_Forest'
        # model = ml + '/' + corr_ml
        model = ml
        mtype = "axv"
        os.chdir(bd)
        os.makedirs("./Log_%s"%mtype, exist_ok=True)
        os.makedirs("./Score_%s"%mtype, exist_ok=True)        
            
        p = MakeScoreTable( tlist      = tlist,
                           targets     = None,
                            modeltype  = mtype,
                            model      = model,
                            dir_log    = './Log_%s/%s' %(mtype, model),
                            dir_score  = './Score_%s/%s' %(mtype, model),
                            )

        p.GetScores_TakeAverage()
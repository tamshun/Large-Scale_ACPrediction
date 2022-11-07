# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 14:09:59 2020

@author: Tamura
"""

import pandas as pd
import numpy as np
import os
import platform
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
    
    def __init__(self, targets, model, modeltype, dir_log, dir_score):
        
        super().__init__()
        self.targets   = targets
        self.model     = model
        self.mtype     = modeltype
        self.dir_log   = dir_log
        self.dir_score = dir_score
        
        os.makedirs(self.dir_score, exist_ok=True)
        
    def GetScores_TakeAverage(self):
        
        self.score_mean = dict()
        self.score_all  = pd.DataFrame()
        
        for target in self.targets:
            trials = self._GetOneScore(target)
            mean   = trials.mean()
            var    = trials.var()
            var.index = [i+'_var' for i in var.index]
            self.score_mean[target] = pd.concat([mean, var])
            trials['target'] = target
            self.score_all = pd.concat([self.score_all, trials], ignore_index=True)
        
        self.score_mean = pd.DataFrame.from_dict(self.score_mean).T
        self.score_all  = pd.DataFrame.from_dict(self.score_all).T    
        
        self.score_mean = self.score_mean.applymap(lambda x: np.round(x,3))  
          
        self.score_mean.to_csv(self.dir_score + '/mean.tsv', sep='\t')
        self.score_all.to_csv(self.dir_score + '/alltrial.tsv', sep='\t')
        print('$ score calculation done')        
        
    def _GetOneScore(self, target):
        
        d = dict()
        
        for i in range(3):
            
            if self.model in ['FCNN', 'MPNN']:
                path_log = os.path.join(self.dir_log, "%s_trial%d_test.tsv" %(target, i))
            elif 'NN' in self.model:
                path_log = os.path.join(self.dir_log, "%s_trial%d_test.tsv" %(target, i))
            else:
                path_log = os.path.join(self.dir_log, "%s_trial%d.tsv" %(target, i))
                
            log      = pd.read_csv(path_log, sep="\t", index_col=0)
            score    = self.CalcScores(log['trueY'], log['predY'], log['prob'])
            score['#tr']    = log['#tr'].iloc[0]
            score['#ac_tr'] = log['#ac_tr'].iloc[0]
            d[i] = score
        
        score = pd.DataFrame.from_dict(d, orient='index')
        
        return score


       

if __name__ == '__main__':
    if platform.system() == 'Darwin':
        bd    = "/Users/tamura/work/ACPredCompare/"
    else:
        bd    = "/home/tamuras0/work/ACPredCompare/"
    
    ml = 'SVM'
    
    # corr_ml = 'Random_Forest'
    # model = ml + '/' + corr_ml
    model = ml
    mtype = "wodirection_trtssplit"
    os.chdir(bd)
    os.makedirs("./Log_%s"%mtype, exist_ok=True)
    os.makedirs("./Score_%s"%mtype, exist_ok=True)
    
    tlist = pd.read_csv('./Dataset/target_list.tsv', sep='\t', index_col=0)
    tlist = tlist.loc[tlist['predictable_trtssplit'], :]
    
    # top10 targets having lots of AC
    # tlist = ['CHEMBL244', 'CHEMBL204', 'CHEMBL205', 'CHEMBL3594', 'CHEMBL261', 'CHEMBL264', 'CHEMBL3242', 'CHEMBL253', 'CHEMBL217', 'CHEMBL3837']
        
    p = MakeScoreTable( targets     = tlist['chembl_tid'],
                        modeltype   = mtype,
                        model       = model,
                        dir_log    = './Log_%s/%s' %(mtype, model),
                        dir_score  = './Score_%s/%s' %(mtype, model)
                        )

    p.GetScores_TakeAverage()
    # p.RenderToFig()
    
    # Barplot()
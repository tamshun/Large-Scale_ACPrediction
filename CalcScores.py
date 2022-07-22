# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 14:09:59 2020

@author: Tamura
"""

from operator import neg
import pandas as pd
import numpy as np
import sys
import os
import platform
import matplotlib.pyplot as plt
#import shap
from tqdm                              import tqdm
from Metrix.scores                     import RenderDF2Fig, CalcBasicScore
from Evaluation.Labelling              import AddLabel
from Evaluation.Score                  import ScoreTable_wDefinedLabel
from Plots.Barplot                     import MakeBarPlotsSeaborn as bar
from Plots.Lineplot                    import MakeLinePlotsSeaborn as line
from functools                         import partial
from collections                       import defaultdict

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
    
    def RenderToFig(self):
        
        ax = RenderDF2Fig(self.score_mean)
        plt.savefig(self.dir_score + 'mean.png')


def Barplot(x='target', y=['recall', 'auc_roc', 'matthews_coeff'] , hue='metric'):
    
    metrices = ['FCNN', 'SVM', 'Random_Forest', 'XGBoost']
    logs = pd.DataFrame()
    
    for metric in metrices:
        log           = pd.read_csv('./Score_trtssplit/%s_wodirection/mean.tsv'%metric, sep='\t', index_col=None)
        log['metric'] = metric
        log['target'] = log.iloc[:,0]
        logs          = pd.concat([logs, log], ignore_index=True)
    log['#tr'].iloc[0]
    for score in y:
        bar(logs, xname=x, yname=score, hue=hue, save_fig_name='./Score_trtssplit/Compare_%s.png'%score, rotate_x=True)            

if __name__ == '__main__':
    if platform.system() == 'Darwin':
        bd    = "/Users/tamura/work/ACPredCompare/"
    else:
        bd    = "/home/tamuras0/work/ACPredCompare/"
    
    ml = 'FCNN_separated'
    # corr_ml = 'Random_Forest'
    # model = ml + '/' + corr_ml
    model = ml
    mtype = "wodirection_trtssplit"
    os.chdir(bd)
    os.makedirs("./Log_%s"%mtype, exist_ok=True)
    os.makedirs("./Score_%s"%mtype, exist_ok=True)
    
    tlist = pd.read_csv('./Dataset/target_list.tsv', sep='\t', index_col=0)
    tlist = tlist.loc[tlist['predictable_trtssplit'], :]
    
        
    p = MakeScoreTable( targets     = tlist['chembl_tid'],
                        modeltype   = mtype,
                        model       = model,
                        dir_log    = './Log_%s/%s' %(mtype, model),
                        dir_score  = './Score_%s/%s' %(mtype, model)
                        )

    p.GetScores_TakeAverage()
    # p.RenderToFig()
    
    # Barplot()
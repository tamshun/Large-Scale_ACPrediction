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
    
    def __init__(self, tlist, model, modeltype, dir_log, dir_score):
        
        super().__init__()
        self.tlist   = tlist
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
    
    tlist = pd.read_csv('./Dataset/Stats/axv.tsv', sep='\t', index_col=0)
    
    for ml in ['MPNN', 'MPNN_separated']:
        # corr_ml = 'Random_Forest'
        # model = ml + '/' + corr_ml
        model = ml
        mtype = "axv"
        os.chdir(bd)
        os.makedirs("./Log_%s"%mtype, exist_ok=True)
        os.makedirs("./Score_%s"%mtype, exist_ok=True)        
            
        p = MakeScoreTable( tlist     = tlist,
                            modeltype   = mtype,
                            model       = model,
                            dir_log    = './Log_%s/%s' %(mtype, model),
                            dir_score  = './Score_%s/%s' %(mtype, model)
                            )

        p.GetScores_TakeAverage()
        # p.RenderToFig()
        
        # Barplot()
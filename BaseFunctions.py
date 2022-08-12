# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 14:09:59 2020

@author: Tamura
"""

#%%
import pandas as pd
import numpy as np
import random
from MMP.make_input                    import LeaveOneCoreOut, GetDiverseCore, DelTestCpdFromTrain
from Tools.ReadWrite                   import ToJson
from Fingerprint.Hash2BitManager       import Hash2Bits, FindBitLength
from collections                       import defaultdict, namedtuple
from sklearn.model_selection           import StratifiedShuffleSplit
from BaseClass                         import Initialize, MultipleAXVSplit, MultipleTrainTestSplit
    
class Base_wodirection(Initialize):
    
    def __init__(self, modeltype, dir_log=None, dir_score=None, data_split_metric='trtssplit'):
        
        super().__init__(modeltype=modeltype, dir_log=dir_log, dir_score=dir_score)
        
        self.trtssplit    = data_split_metric
        self.col          = ["core", "sub1", "sub2", "overlap"]


    def _SetParams(self, target):

        self.nbits_c = FindBitLength(self.ecfp, [self.col[0]])
        self.nbits_s = FindBitLength(self.ecfp, self.col[1:] )
        
        if self.trtssplit == 'axv':
            # Leave One Core Out
            all_seeds = pd.read_csv('./Dataset/Stats/axv.tsv', sep='\t', index_col=0).index
            seeds = [int(i.split('-Seed')[1]) for i in all_seeds if target == i.split('-Seed')[0]]
            self.data_split_generator = MultipleAXVSplit(self.main, seeds=seeds)
            self.testsetidx           = self.data_split_generator.keys()
            self.predictable          = True
        
        elif self.trtssplit == 'trtssplit':
            # Stratified Shuffled split
            if self._IsPredictableSet():
                self.data_split_generator = MultipleTrainTestSplit(self.main, n_dataset=3)
                self.testsetidx           = self.data_split_generator.keys()
                self.predictable          = True
            else:
                self.predictable          = False

        
    def _ReadDataFile(self, target):
   
        self.main = pd.read_csv("./Dataset/Data/%s.tsv" %target, sep="\t", index_col=0) # Single class MMSs were removed in curation.py.
        self.ecfp = pd.read_csv("./Dataset/ECFP/%s.tsv" %target, sep="\t", index_col=0)


    def _GetMatrices(self, cid, separated_input=False, unbiased=False):
        '''
        Wrapper function of TrainTestSplit_main, TrainTestSplit_ecfp, Hash2Bits 
        '''
        
        if self.trtssplit == 'trtssplit':
            tr, ts = self._TrainTestSplit_main(cid)
            
            if unbiased:
                tr = self._MakeTrainUnbiased(tr, seed=cid)
                
            df_trX, df_tsX = self._TrainTestSplit_ecfp(tr, ts)
            
            if not separated_input:
                df_trY, df_tsY, trX, trY, tsX, tsY = self._Hash2Bits(tr, ts, df_trX, df_tsX)
                
            elif separated_input:
                df_trY, df_tsY, trX, trY, tsX, tsY = self._Hash2Bits_separated(tr, ts, df_trX, df_tsX)

            return tr, ts, df_trX, df_trY, df_tsX, df_tsY, trX, trY, tsX, tsY
        
        elif self.trtssplit == 'axv':
            tr, cpdout, bothout = self._TrainTestSplit_main_axv(cid)
            
            if unbiased:
                tr = self._MakeTrainUnbiased(tr, seed=cid)
                
            df_trX, df_cpdoutX, df_bothoutX = self._TrainTestSplit_ecfp_axv(tr, cpdout, bothout)
            
            if not separated_input:
                df_trY, df_cpdoutY, df_bothoutY, trX, trY, cpdoutX, cpdoutY, bothoutX, bothoutY = self._Hash2Bits_axv(tr, cpdout, bothout, df_trX, df_cpdoutX, df_bothoutX)
                
            elif separated_input:
                df_trY, df_cpdoutY, df_bothoutY, trX, trY, cpdoutX, cpdoutY, bothoutX, bothoutY = self._Hash2Bits_separated_axv(tr, cpdout, bothout, df_trX, df_cpdoutX, df_bothoutX)
            
            return tr, cpdout, bothout, df_trX, df_trY, df_cpdoutX, df_cpdoutY, df_bothoutX, df_bothoutY, trX, trY, cpdoutX, cpdoutY, bothoutX, bothoutY
    
        
    def _TrainTestSplit_main(self, cid):
        
        generator = self.data_split_generator[cid]
        tr        = self.main.loc[generator.tridx,:]
        ts        = self.main.loc[generator.tsidx,:]
            
        return tr, ts
    
    
    def _TrainTestSplit_ecfp(self, tr, ts):
        
        df_trX = self.ecfp.loc[tr.index, :]
        df_tsX = self.ecfp.loc[ts.index, :]
        
        return df_trX, df_tsX
    
    def _MakeTrainUnbiased(self, tr, seed):
        
        ac    = tr[tr['class']==1]
        nonac = tr[tr['class']!=1]
        nac   = ac.shape[0]
        
        # random sample so that #non-ac equals to #ac 
        random.seed(seed)
        idx   = random.sample(range(nonac.shape[0]), nac)
        nonac = nonac.iloc[idx]
        
        tr = pd.concat([ac, nonac])
        
        return tr

    def _Hash2Bits(self, tr, ts, df_trX, df_tsX):
        
        df_trY = tr["class"]
        df_tsY = ts["class"]

        forward  = Hash2Bits(subdiff=False, sub_reverse=False)
        trX, trY = forward.GetMMPfingerprints_DF_unfold(df=df_trX, cols=self.col, Y=df_trY, nbits=[self.nbits_c, self.nbits_s], overlap="concat")
        tsX, tsY = forward.GetMMPfingerprints_DF_unfold(df=df_tsX, cols=self.col, Y=df_tsY, nbits=[self.nbits_c, self.nbits_s], overlap="concat")

        return df_trY, df_tsY, trX, trY, tsX, tsY
    
    def _Hash2Bits_separated(self, tr, ts, df_trX, df_tsX):
        
        df_trY = tr["class"]
        df_tsY = ts["class"]

        forward  = Hash2Bits(subdiff=False, sub_reverse=False)
        trX, trY = forward.GetSeparatedfingerprints_DF_unfold(df=df_trX, cols=self.col, Y=df_trY, nbits=[self.nbits_c, self.nbits_s])
        tsX, tsY = forward.GetSeparatedfingerprints_DF_unfold(df=df_tsX, cols=self.col, Y=df_tsY, nbits=[self.nbits_c, self.nbits_s])

        return df_trY, df_tsY, trX, trY, tsX, tsY
    
    def _TrainTestSplit_main_axv(self, cid):
        
        generator = self.data_split_generator[cid]
        tr        = self.main.loc[generator.tridx,        :]
        cpdout    = self.main.loc[generator.compound_out, :]
        bothout   = self.main.loc[generator.both_out,     :]        
            
        return tr, cpdout, bothout
    
    
    def _TrainTestSplit_ecfp_axv(self, tr, cpdout, bothout):
        
        df_trX      = self.ecfp.loc[tr.index     , :]
        df_cpdoutX  = self.ecfp.loc[cpdout.index , :]
        df_bothoutX = self.ecfp.loc[bothout.index, :]
        
        return df_trX, df_cpdoutX, df_bothoutX
    
    
    def _Hash2Bits_axv(self, tr, cpdout, bothout, df_trX, df_cpdoutX, df_bothoutX):
        
        df_trY      = tr["class"]
        df_cpdoutY  = cpdout["class"]
        df_bothoutY = bothout["class"]

        forward            = Hash2Bits(subdiff=False, sub_reverse=False)
        trX     , trY      = forward.GetMMPfingerprints_DF_unfold(df=df_trX, cols=self.col, Y=df_trY, nbits=[self.nbits_c, self.nbits_s], overlap="concat")
        cpdoutX , cpdoutY  = forward.GetMMPfingerprints_DF_unfold(df=df_cpdoutX , cols=self.col, Y=df_cpdoutY , nbits=[self.nbits_c, self.nbits_s], overlap="concat")
        bothoutX, bothoutY = forward.GetMMPfingerprints_DF_unfold(df=df_bothoutX, cols=self.col, Y=df_bothoutY, nbits=[self.nbits_c, self.nbits_s], overlap="concat")

        return df_trY, df_cpdoutY, df_bothoutY, trX, trY, cpdoutX, cpdoutY, bothoutX, bothoutY
    
    
    def _Hash2Bits_separated_axv(self, tr, cpdout, bothout, df_trX, df_cpdoutX, df_bothoutX):
        
        df_trY      = tr["class"]
        df_cpdoutY  = cpdout["class"]
        df_bothoutY = bothout["class"]

        forward            = Hash2Bits(subdiff=False, sub_reverse=False)
        trX     , trY      = forward.GetSeparatedfingerprints_DF_unfold(df=df_trX, cols=self.col, Y=df_trY, nbits=[self.nbits_c, self.nbits_s])
        cpdoutX , cpdoutY  = forward.GetSeparatedfingerprints_DF_unfold(df=df_cpdoutX , cols=self.col, Y=df_cpdoutY , nbits=[self.nbits_c, self.nbits_s])
        bothoutX, bothoutY = forward.GetSeparatedfingerprints_DF_unfold(df=df_bothoutX, cols=self.col, Y=df_bothoutY, nbits=[self.nbits_c, self.nbits_s])

        return df_trY, df_cpdoutY, df_bothoutY, trX, trY, cpdoutX, cpdoutY, bothoutX, bothoutY
    
    def _GetMatrices_3parts(self, cid):
        '''
        Wrapper function of TrainTestSplit_main, TrainTestSplit_ecfp, Hash2Bits 
        '''
        tr, ts = self._TrainTestSplit_main(cid)
        
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
    
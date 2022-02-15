from operator import add
from nbformat import read
import pandas as pd
import numpy as np
import os
from util.utility       import MakeLogFP, WriteMsgLogStdout
from Chemistry.Property import CountHeavyatoms, GetMolecularWeight


def ExploreDataset(bd, target):

    os.makedirs(bd, exist_ok=True)
    
    log  = MakeLogFP(bd + "%s.txt" %target)

    data_all = pd.read_csv("./Dataset/Data/%s.tsv" %target, sep="\t", index_col="id")
    row      = pd.read_csv('./Dataset/Row/%s.tsv' %target, sep="\t", index_col="chembl_cid")
    data     = data_all.copy()
    
    n_data   = data.shape[0]
    n_ac     = np.where(data["class"]==1)[0].shape[0]
    n_nonac  = np.where(data["class"]==-1)[0].shape[0]
    n_mms    = int(data["core_id"].max() + 1)

    used_cpd_idx = np.union1d(data["chembl_cid1"], data["chembl_cid2"])
    used_cpd     = row.loc[used_cpd_idx, :]

    all_pots     = pd.concat([data['pot1'], data['pot2']])
    stats_ha     = CountHeavyatoms(used_cpd, "nonstereo_aromatic_smiles").describe()
    stats_mw     = GetMolecularWeight(used_cpd, "nonstereo_aromatic_smiles").describe()
    stats_sub1ha = CountHeavyatoms(data, "sub1").describe()
    stats_sub2ha = CountHeavyatoms(data, "sub2").describe()

    WriteMsgLogStdout(log, "Target name: %s" %target, add_newline=True)
    WriteMsgLogStdout(log, "\n")

    WriteMsgLogStdout(log, "--- Whole dataset stats---")
    WriteMsgLogStdout(log, "#data: %d" %n_data)
    WriteMsgLogStdout(log, "#ac: %d" %n_ac)
    WriteMsgLogStdout(log, "#non-ac: %d" %n_nonac)
    WriteMsgLogStdout(log, "#MMS: %d" %n_mms, add_newline=True)
    WriteMsgLogStdout(log, "\n")

    WriteMsgLogStdout(log, "pKd_max: %.2f" %all_pots.max())
    WriteMsgLogStdout(log, "pKd_ave: %.2f" %all_pots.mean())
    WriteMsgLogStdout(log, "pKd_min: %.2f" %all_pots.min(), add_newline=True)
    WriteMsgLogStdout(log, "pKd_thres: %.3f" %data['thres'].iloc[0])
    WriteMsgLogStdout(log, "\n")

    WriteMsgLogStdout(log, "#HA_max: %.2f" %stats_ha["max"])
    WriteMsgLogStdout(log, "#HA_ave: %.2f" %stats_ha["mean"])
    WriteMsgLogStdout(log, "#HA_min: %.2f" %stats_ha["min"], add_newline=True)
    WriteMsgLogStdout(log, "\n")

    WriteMsgLogStdout(log, "MW_max: %.2f" %stats_mw["max"])
    WriteMsgLogStdout(log, "MW_ave: %.2f" %stats_mw["mean"])
    WriteMsgLogStdout(log, "MW_min: %.2f" %stats_mw["min"], add_newline=True)
    WriteMsgLogStdout(log, "\n")

    WriteMsgLogStdout(log, "#HA_sub1_max: %.2f" %stats_sub1ha["max"])
    WriteMsgLogStdout(log, "#HA_sub1_ave: %.2f" %stats_sub1ha["mean"])
    WriteMsgLogStdout(log, "#HA_sub1_min: %.2f" %stats_sub1ha["min"], add_newline=True)
    WriteMsgLogStdout(log, "\n")

    WriteMsgLogStdout(log, "#HA_sub2_max: %.2f" %stats_sub2ha["max"])
    WriteMsgLogStdout(log, "#HA_sub2_ave: %.2f" %stats_sub2ha["mean"])
    WriteMsgLogStdout(log, "#HA_sub2_min: %.2f" %stats_sub2ha["min"], add_newline=True)
    WriteMsgLogStdout(log, "\n")

    WriteMsgLogStdout(log, "--- Stats by MMS---", add_newline=True)
    WriteMsgLogStdout(log, "\n")

    for cid in range(n_mms):
        WriteMsgLogStdout(log, "--- cid: %d---" %cid, add_newline=True)

        data     = data_all[data_all["core_id"]==cid]
        n_data   = data.shape[0]
        n_ac     = np.where(data["class"]==1)[0].shape[0]
        n_nonac  = np.where(data["class"]==-1)[0].shape[0]
        n_mms    = data["core_id"].max() + 1

        used_cpd_idx = np.union1d(data["chembl_cid1"], data["chembl_cid2"])
        used_cpd     = row.loc[used_cpd_idx, :]

        stats_pki = used_cpd["pPot_mean"].describe()
        stats_ha  = CountHeavyatoms(used_cpd, "nonstereo_aromatic_smiles").describe()
        stats_mw  = GetMolecularWeight(used_cpd, "nonstereo_aromatic_smiles").describe()
        stats_sub1ha = CountHeavyatoms(data, "sub1").describe()
        stats_sub2ha = CountHeavyatoms(data, "sub2").describe()

        WriteMsgLogStdout(log, "#data: %d" %n_data)
        WriteMsgLogStdout(log, "#ac: %d" %n_ac)
        WriteMsgLogStdout(log, "#non-ac: %d" %n_nonac)
        WriteMsgLogStdout(log, "#MMS: %d" %n_mms, add_newline=True)
        WriteMsgLogStdout(log, "\n")

        WriteMsgLogStdout(log, "pKi_max: %.2f" %stats_pki["max"])
        WriteMsgLogStdout(log, "pKi_ave: %.2f" %stats_pki["mean"])
        WriteMsgLogStdout(log, "pKi_min: %.2f" %stats_pki["min"], add_newline=True)
        WriteMsgLogStdout(log, "\n")

        WriteMsgLogStdout(log, "#HA_max: %.2f" %stats_ha["max"])
        WriteMsgLogStdout(log, "#HA_ave: %.2f" %stats_ha["mean"])
        WriteMsgLogStdout(log, "#HA_min: %.2f" %stats_ha["min"], add_newline=True)
        WriteMsgLogStdout(log, "\n")

        WriteMsgLogStdout(log, "MW_max: %.2f" %stats_mw["max"])
        WriteMsgLogStdout(log, "MW_ave: %.2f" %stats_mw["mean"])
        WriteMsgLogStdout(log, "MW_min: %.2f" %stats_mw["min"], add_newline=True)
        WriteMsgLogStdout(log, "\n")

        WriteMsgLogStdout(log, "#HA_sub1_max: %.2f" %stats_sub1ha["max"])
        WriteMsgLogStdout(log, "#HA_sub1_ave: %.2f" %stats_sub1ha["mean"])
        WriteMsgLogStdout(log, "#HA_sub1_min: %.2f" %stats_sub1ha["min"], add_newline=True)
        WriteMsgLogStdout(log, "\n")

        WriteMsgLogStdout(log, "#HA_sub2_max: %.2f" %stats_sub2ha["max"])
        WriteMsgLogStdout(log, "#HA_sub2_ave: %.2f" %stats_sub2ha["mean"])
        WriteMsgLogStdout(log, "#HA_sub2_min: %.2f" %stats_sub2ha["min"], add_newline=True)
        WriteMsgLogStdout(log, "\n")
        
    
    log.close()
    
    
def WriteNumSamples():
    
    file = './Dataset/target_list.tsv'
    
    df = pd.read_csv(file, sep='\t', index_col='chembl_tid')
    df['#mmp'] = None
    df['#mms'] = None
    
    for tid in df.index:
        data = pd.read_csv('./Dataset/Data/%s.tsv' %tid, sep='\t', index_col=0)
        df.loc[tid, '#mmp'] = data.shape[0]
        df.loc[tid, '#mms'] = np.unique(data['core_id']).shape[0]
    
    df = df.reset_index().set_index('Unnamed: 0')
    
    df.to_csv(file, sep='\t')
        
        

if __name__ == "__main__":

    # tlist = pd.read_csv('./Dataset/target_list.tsv', sep='\t', index_col=0)
    
    # for sr in tlist.iterrows():
    #     target = sr[1]['chembl_tid']
    #     ExploreDataset("./Dataset/Stats/", target)
        
    WriteNumSamples()
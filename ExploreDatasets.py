from operator import add
import pandas as pd
import numpy as np
import os
from util.utility       import MakeLogFP, WriteMsgLogStdout
from Chemistry.Property import CountHeavyatoms, GetMolecularWeight


def ExploreDataset(bd, target):

    log  = MakeLogFP(bd + "Stats_%s.txt" %target)

    data_all = pd.read_csv("./Dataset/Data/%s.tsv" %target, sep="\t", index_col="id")
    #ecfp     = pd.read_csv("./ECFP/%s.tsv", sep="\t", index_col="id")
    data     = data_all.copy()
    

    n_data   = data.shape[0]
    n_ac     = np.where(data["class"]==1)[0].shape[0]
    n_nonac  = np.where(data["class"]==-1)[0].shape[0]
    n_mms    = data["core_id"].max() + 1

    used_cpd_idx = np.union1d(data["funatsu_lab_id1"], data["funatsu_lab_id2"]).astype(int)

    all_pots     = pd.concat([data['pot1'], data['pot2']])
    # stats_ha     = CountHeavyatoms(used_cpd, "non_stereo_aromatic_smieles").describe()
    # stats_mw     = GetMolecularWeight(used_cpd, "non_stereo_aromatic_smieles").describe()
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

    WriteMsgLogStdout(log, "pKi_max: %.2f" %all_pots.max())
    WriteMsgLogStdout(log, "pKi_ave: %.2f" %all_pots.mean())
    WriteMsgLogStdout(log, "pKi_min: %.2f" %all_pots.min(), add_newline=True)
    WriteMsgLogStdout(log, "\n")

    # WriteMsgLogStdout(log, "#HA_max: %.2f" %stats_ha["max"])
    # WriteMsgLogStdout(log, "#HA_ave: %.2f" %stats_ha["mean"])
    # WriteMsgLogStdout(log, "#HA_min: %.2f" %stats_ha["min"], add_newline=True)
    # WriteMsgLogStdout(log, "\n")

    # WriteMsgLogStdout(log, "MW_max: %.2f" %stats_mw["max"])
    # WriteMsgLogStdout(log, "MW_ave: %.2f" %stats_mw["mean"])
    # WriteMsgLogStdout(log, "MW_min: %.2f" %stats_mw["min"], add_newline=True)
    # WriteMsgLogStdout(log, "\n")

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

    # for cid in range(n_mms):
    #     WriteMsgLogStdout(log, "--- cid: %d---" %cid, add_newline=True)

    #     data     = data_all[data_all["core_id"]==cid]
    #     n_data   = data.shape[0]
    #     n_ac     = np.where(data["class"]==1)[0].shape[0]
    #     n_nonac  = np.where(data["class"]==-1)[0].shape[0]
    #     n_mms    = data["core_id"].max() + 1

    #     used_cpd_idx = np.union1d(data["funatsu_lab_id1"], data["funatsu_lab_id2"]).astype(int)
    #     used_cpd     = cpd_inf.loc[used_cpd_idx, :]

    #     stats_pki = used_cpd["pot.(log,Ki)"].describe()
    #     stats_ha  = CountHeavyatoms(used_cpd, "non_stereo_aromatic_smieles").describe()
    #     stats_mw  = GetMolecularWeight(used_cpd, "non_stereo_aromatic_smieles").describe()
    #     stats_sub1ha = CountHeavyatoms(data, "sub1").describe()
    #     stats_sub2ha = CountHeavyatoms(data, "sub2").describe()

    #     WriteMsgLogStdout(log, "#data: %d" %n_data)
    #     WriteMsgLogStdout(log, "#ac: %d" %n_ac)
    #     WriteMsgLogStdout(log, "#non-ac: %d" %n_nonac)
    #     WriteMsgLogStdout(log, "#MMS: %d" %n_mms, add_newline=True)
    #     WriteMsgLogStdout(log, "\n")

    #     WriteMsgLogStdout(log, "pKi_max: %.2f" %stats_pki["max"])
    #     WriteMsgLogStdout(log, "pKi_ave: %.2f" %stats_pki["mean"])
    #     WriteMsgLogStdout(log, "pKi_min: %.2f" %stats_pki["min"], add_newline=True)
    #     WriteMsgLogStdout(log, "\n")

    #     # WriteMsgLogStdout(log, "#HA_max: %.2f" %stats_ha["max"])
    #     # WriteMsgLogStdout(log, "#HA_ave: %.2f" %stats_ha["mean"])
    #     # WriteMsgLogStdout(log, "#HA_min: %.2f" %stats_ha["min"], add_newline=True)
    #     # WriteMsgLogStdout(log, "\n")

    #     # WriteMsgLogStdout(log, "MW_max: %.2f" %stats_mw["max"])
    #     # WriteMsgLogStdout(log, "MW_ave: %.2f" %stats_mw["mean"])
    #     # WriteMsgLogStdout(log, "MW_min: %.2f" %stats_mw["min"], add_newline=True)
    #     # WriteMsgLogStdout(log, "\n")

    #     WriteMsgLogStdout(log, "#HA_sub1_max: %.2f" %stats_sub1ha["max"])
    #     WriteMsgLogStdout(log, "#HA_sub1_ave: %.2f" %stats_sub1ha["mean"])
    #     WriteMsgLogStdout(log, "#HA_sub1_min: %.2f" %stats_sub1ha["min"], add_newline=True)
    #     WriteMsgLogStdout(log, "\n")

    #     WriteMsgLogStdout(log, "#HA_sub2_max: %.2f" %stats_sub2ha["max"])
    #     WriteMsgLogStdout(log, "#HA_sub2_ave: %.2f" %stats_sub2ha["mean"])
    #     WriteMsgLogStdout(log, "#HA_sub2_min: %.2f" %stats_sub2ha["min"], add_newline=True)
    #     WriteMsgLogStdout(log, "\n")
        
    
    log.close()

if __name__ == "__main__":

    tlist = pd.read_csv('./Dataset/target_list.tsv', sep='\t', index_col=0)
    
    for sr in tlist.iterrows():
        target = sr[1]['target']
        ExploreDataset("./Dataset/", target)
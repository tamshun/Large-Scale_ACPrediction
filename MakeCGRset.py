import pandas as pd
import os
from tqdm    import tqdm
from MMP.CGR import GetCGR
from sys import path
import numpy as np
from itertools                       import compress
from DataSetGeneration.GenerateFPSet import CalcECFP_withTrack
from util.utility                    import MakeLogFP, WriteMsgLogStdout
from chem.transform import SmilesToOEGraphMol
from Fingerprint.ECFP4 import CalcECFPSparse_tracker, ConvertAnnotationsIntoStr
from functools import partial



def CalcECFP_withTrack(df, cols, diameter=4, nbits=4096, use_features=False, smarts=None, direction=False):
    
    df_fp = df.iloc[:,:3]
    
    func_genfp = partial(CalcECFPSparse_tracker, diameter=diameter, nbits=nbits)

    for i, col in enumerate(cols):

        if direction != False:
            func_genfp = partial(func_genfp, pos=i)

        hashes = []
        patts  = []
        pid    = []
        subsmi = []
        
        print("-------Calculations for %s-------" %col)
        for smiles in tqdm(df[col]):
            mol = SmilesToOEGraphMol(smiles)
            anno  = func_genfp(mol)
            fold = isinstance(nbits, int)
            str_hash, str_pidx, str_pat, str_smi = ConvertAnnotationsIntoStr(anno, is_folded=fold)
            
            hashes.append(str_hash)
            patts.append(str_pat)
            pid.append(str_pidx)
            subsmi.append(str_smi) 
        
        df_fp[col]          = hashes
        df_fp[col+"_pidx"]  = pid
        df_fp[col+"_patts"] = patts
        df_fp[col+"_smi"]   = subsmi

        
    return df_fp

def main(target_list, outdir, col=['core', 'sub1', 'sub2']):
    
    os.makedirs(outdir, exist_ok=True)
    
    for target in target_list:
        print('$ %s is ongoing.' %target)
        df = pd.read_csv('./Dataset/Data/%s.tsv' %target, sep='\t', index_col=0)
        df_cgr = GetCGR_df(df, col)
        df_cgr.to_csv(outdir + target + '.tsv', sep='\t')
        
        
def GetCGR_df(df, col):
    
    cgr_list = [GetCGR(sr[col[0]], sr[col[1]], sr[col[2]]) for i, sr in tqdm(df.iterrows())]
    df_cgr   = df.copy()
    df_cgr['CGR'] = cgr_list
    
    return df_cgr


class Curation_Base():

    def __init__(self, bd, target, direc=True):

        self.basedir       = bd
        self.t             = target
        self.log           = MakeLogFP(bd+"/log_CGR_ECFP/log_%s.txt" %target)
        self.original_file = self._readfile(bd)
        self.direc         = direc
        
        os.makedirs(bd+"/CGR_ECFP", exist_ok=True)

        self.path_two_cls_main = os.path.join(self.basedir, "Data/%s_del_one_cls_mms.tsv" %target)
        self.path_curated_main = os.path.join(self.basedir, "Data/%s.tsv" %target)
        self.path_wosubdiff    = os.path.join(bd, "CGR_ECFP/%s_wosubdiff.tsv" %target)
        self.path_dellowlv     = os.path.join(bd, "CGR_ECFP/%s_dellowlv.tsv" %target)
        self.path_finalproduct = os.path.join(bd, "CGR_ECFP/%s.tsv" %target)

        self.path_prevout      = None
        

    def _readfile(self, bd):
        path = os.path.join(bd, "Data/%s.tsv" %self.t)
        df = pd.read_csv(path, sep='\t', index_col=0)

        WriteMsgLogStdout(self.log, "--- Original data load ---")
        WriteMsgLogStdout(self.log, "    # Original data : %d"%df.shape[0])

        return df


    def RemoveOneClassMMS(self):

        df = self.original_file
        # df, _ = GetDiverseCore(df)

        outpath = self.path_two_cls_main

        WriteMsgLogStdout(self.log, "--- Remove MMS with only one class ---")
        # WriteMsgLogStdout(self.log, "    # Cpds : %d"%df.shape[0])
        # WriteMsgLogStdout(self.log, "    # MMPs : %d"%df["core_id"].max())

        df.to_csv(outpath, sep="\t")

    
    def CalcECFP(self, cols, nbits, diameter):
        """
        Note: add foward and reverse
        """
        if not isinstance(nbits, int):
            nbits = "unfold"

        self.cols     = cols
        self.diameter = diameter

        df = pd.read_csv(self.path_two_cls_main, sep="\t", index_col=0)

        WriteMsgLogStdout(self.log, "--- Calclate ECFP ---")
        WriteMsgLogStdout(self.log, "    # Loaded cpds : %d"%df.shape[0])
        WriteMsgLogStdout(self.log, "    Diameter      : %d"%self.diameter)
        WriteMsgLogStdout(self.log, "    nbits         : %s"%nbits)

        wodirec   = CalcECFP_withTrack(df, cols=cols, diameter=diameter, nbits=nbits, direction=False)
        wodirec.to_csv(os.path.join(self.basedir, "CGR_ECFP/%s_nodirec_original.tsv" %self.t), sep="\t")

        if self.direc:
            wdirec_f  = CalcECFP_withTrack(df, diameter=diameter, nbits=nbits, direction="forward")
            wdirec_r  = CalcECFP_withTrack(df, diameter=diameter, nbits=nbits, direction="reverse")

            wodirec.to_csv(os.path.join(self.basedir, "CGR_ECFP/%s_forward_original.tsv" %self.t), sep="\t")
            wodirec.to_csv(os.path.join(self.basedir, "CGR_ECFP/%s_reverse_original.tsv" %self.t), sep="\t")

            wdirec_f.columns = [c+"_forward" for c in wdirec_f.columns]
            wdirec_r.columns = [c+"_reverse" for c in wdirec_r.columns]

            df_fp = pd.concat([wodirec, wdirec_f.iloc[:,3:], wdirec_r.iloc[:,3:]], axis=1, sort=False)
        
        else:
            df_fp = wodirec.copy()

        df_fp.index = df.index
        df_fp.to_csv(self.path_wosubdiff, sep="\t")

        self.path_prevout = self.path_wosubdiff


    def DeleteLowLevel(self, deletelv):

        df    = pd.read_csv(self.path_two_cls_main, sep="\t", index_col=0)
        ecfp  = pd.read_csv(self.path_prevout, sep="\t", index_col=0)

        nbits = "unfold"

        WriteMsgLogStdout(self.log, "--- Delete low level hash ---")
        WriteMsgLogStdout(self.log, "    # Loaded cpds    : %d"%df.shape[0])
        WriteMsgLogStdout(self.log, "    Deleted diameter : %d"%deletelv)
        WriteMsgLogStdout(self.log, "    nbits            : %s"%nbits)

        cols = ecfp.columns[3:]

        ecfp_dellow = pd.DataFrame(columns=ecfp.columns)

        for i, item in tqdm(ecfp.iterrows()):
            
            new_item = []

            dict_flag = {}

            for part in self.cols:

                for deldiameter in range(deletelv+1):

                    boolen = np.array([True if "lvl%d"%deldiameter  in pidx else False for pidx in item["%s_pidx"%part].split(" ")])
                    
                    if deldiameter==0:
                        flag = boolen
                    else:
                        flag += boolen
                    
                dict_flag[part] = ~flag

            for c in ecfp.columns:

                if c not in cols:
                    new_item += [item[c]]

                else:
                    
                    part = c.split("_")[0]
                    item_split = np.array(item[c].split(" "))
                    idx = dict_flag[part]
                    new_item += [" ".join(item_split[idx].tolist())]

            sr_new_item = pd.Series(new_item, index=ecfp.columns)
            ecfp_dellow = ecfp_dellow.append(sr_new_item, ignore_index=True)

        ecfp_dellow.index = df.index
        ecfp_dellow.to_csv(self.path_dellowlv, sep="\t")
        self.path_prevout = self.path_dellowlv

        _, _ = self._NanCheck(df)
        


    def _NanCheck(self, data):
        
        df = pd.read_csv(self.path_prevout, sep="\t", index_col=0)

        WriteMsgLogStdout(self.log, "--- Delete MMPs with Nan ---")

        if "overlap" in df.columns:
            cols = [c for c in df.columns if "overlap" not in c]
            df_copy = df.copy().loc[:, cols]
        else:
            df_copy = df.copy()

        # Loaded df can include cells with the value of Nan because of previous subdiff operation. 
        flag   = df_copy.isna().any(axis=1)
        idx_na = df_copy.index[np.where(flag)]

        if idx_na.shape[0] == 0:
            WriteMsgLogStdout(self.log, "    No Nan was found.")
        else:
            WriteMsgLogStdout(self.log, "    Nan was found!")

        for idx in idx_na:
            WriteMsgLogStdout(self.log, "      pd_idx : %d"%int(idx))
            WriteMsgLogStdout(self.log, "      id     : %s"%df.loc[idx, "id"])
            WriteMsgLogStdout(self.log, "      core   : %s"%data.loc[idx, "core"])
            WriteMsgLogStdout(self.log, "      sub1   : %s"%data.loc[idx, "sub1"])
            WriteMsgLogStdout(self.log, "      sub2   : %s"%data.loc[idx, "sub2"])
            WriteMsgLogStdout(self.log, "      class  : %s\n"%data.loc[idx, "class"])

        df   = df.loc[~flag, :]
        data = data.loc[~flag, :]
        WriteMsgLogStdout(self.log, "    All rows containing Nan was deleted")

        data.to_csv(self.path_curated_main, sep="\t")
        df.to_csv(self.path_prevout, sep="\t")

        return data, df

    def ApplySubdiff_concat(self):
        """
        Concat ver
        """
        data = pd.read_csv(self.path_curated_main, sep="\t", index_col=0)
        df   = pd.read_csv(self.path_prevout, sep="\t", index_col=0)

        df_subdiff = pd.DataFrame(columns=df.columns) 

        WriteMsgLogStdout(self.log, "--- Apply substructure difference ---")
        WriteMsgLogStdout(self.log, "    # Loaded cpds : %d"%df.shape[0])

        for i, sr in tqdm(df.iterrows()):
            sr_new = self._subdiff_cat(sr)
            df_subdiff = df_subdiff.append(sr_new, ignore_index=True)
        
        df_subdiff.index = df.index
        df_subdiff.to_csv(self.path_finalproduct, sep="\t")

        self.path_prevout = self.path_finalproduct

        _, _ = self._NanCheck(data)


    def _subdiff_cat(self, sr):

        container = {}
        sub1_wodirec   = [int(i) for i in sr["sub1"].split(" ")]
        sub2_wodirec   = [int(i) for i in sr["sub2"].split(" ")]

        common_feature = np.intersect1d(sub1_wodirec, sub2_wodirec)
        flag_s1 = ~np.isin(sub1_wodirec, common_feature) #unique features
        flag_s2 = ~np.isin(sub2_wodirec, common_feature) #unique features

        for idx in sr.index:
            
            if ("sub1" not in idx) and ("sub2" not in idx):
                container[idx] = sr[idx]
                continue
            
            else:
                if "sub1" in idx:
                    flag = flag_s1

                    idx_ov = "overlap" + idx[4:]
                    features_ov = list(compress(sr[idx].split(" "), ~flag))
                    container[idx_ov] = " ".join(features_ov)

                elif "sub2" in idx:
                    flag = flag_s2
                
                features_subdiff = list(compress(sr[idx].split(" "), flag))
                container[idx]   = " ".join(features_subdiff)

        cols = list(sr.index) + self._make_ovcol()
        sr_new = pd.Series(container)
        sr_new = sr_new.loc[cols]

        return sr_new
    
    def _make_ovcol(self):

        part   = [None, "_forward", "_reverse"]
        suffix = [None, "_pidx", "_patts", "_smi"]

        col_ov = []
        for s in suffix:
            c = "overlap"
            if s != None:
                c += s            
            for p in part:
                colname = c
                if p != None:
                    colname += p
                col_ov += [colname]

        return col_ov

    def AssignNewHashBits(self):
        
        data = pd.read_csv(self.path_curated_main, sep="\t", index_col=0)
        df   = pd.read_csv(self.path_prevout     , sep="\t", index_col=0)

        WriteMsgLogStdout(self.log, "--- Assign new hash values to unfolded hash ---")
        WriteMsgLogStdout(self.log, "    # Loaded cpds : %d"%df.shape[0])

        prefix = ["core", "sub1", "sub2"]

        if "overlap" in df.columns:
            prefix += ["overlap"]

        for suffix in [None, "_forward", "_reverse"]:
            
            if suffix is not None:
               cols = [p + suffix for p in prefix]
            else:
                cols = prefix 

            # For core
            # coleect hashes
            c        = cols[0]
            hash_c   = df[c].apply(lambda x:[int(h) for h in x.split(" ")])
            # Assign new hashes
            uhash_c  = np.sort(np.unique(hash_c.sum()))
            sr_c     = pd.Series(range(uhash_c.shape[0]), index=uhash_c)
            # Replace old hash to new one
            df[c]    = hash_c.apply(lambda x:" ".join([str(sr_c[i]) for i in x]))

            # For sub
            len_c    = sr_c.shape[0]

            s1       = cols[1]
            hash_s1  = df[s1].apply(lambda x:[int(h) for h in x.split(" ")])
            uhash_s1 = np.unique(hash_s1).sum()

            s2       = cols[2]
            hash_s2  = df[s2].apply(lambda x:[int(h) for h in x.split(" ")])
            uhash_s2 = np.unique(hash_s2).sum()

            uhash_s  = np.union1d(uhash_s1, uhash_s2)

            # if "overlap" in df.columns:
            #     ov       = cols[3]
            #     sr_ov    = df.copy()[ov].dropna()
            #     hash_ov  = sr_ov.apply(lambda x:[int(h) for h in x.split(" ")])
            #     uhash_ov = np.unique(hash_ov).sum()
            #     uhash_s  = np.union1d(uhash_s, uhash_ov)

            sr_s     = pd.Series(range(uhash_s.shape[0]), index=uhash_s)
            print("no moving")
            #sr_s     = sr_s + len_c # Add max of core hash for concatnation in MMPfingerprints
            df[s1]   = hash_s1.apply(lambda x:" ".join([str(sr_s[i]) for i in x]))
            df[s2]   = hash_s2.apply(lambda x:" ".join([str(sr_s[i]) for i in x]))           

            # if "overlap" in df.columns:
            #     newov = []
            #     for item in df[ov]:
            #         if isinstance(item, str):
            #             hash_overlap  = [int(h) for h in item.split(" ")]
            #             newov += [" ".join([str(sr_s[i]) for i in hash_overlap])]
            #         else:
            #             newov += [None]
            #     df[ov] = newov

        WriteMsgLogStdout(self.log, "    # calculated ecfp : %d"%df.shape[0])
        
        df.to_csv(self.path_newhash, sep="\t")
        self.path_prevout = self.path_newhash
        _, _ = self._NanCheck(data)



    def Close_log(self):
        
        self.log.close()


class Curation_classification(Curation_Base):

    def __init__(self, bd, target, direc =True):
        super().__init__(bd, target, direc=direc)

        self.path_subdiff      = os.path.join(bd, "ECFP/%s_wsubdiff.tsv" %target)
        self.path_newhash      = os.path.join(bd, "ECFP/%s.tsv" %target)


    def ApplySubdiff(self):

        data = pd.read_csv(self.path_two_cls_main, sep="\t", index_col=0)
        df = pd.read_csv(self.path_dellowlv, sep="\t", index_col=0)

        data, df = self._NanCheck(data)

        df_subdiff  = pd.DataFrame(columns=df.columns)

        WriteMsgLogStdout(self.log, "--- Apply substructure difference ---")
        WriteMsgLogStdout(self.log, "    # Loaded cpds : %d"%df.shape[0])

        for i, sr in tqdm(df.iterrows()):
            df_subdiff = df_subdiff.append(self._subdiff(sr), ignore_index=True)
        
        df_subdiff.index = df.index
        df_subdiff.to_csv(self.path_subdiff, sep="\t")

        self.path_prevout = self.path_subdiff
        _,_ = self._NanCheck(data)


    def _subdiff(self, sr):

        sub1_wodirec   = [int(i) for i in sr["sub1"].split(" ")]
        sub2_wodirec   = [int(i) for i in sr["sub2"].split(" ")]

        flag_s1 = ~np.isin(sub1_wodirec, sub2_wodirec)
        flag_s2 = ~np.isin(sub2_wodirec, sub1_wodirec)

        for idx in sr.index:
            
            if ("sub1_" not in idx) and ("sub2_" not in idx):
                continue

            elif "sub1" in idx:
                flag = flag_s1

            elif "sub2" in idx:
                flag = flag_s2
            
            features_subdiff = list(compress(sr[idx].split(" "), flag))
            sr[idx] = " ".join(features_subdiff)

        return sr
    
    
def Run(tname, diameter, deletelv):

    bd = './Dataset/' #"/home/tamura/work/ACPredCompare/DataSets"    
    p = Curation_classification(bd, tname, direc=False)
    p.RemoveOneClassMMS()
    p.CalcECFP("unfold", cols=['CGR'], diameter=diameter)
    p.DeleteLowLevel(deletelv=deletelv)
    p.AssignNewHashBits()
    p.ApplySubdiff_concat()
    p.Close_log


        
if __name__ == '__main__':
    
    target_list = pd.read_csv('./Dataset/target_list.tsv', sep='\t', index_col=0)
    outdir = './Dataset/CGR/'
    
    main(target_list['chembl_tid'], outdir)
    

        
    
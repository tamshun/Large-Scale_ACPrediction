from sys import path
import pandas as pd
import numpy as np
import os
from itertools                       import compress
from tqdm                            import tqdm
from DataSetGeneration.GenerateFPSet import CalcECFP_withTrack
from MMP.make_input                  import GetDiverseCore
from util.utility                    import MakeLogFP, WriteMsgLogStdout
from Fingerprint.Hash2BitManager     import Hash2Bits, FindBitLength
from collections                     import defaultdict

class Curation_Base():

    def __init__(self, bd, target, direc=True):

        self.basedir       = bd
        self.t             = target
        self.log           = MakeLogFP(bd+"/log_curation_%s.txt" %target)
        self.original_file = self._readfile(bd)
        self.direc         = direc
        
        os.makedirs(bd+"/ECFP", exist_ok=True)

        self.path_two_cls_main = os.path.join(self.basedir, "Data/%s_del_one_cls_mms.tsv" %target)
        self.path_curated_main = os.path.join(self.basedir, "Data/%s.tsv" %target)
        self.path_wosubdiff    = os.path.join(bd, "ECFP/%s_wosubdiff.tsv" %target)
        self.path_dellowlv     = os.path.join(bd, "ECFP/%s_dellowlv.tsv" %target)
        self.path_finalproduct = os.path.join(bd, "ECFP/%s.tsv" %target)

        self.path_prevout      = None
        

    def _readfile(self, bd):
        path = os.path.join(bd, "Data/%s_all.tsv" %self.t)
        df = pd.read_csv(path, sep='\t', index_col=0)

        WriteMsgLogStdout(self.log, "--- Original data load ---")
        WriteMsgLogStdout(self.log, "    # Original data : %d"%df.shape[0])

        return df


    def RemoveOneClassMMS(self):

        df = self.original_file
        df, _ = GetDiverseCore(df)

        outpath = self.path_two_cls_main

        WriteMsgLogStdout(self.log, "--- Remove MMS with only one class ---")
        WriteMsgLogStdout(self.log, "    # Cpds : %d"%df.shape[0])
        WriteMsgLogStdout(self.log, "    # MMPs : %d"%df["core_id"].max())

        df.to_csv(outpath, sep="\t")

    
    def CalcECFP(self, nbits, diameter):
        """
        Note: add foward and reverse
        """
        if not isinstance(nbits, int):
            nbits = "unfold"

        self.diameter = diameter

        df = pd.read_csv(self.path_two_cls_main, sep="\t", index_col=0)

        WriteMsgLogStdout(self.log, "--- Calclate ECFP ---")
        WriteMsgLogStdout(self.log, "    # Loaded cpds : %d"%df.shape[0])
        WriteMsgLogStdout(self.log, "    Diameter      : %d"%self.diameter)
        WriteMsgLogStdout(self.log, "    nbits         : %s"%nbits)

        wodirec   = CalcECFP_withTrack(df, diameter=diameter, nbits=nbits, direction=False)
        wodirec.to_csv(os.path.join(self.basedir, "ECFP/%s_nodirec_original.tsv" %self.t), sep="\t")

        if self.direc:
            wdirec_f  = CalcECFP_withTrack(df, diameter=diameter, nbits=nbits, direction="forward")
            wdirec_r  = CalcECFP_withTrack(df, diameter=diameter, nbits=nbits, direction="reverse")

            wodirec.to_csv(os.path.join(self.basedir, "ECFP/%s_forward_original.tsv" %self.t), sep="\t")
            wodirec.to_csv(os.path.join(self.basedir, "ECFP/%s_reverse_original.tsv" %self.t), sep="\t")

            wdirec_f.columns = [c+"_forward" for c in wdirec_f.columns]
            wdirec_r.columns = [c+"_reverse" for c in wdirec_r.columns]

            df_fp = pd.concat([wodirec, wdirec_f.iloc[:,3:], wdirec_r.iloc[:,3:]], axis=1, sort=False)
        
        else:
            df_fp = wodirec.copy()

        df_fp.index = df.index
        df_fp.to_csv(self.path_wosubdiff, sep="\t")

        self.path_prevout = self.path_wosubdiff


    def DeleteLowLevel(self, deletelv):

        df = pd.read_csv(self.path_two_cls_main, sep="\t", index_col=0)
        ecfp = pd.read_csv(self.path_prevout, sep="\t", index_col=0)

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

            for part in ["core", "sub1", "sub2"]:

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


    
class Curation_regression(Curation_Base):

    def __init__(self, bd, target):
        super().__init__(bd, target, direc=False)

        self.path_newhash      = os.path.join(bd, "ECFP/%s_newhash.tsv" %target)
        self.path_subdiff      = os.path.join(bd, "ECFP/%s.tsv" %target)
        


    def AssignNewHashBits(self):
        
        data = pd.read_csv(self.path_two_cls_main, sep="\t", index_col=0)
        df   = pd.read_csv(self.path_dellowlv, sep="\t", index_col=0)

        #data = pd.read_csv(self.path_curated_main, sep="\t", index_col=0)
        #df   = pd.read_csv(self.path_subdiff, sep="\t", index_col=0)

        WriteMsgLogStdout(self.log, "--- Assign new hash values to unfolded hash ---")
        WriteMsgLogStdout(self.log, "    # Loaded cpds : %d"%df.shape[0])

        prefix = ["core", "sub1", "sub2"]

        for suffix in [None]: #, "_forward", "_reverse"]:
            
            if suffix is not None:
               cols = [p + suffix for p in prefix]
            else:
                cols = prefix 

            # For core
            c        = cols[0]
            hash_c   = df[c].apply(lambda x:[int(h) for h in x.split(" ")])
            uhash_c  = np.sort(np.unique(hash_c.sum()))
            sr_c     = pd.Series(range(uhash_c.shape[0]), index=uhash_c)
            df[c]    = hash_c.apply(lambda x:" ".join([str(sr_c[i]) for i in x]))

            # For sub
            len_c    = sr_c.shape[0]

            s1       = cols[1]
            hash_s1  = df[s1].apply(lambda x:[int(h) for h in x.split(" ")])
            uhash_s1 = np.sort(np.unique(hash_s1).sum())

            s2       = cols[2]
            hash_s2  = df[s2].apply(lambda x:[int(h) for h in x.split(" ")])
            uhash_s2 = np.sort(np.unique(hash_s2).sum())
            
            uhash_s  = np.union1d(uhash_s1, uhash_s2)
            sr_s     = pd.Series(range(uhash_s.shape[0]), index=uhash_s)
            print("no moving")
            #sr_s     = sr_s + len_c # Add max of core hash for concatnation in MMPfingerprints
            df[s1]   = hash_s1.apply(lambda x:" ".join([str(sr_s[i]) for i in x]))
            df[s2]   = hash_s2.apply(lambda x:" ".join([str(sr_s[i]) for i in x]))           

        WriteMsgLogStdout(self.log, "    # calculated ecfp : %d"%df.shape[0])
        
        df.to_csv(self.path_newhash, sep="\t")
        self.path_prevout = self.path_newhash
        _, _ = self._NanCheck(data)


    def ApplySubdiff(self):
        """
        Concat ver
        """
        data = pd.read_csv(self.path_curated_main, sep="\t", index_col=0)
        df   = pd.read_csv(self.path_newhash, sep="\t", index_col=0)

        df_subdiff = pd.DataFrame(columns=df.columns)
        sr_overlap = [] 

        WriteMsgLogStdout(self.log, "--- Apply substructure difference ---")
        WriteMsgLogStdout(self.log, "    # Loaded cpds : %d"%df.shape[0])

        for i, sr in tqdm(df.iterrows()):
            sr_new, overlap = self._subdiff(sr)
            df_subdiff      = df_subdiff.append(sr_new, ignore_index=True)
            sr_overlap.append(overlap)
        
        df_subdiff["overlap"] = sr_overlap
        df_subdiff.index = df.index
        df_subdiff.to_csv(self.path_subdiff, sep="\t")
        self.path_prevout = self.path_subdiff
        _, _ = self._NanCheck(data)


    def _subdiff(self, sr):

        sub1_wodirec   = [int(i) for i in sr["sub1"].split(" ")]
        sub2_wodirec   = [int(i) for i in sr["sub2"].split(" ")]

        flag_s1 = ~np.isin(sub1_wodirec, sub2_wodirec)
        flag_s2 = ~np.isin(sub2_wodirec, sub1_wodirec)

        features_overlap = list(compress(sr["sub1"].split(" "), ~flag_s1))

        for idx in sr.index:
            
            if ("sub1" not in idx) and ("sub2" not in idx):
                continue

            elif "sub1" in idx:
                flag = flag_s1

            elif "sub2" in idx:
                flag = flag_s2
            
            features_subdiff = list(compress(sr[idx].split(" "), flag))
            
            sr[idx] = " ".join(features_subdiff)
            
        overlap = " ".join(features_overlap)

        return sr, overlap

class Curation_activebit(Curation_Base):

    def __init__(self, bd, target, direc):
        super().__init__(bd, target, direc=direc)

        self.path_subdiff = os.path.join(bd, "ECFP/%s_wsubdiff.tsv" %target)
        self.path_newhash = os.path.join(bd, "ECFP/%s_newhash.tsv" %target)
        self.path_aconly  = os.path.join(bd, "ECFP/%s_aconly.tsv" %target)

    def _collect_ACbit(self, data, ecfp, cols):

        idx_ac = data[data["class"]==1].index
        ecfp_ac = ecfp.loc[idx_ac, :]

        acbit = []

        for i, item in ecfp_ac.iterrows():

            for col in cols:
                acbit += [int(hashstring) for hashstring in item[col].split(" ")]
        

        acbit = np.unique(acbit)

        return acbit

    def RemoveNonACBits(self, path_input):

        data = pd.read_csv(self.path_curated_main, sep="\t", index_col=0)
        ecfp = pd.read_csv(path_input, sep="\t", index_col=0)

        acbits_c = self._collect_ACbit(data, ecfp, ["core"])
        acbits_s = self._collect_ACbit(data, ecfp, ["sub1", "sub2"])

        nbits = "unfold"

        WriteMsgLogStdout(self.log, "--- Delete NonAC bits ---")
        WriteMsgLogStdout(self.log, "    # Loaded cpds    : %d"%data.shape[0])

        cols = ecfp.columns[3:]

        ecfp_dellow = pd.DataFrame(columns=ecfp.columns)

        for i, item in tqdm(ecfp.iterrows()):
            
            new_item = []

            dict_flag = {}

            for part in ["core", "sub1", "sub2"]:

                hashes = [int(hashstring) for hashstring in item[part].split(" ")] 
                
                if part == "core":
                    acbits = acbits_c
                
                else:
                    acbits = acbits_s
                
                flag = np.array([True if h in acbits else False for h in hashes])
                    
                dict_flag[part] = flag

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

        ecfp_dellow.index = data.index
        ecfp_dellow.to_csv(self.path_aconly, sep="\t")



def Run(tname, type_prediction, diameter, deletelv):

    bd    = './Dataset/' #"/home/tamura/work/ACPredCompare/DataSets"

    if type_prediction == "acbits_only":
        bd = bd + "_ACbits"
        p = Curation_activebit(bd, tname, direc=True)
        p.RemoveOneClassMMS()
        p.CalcECFP("unfold", diameter=diameter)
        p.DeleteLowLevel(deletelv=deletelv)
        p.RemoveNonACBits()
        p.AssignNewHashBits()
        p.ApplySubdiff_concat()
        p.Close_log

    elif type_prediction == "classification":
        
        p = Curation_classification(bd, tname, direc=True)
        p.RemoveOneClassMMS()
        p.CalcECFP("unfold", diameter=diameter)
        p.DeleteLowLevel(deletelv=deletelv)
        p.AssignNewHashBits()
        p.ApplySubdiff_concat()
        p.Close_log

    elif type_prediction == "regression":

        bd = bd + "_regression"
        p = Curation_regression(bd, tname)
        p.RemoveOneClassMMS()
        p.CalcECFP("unfold", diameter=diameter)
        p.DeleteLowLevel(deletelv=deletelv)
        p.AssignNewHashBits()
        p.ApplySubdiff()
        p.Close_log

    else:
        raise ValueError("type_prediction must be classification or regression.")  

if __name__=="__main__":

    tlist = pd.read_csv('./Dataset/target_list.tsv', sep='\t', index_col=0)
    
    for i, sr in tlist.iterrows():

        tname = sr['target']
        type  = "classification"
        
        if tname+'.tsv' in os.listdir('./Dataset/Data'):
            continue

        Run(tname, type, 4, 0)
    

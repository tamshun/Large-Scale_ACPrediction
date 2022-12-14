import pandas as pd
import numpy as np
import os
import joblib
from collections import namedtuple
from sklearn.model_selection import StratifiedShuffleSplit
import random
from importlib_metadata import functools

LOCOCIdx = namedtuple("LOCOIndex", ("tsidx", "tridx"))
def LeaveOneCoreOut(df):
    
    LOCO = dict()
    
    cores = np.unique(df["core_id"])
    
    group = df.groupby("core_id")
    
    for cid in cores:
        
        tsidx = group.groups[cid]
        tridx = [idx for idx in df.index if idx not in tsidx]
        
        LOCO[cid] = LOCOCIdx(tsidx, tridx)
    
    return LOCO

def DelTestCpdFromTrain(df_ts, df_tr, id1 = "funatsu_lab_id1", id2 = "funatsu_lab_id2", deltype="both", biased_check=True):

    cpdidlist = list(set(df_ts[id1].tolist() + df_ts[id2].tolist()))

    delidx = []

    for idx in df_tr.index:

        if deltype=="both":
            if df_tr.loc[idx, id1] in cpdidlist or df_tr.loc[idx,id2] in cpdidlist:
                delidx.append(idx)
        
        elif deltype=="left":
            if df_tr.loc[idx, id1] in cpdidlist:
                delidx.append(idx)

        elif deltype=="right":
            if df_tr.loc[idx, id2] in cpdidlist:
                delidx.append(idx)
    
    Useidx = [i for i in df_tr.index if i not in delidx]
    df_tr = df_tr.loc[Useidx,:]
    
    if biased_check:
        df_tr = BiasedCheck(df_tr)
        
    return df_tr

def BiasedCheck(df):
    
    DelIdx = list()
    
    group = df.groupby("core_id")
    
    for cid in np.unique(df["core_id"]):
        df_c = group.get_group(cid)
    
        if len(set(df_c["class"]))==1:
            DelIdx += [int(idx) for idx in group.groups[cid]]
    
    UseIdx = [idx for idx in df.index if idx not in DelIdx]
    
    df = df.loc[UseIdx, :]
    
    return df

TrainTestIdx = namedtuple("TrainTestIndex", ("tsidx", "tridx"))
def MultipleTrainTestSplit(df, n_dataset=3):
    
    sss = StratifiedShuffleSplit(n_splits=n_dataset, test_size=0.3, random_state=0)
    trtssplit = dict()
    
    i = 0
    for tridx, tsidx in sss.split(X=np.zeros(df.shape[0]), y=df['class']):
        trtssplit[i] = TrainTestIdx(df.index[tsidx], df.index[tridx])
        i+=1
    
    return trtssplit

AXVIdx = namedtuple('AXVIndex', ('tridx', 'compound_out', 'both_out'))
def MultipleAXVSplit(df, seeds):
    
    axvsplit = dict() 
    
    for i in seeds:
        axv = AXV_generator(df, seed=i)
        axvsplit[i] = AXVIdx(axv.get_subset_idx('train'),
                             axv.get_subset_idx('compound_out'),
                             axv.get_subset_idx('both_out')
                             )
        
    return axvsplit

class AXV_generator():
    
    def __init__(self, data, keep_out_rate=0.2, seed=0, cols=['chembl_cid1', 'chembl_cid2']) -> None:
        self.data = data
        self.rate = keep_out_rate
        self.seed = seed
        self.cols = cols
        self.whole_cpds = np.union1d(np.unique(data[cols[0]]),np.unique(data[cols[1]])).tolist()
        
        self.keepout = self._get_keepout()
        self.identifier = self._set_identifier()
        
         
    def _get_keepout(self):
        random.seed(self.seed)
        sample_size = int(len(self.whole_cpds)*self.rate)
        keep_out = random.sample(self.whole_cpds, sample_size)
        
        return keep_out
    
    def _identifier(self, mmp:pd.Series):
    
        cpd1 = mmp[self.cols[0]]
        cpd2 = mmp[self.cols[1]]
        
        isin_cpd1 = cpd1 in self.keepout
        isin_cpd2 = cpd2 in self.keepout
        
        if (isin_cpd1==False) and (isin_cpd2==False):
            return 0
        
        elif (isin_cpd1==True) and (isin_cpd2==True):
            return 2
        
        elif (isin_cpd1==True) or (isin_cpd2==True):
            return 1
        
    def _set_identifier(self):
        return [self._identifier(sr) for i, sr in self.data.iterrows()]
    
    def get_subset_idx(self, name):
        
        if name.lower() == 'train':   
            mask = [True if i==0 else False for i in self.identifier]
            
        elif name.lower() == 'compound_out':
            mask = [True if i==1 else False for i in self.identifier]
            
        elif name.lower() == 'both_out':
            mask = [True if i==2 else False for i in self.identifier]
            
        return mask           
         
def RestoreFPfromstr(fp, bits):
    
    if isinstance(fp, str):
        bit = [int(i) for i in fp.split(" ") if len(i) != 0]

    elif fp is None:
        bit = []
    
    elif np.isnan(fp):
        bit = []

    else:
        bit = [int(fp)]

    fp_vector  = np.zeros((bits), dtype=int)     
    for i in bit :
        fp_vector[i] = 1
    
    return fp_vector

def GenerateFpsArray(sr_fp, nbits):
    
    fps = np.array([RestoreFPfromstr(str_fp,bits=nbits) for str_fp in sr_fp])

    return fps

def SetOffCommonBit(Xsub1, Xsub2):
    
    Xsub = Xsub1 + Xsub2
    
    commonbit = np.where(Xsub>1)
    
    for row, col in zip(commonbit[0], commonbit[1]):
        
        Xsub1[row, col] = 0
        Xsub2[row, col] = 0
        
    return Xsub1, Xsub2

class Hash2Bits():
    
    def __init__(self, subdiff=True, sub_reverse=False):

        self.subdiff = subdiff
        self.sub_rev = sub_reverse

    def GetMMPfingerprints_DF(self, X, Y=None, nbits=4096):

        c, s1, s2 = X

        if isinstance(nbits, list):
            nbits_c,  nbits_s = nbits
        else:
            nbits_c = nbits_s = nbits

        Xcore = GenerateFpsArray( c, nbits_c)
        Xsub1 = GenerateFpsArray(s1, nbits_s)
        Xsub2 = GenerateFpsArray(s2, nbits_s)
        
        if self.subdiff:
            Xsub1, Xsub2 = SetOffCommonBit(Xsub1, Xsub2)
        
        if not self.sub_rev:
            Xsub = np.hstack((Xsub1, Xsub2))
        else:
            Xsub = np.hstack((Xsub2, Xsub1))
        
        flag       = np.full(Xcore.shape[0], True, dtype=bool)
        zero       = np.where(np.sum(Xsub, axis=1)==0)[0]
        flag[zero] = False

        Xcore      = Xcore[flag, :]
        Xsub       = Xsub[flag, :]
        
        X = np.hstack([Xcore, Xsub])

        if Y is not None:
            Y = Y.values.reshape(-1)
            Y = Y[flag]

            return X, Y, flag

        else:
            return X, flag


    def GetMMPfingerprints_DF_SubOnly(self, X, Y=None, nbits=4096):

        s1, s2 = X

        nbits_s = nbits

        Xsub1 = GenerateFpsArray(s1, nbits_s)
        Xsub2 = GenerateFpsArray(s2, nbits_s)
        
        if self.subdiff:
            Xsub1, Xsub2 = SetOffCommonBit(Xsub1, Xsub2)
        
        if not self.sub_rev:
            Xsub = np.hstack((Xsub1, Xsub2))
        else:
            Xsub = np.hstack((Xsub2, Xsub1))
        
        flag       = np.full(Xsub.shape[0], True, dtype=bool)
        zero       = np.where(np.sum(Xsub, axis=1)==0)[0]
        flag[zero] = False
        
        Xsub       = Xsub[flag, :]
        
        X = Xsub

        if Y is not None:
            Y = Y.values.reshape(-1)
            Y = Y[flag]

            return X, Y, flag

        else:
            return X, flag


    def GetMMPfingerprints_DF_unfold(self, df, cols, Y=None, nbits=None, overlap="delete"):
        """
        - Generate MMPfingerprints binary vectors from str hash in given pd.DataFrame
        - This is for unfolded fingerprints

        Args:
            df (pd.DataFrame)       : A dataframe stores str hash values
            cols (list)             : column names used for the fingerprints generation
            Y (array-like, optional): given Y values will become one dimentional vector. Defaults to None.
            nbits (int)             : This func needs specifing nbits. Find bit length with FindBitLength func before using.
            overlap (delete/concat) : An option to specify how overlaping sub features are treated.
                                         

        Note:
            Substructure difference should be applyed in data curation phase.
            This function do not do this operation

        Returns:
            X: fingerprints binary matrix for ML input
        """        
        
        if not (overlap == "delete") and not (overlap == "concat"):
            raise ValueError("Overlap option should be delete/concat, not %s." %overlap)

        print("    $ Overlap option is selected as %s" %overlap)

        c  = df[cols[0]]
        s1 = df[cols[1]]
        s2 = df[cols[2]]

        Xcore = GenerateFpsArray( c, nbits[0])
        Xsub1 = GenerateFpsArray(s1, nbits[1])
        Xsub2 = GenerateFpsArray(s2, nbits[1])
        
        Xsub  = (Xsub1 + Xsub2).astype(bool).astype(int)

        X = np.hstack([Xcore, Xsub])

        if overlap == "concat":
            o = df[cols[3]]
            Xoverlap = GenerateFpsArray(o, nbits[1])

            X = np.hstack([X, Xoverlap])

        if Y is not None:
            Y = Y.values.reshape(-1)

            return X, Y

        else:
            return X
        
        
    def GetSeparatedfingerprints_DF_unfold(self, df, cols, Y=None, nbits=None, overlap="delete"):
        """
        - Generate MMPfingerprints binary vectors for 3 parts from str hash in given pd.DataFrame
        - This is for unfolded fingerprints

        Args:
            df (pd.DataFrame)       : A dataframe stores str hash values
            cols (list)             : column names used for the fingerprints generation
            Y (array-like, optional): given Y values will become one dimentional vector. Defaults to None.
            nbits (int)             : This func needs specifing nbits. Find bit length with FindBitLength func before using.
            overlap (delete/concat) : An option to specify how overlaping sub features are treated.
                                         

        Note:
            Substructure difference should be applied in data curation phase.
            This function do not do this operation

        Returns:
            X: fingerprints binary matrix for ML input
        """        
        
        if not (overlap == "delete") and not (overlap == "concat"):
            raise ValueError("Overlap option should be delete/concat, not %s." %overlap)

        print("    $ Overlap option is selected as %s" %overlap)

        c  = df[cols[0]]
        s1 = df[cols[1]]
        s2 = df[cols[2]]

        Xcore = GenerateFpsArray( c, nbits[0])
        Xsub1 = GenerateFpsArray(s1, nbits[1])
        Xsub2 = GenerateFpsArray(s2, nbits[1])

        #X = np.hstack([Xcore, Xsub])

        if overlap == "concat":
            o = df[cols[3]]
            Xoverlap = GenerateFpsArray(o, nbits[1])

            X = np.hstack([X, Xoverlap])

        Y = Y.values.reshape(-1)

        return {'core':Xcore, 'sub1':Xsub1, 'sub2':Xsub2}, Y

def FindBitLength(df, cols):

    df = df[cols]

    hashes_all = []

    for col in cols:
        df[col] = df[col].apply(lambda x:[int(h) for h in x.split(" ")] if isinstance(x, str) else [-1])
        hashes_all += df[col].sum()

    return np.max(hashes_all) + 1


class Initialize():
    
    def __init__(self, modeltype, dir_log=None, dir_score=None):
        
        self.mtype = modeltype
        self.logdir, self.scoredir, self.modeldir = self._MakeLogDir(dir_log, dir_score)  


    def _MakeLogDir(self, logdir, scoredir):
        
        if logdir is None:
            logdir   = "./Results/"
        
        if scoredir is None:
            scoredir = "./Scores/"
        
        modeldir = os.path.join(logdir, "Models")

        os.makedirs(logdir  , exist_ok=True)
        os.makedirs(scoredir, exist_ok=True)
        os.makedirs(modeldir, exist_ok=True)
        
        return logdir, scoredir, modeldir
    
        
    def run(self, target, load_params=False, debug=False):
        
        print("\n----- %s is proceeding -----\n" %target)
        
        self.debug = debug
        
        self._ReadDataFile(target)
        
        if load_params:
            self._SetParams(target)
            self._AllMMSPred_Load(target)
            
        else:
            
            if self._IsPredictableSet():
                self._SetParams(target)
                self._AllMMSPred(target)
                
            else:
                print('    $ %s is skipped because of lack of the actives' %target) 
                

    # def run_parallel(self, target_list, njob=-1):
    #     result = joblib.Parallel(n_jobs=njob)(joblib.delayed(self.run)(target) for target in target_list)
    
        
    def run_parallel(self, target_list, load_params=False, njob=-1):
        
        if load_params:
            run = functools.partial(self.run, load_params=load_params)
            result = joblib.Parallel(n_jobs=njob, backend='loky')(joblib.delayed(run)(target) for target in target_list)
        else:
            result = joblib.Parallel(n_jobs=njob, backend='loky')(joblib.delayed(self.run)(target) for target in target_list)
            
            
    def _IsPredictableTarget(self):
        data = self.main
        
        bool_nsample = bool(data.shape[0] > 50)
        bool_nmms    = bool(np.unique(data['core_id']).shape[0] > 1)
        flag         = bool(bool_nsample * bool_nmms)
        
        if flag:
            gbr         = data.groupby(['core_id', 'class'])
            n_mms       = gbr.count().index.max()[0] + 1
            n_class     = gbr.count().index.shape[0]
            bool_nclass = bool((n_class - n_mms) >= 2)
            
            flag = bool(flag * bool_nclass)
        
        return flag
    
    
    def _IsPredictableSeries(self, tr, ts, min_npos):
        '''
        analyse if given series is predictable or not.
        
        Requirement
        - tr/ts have at least one data
        - tr has positive sample at least the number of cv so that each cv set has at least one positive sample  
        '''
        n_tr = bool(tr.shape[0] > 0)
        n_ts = bool(ts.shape[0] > 0) 
        n_pos_tr = bool(tr[tr['class']==1].shape[0] >= min_npos)
        
        return bool(n_tr * n_ts * n_pos_tr)         
    
    def _IsPredictableSet(self):
        bool_nac = np.where(self.main['class']==1)[0].shape[0] > 1
        return bool_nac
    
    
    def _ReadDataFile(self, target):
        pass
    
    
    def _SetParams(self, target):
        pass
    
                    
    def _SetML(self):
        
        '''
        A model object is needed to run this code.
        You should set a model you use in your main class that inherits this class.
        '''
        pass
    
    
    def _AllMMSPred(self, target):
        
        '''
        main scripts
        '''
        pass
    
    def _AllMMSPred_Load(self, target):
        pass
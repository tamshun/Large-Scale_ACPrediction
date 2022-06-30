# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 14:09:59 2020

@author: Tamura
"""

#%%
from importlib_metadata import functools
import pandas as pd
import numpy as np
import os
import joblib
from collections import namedtuple
from sklearn.model_selection import StratifiedShuffleSplit
from rdkit import Chem
import torch
from torch import nn
import random


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
    
    def get_subset(self, name):
        
        if name.lower() == 'train':   
            mask = [True if i==0 else False for i in self.identifier]
            
        elif name.lower() == 'compound_out':
            mask = [True if i==1 else False for i in self.identifier]
            
        elif name.lower() == 'both_out':
            mask = [True if i==2 else False for i in self.identifier]
            
        return self.data.loc[mask, :]       
         
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

class Base_wodirection():
    
    def __init__(self, modeltype, dir_log=None, dir_score=None, aconly=False, data_split_metric='LOCO'):
        #self.bd      = "/home/tamura/work/Interpretability"
        #os.chdir(self.bd)
        self.mtype        = modeltype
        self.debug        = False        
        self.aconly       = aconly
        self.trtssplit    = data_split_metric
        self.col          = ["core", "sub1", "sub2", "overlap"]
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


    def _SetParams(self):

        self.nbits_c = FindBitLength(self.ecfp, [self.col[0]])
        self.nbits_s = FindBitLength(self.ecfp, self.col[1:] )
        
        self.nepoch  = self._Setnepoch()
        
        if self.trtssplit == 'LOCO':
            # Leave One Core Out
            self.data_split_generator = LeaveOneCoreOut(self.main)
            self.testsetidx           = self.data_split_generator.keys()
            self.del_leak             = True
        
        elif self.trtssplit == 'trtssplit':
            # Stratified Shuffled split
            self.data_split_generator = MultipleTrainTestSplit(self.main, n_dataset=3)
            self.testsetidx           = self.data_split_generator.keys()
            self.del_leak             = False


    def _Setnepoch(self):
        
        if self.debug:
            nepoch = [2, 2, 2]
        else:
            nepoch = [50, 100, 100]
            
        return nepoch
    
        
    def _ReadDataFile(self, target, acbits=False):

        if acbits:
            main = pd.read_csv("./Dataset_ACbits/Data/%s.tsv" %target, sep="\t", index_col=0) # Single class MMSs were removed in curation.py.
            ecfp = pd.read_csv("./Dataset_ACbits/ECFP/%s.tsv" %target, sep="\t", index_col=0)
        else:    
            main = pd.read_csv("./Dataset/Data/%s.tsv" %target, sep="\t", index_col=0) # Single class MMSs were removed in curation.py.
            ecfp = pd.read_csv("./Dataset/ECFP/%s.tsv" %target, sep="\t", index_col=0)

        return main, ecfp


    def _GetMatrices(self, cid, aconly=False):
        '''
        Wrapper function of TrainTestSplit_main, TrainTestSplit_ecfp, Hash2Bits 
        '''
        tr, ts = self._TrainTestSplit_main(cid, aconly)
        
        df_trX, df_tsX = self._TrainTestSplit_ecfp(tr, ts)
        
        df_trY, df_tsY, trX, trY, tsX, tsY = self._Hash2Bits(tr, ts, df_trX, df_tsX)

        return tr, ts, df_trX, df_trY, df_tsX, df_tsY, trX, trY, tsX, tsY
        
    
    def _TrainTestSplit_main(self, cid, aconly):
        
        generator = self.data_split_generator[cid]
        tr        = self.main.loc[generator.tridx,:]
        ts        = self.main.loc[generator.tsidx,:]
        
        # Check leak cpds
        if self.del_leak:
            tr = DelTestCpdFromTrain(ts, tr, id1='chembl_cid1', id2='chembl_cid2', deltype="both", biased_check=False)
        
        # Assign pd_sr of fp; [core, sub1, sub2]
        if aconly:
            print(    "Only AC-MMPs are used for making training data.\n")
            tr = tr[tr["class"]==1]
            
        return tr, ts
    
    def _TrainTestSplit_ecfp(self, tr, ts):
        
        df_trX = self.ecfp.loc[tr.index, :]
        df_tsX = self.ecfp.loc[ts.index, :]
        
        return df_trX, df_tsX
    
    def _Hash2Bits(self, tr, ts, df_trX, df_tsX):
        
        df_trY = tr["class"]
        df_tsY = ts["class"]

        forward  = Hash2Bits(subdiff=False, sub_reverse=False)
        trX, trY = forward.GetMMPfingerprints_DF_unfold(df=df_trX, cols=self.col, Y=df_trY, nbits=[self.nbits_c, self.nbits_s], overlap="concat")
        tsX, tsY = forward.GetMMPfingerprints_DF_unfold(df=df_tsX, cols=self.col, Y=df_tsY, nbits=[self.nbits_c, self.nbits_s], overlap="concat")

        return df_trY, df_tsY, trX, trY, tsX, tsY
    
    
    def _GetMatrices_3parts(self, cid, aconly=False):
        '''
        Wrapper function of TrainTestSplit_main, TrainTestSplit_ecfp, Hash2Bits 
        '''
        tr, ts = self._TrainTestSplit_main(cid, aconly)
        
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
    
    
    def run(self, target, debug=False, onlyfccalc=False):
        
        print("\n----- %s is proceeding -----\n" %target)
        
        if debug:
            self.debug=True
        
        self.main, self.ecfp = self._ReadDataFile(target, acbits=self.aconly)
        
        if self._IsPredictableSet():
            self._SetParams()
            self._AllMMSPred(target)
            
        else:
            print('    $ %s is skipped because of lack of the actives' %target)
        
        
    def run_parallel(self, target_list, njob=-1):
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
    
    
def torch2numpy(x):
    return x.to("cpu").detach().numpy().copy()

def onek_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]

ELEM_LIST = list(range(1,119))
ATOM_FDIM, BOND_FDIM = len(ELEM_LIST) + 21, 11

def atom_features(atom):
    return onek_encoding_unk(atom.GetAtomicNum() , ELEM_LIST) + onek_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5])+ onek_encoding_unk(atom.GetFormalCharge(), [-1, -2, 1, 2, 0]) + onek_encoding_unk(int(atom.GetChiralTag()), [0, 1, 2, 3])+onek_encoding_unk(int(atom.GetHybridization()),[
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ])+[1 if atom.GetIsAromatic() else 0]  

def bond_features(bond):
    bt = bond.GetBondType()
    stereo = int(bond.GetStereo())
    fbond = [bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC, bond.IsInRing()]
    fstereo = onek_encoding_unk(stereo, [0,1,2,3,4,5])
    fbond=fbond + fstereo
    return fbond

def get_atom(graph, atom_num):
    for atom in graph.GetAtoms():
        if atom.GetIdx() == atom_num:
            return atom
        else:
            continue

def index_select_ND(source: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """
    Selects the message features from source corresponding to the atom or bond indices in index.
    :param source: A tensor of shape (num_bonds, hidden_size) containing message features.
    :param index: A tensor of shape (num_atoms/num_bonds, max_num_bonds) containing the atom or bond
    indices to select from source.
    :return: A tensor of shape (num_atoms/num_bonds, max_num_bonds, hidden_size) containing the message
    features corresponding to the atoms/bonds specified in index.
    """
    # source = source.long()
    source     = source.float()
    index      = index.long()
    index_size = index.size()
    suffix_dim = source.size()[1:]
    final_size = index_size + suffix_dim
    target     = source.index_select(dim=0, index=index.view(-1))
    target     = target.view(final_size)
    
    return target
class MolGraph:
    """
    A MolGraph represents the graph structure and featurization of a single molecule.
    A MolGraph computes the following attributes:
    - smiles: Smiles string.
    - n_atoms: The number of atoms in the molecule.
    - n_bonds: The number of bonds in the molecule.
    - f_atoms: A mapping from an atom index to a list atom features.
    - f_bonds: A mapping from a bond index to a list of bond features.
    - a2b: A mapping from an atom index to a list of incoming bond indices.
    - b2a: A mapping from a bond index to the index of the atom the bond originates from.
    - b2revb: A mapping from a bond index to the index of the reverse bond.
    """

    def __init__(self, smiles: str, args=None, role=None):
        """
        Computes the graph structure and featurization of a molecule.
        :param smiles: A smiles string.
        :param args: Arguments.
        """
        self.smiles = smiles
        self.n_atoms, self.n_bonds = 0, 0
        self.f_atoms, self.f_bonds = [], []
        self.a2b, self.b2a, self.b2revb = [], [], []
        
        mol = Chem.MolFromSmiles(smiles)
        self.n_atoms = mol.GetNumAtoms()
        for i, atom in enumerate(mol.GetAtoms()):
            
            self.f_atoms.append(atom_features(atom))
            
        self.f_atoms = [self.f_atoms[i] for i in range(self.n_atoms)]
        for _ in range(self.n_atoms):
            self.a2b.append([])

        for a1 in range(self.n_atoms):
            for a2 in range(a1 + 1, self.n_atoms):
                bond = mol.GetBondBetweenAtoms(a1, a2)

                if bond is None:
                    continue

                f_bond = bond_features(bond)
                self.f_bonds.append(f_bond)
                self.f_bonds.append(f_bond)
                b1 = self.n_bonds
                b2 = b1 + 1

                self.a2b[a2].append(b1)
                self.b2a.append(a1)
                self.a2b[a1].append(b2)
                self.b2a.append(a2)
                self.b2revb.append(b2)
                self.b2revb.append(b1)
                self.n_bonds += 2
            


class BatchMolGraph:
    """
    A BatchMolGraph represents the graph structure and featurization of a batch of molecules.
    A BatchMolGraph contains the attributes of a MolGraph plus:
    - smiles_batch: A list of smiles strings.
    - n_mols: The number of molecules in the batch.
    - atom_fdim: The dimensionality of the atom features.
    - bond_fdim: The dimensionality of the bond features (technically the combined atom/bond features).
    - a_scope: A list of tuples indicating the start and end atom indices for each molecule.
    - b_scope: A list of tuples indicating the start and end bond indices for each molecule.
    - max_num_bonds: The maximum number of bonds neighboring an atom in this batch.
    - b2b: (Optional) A mapping from a bond index to incoming bond indices.
    - a2a: (Optional): A mapping from an atom index to neighboring atom indices.
    """

    def __init__(self, mol_graphs: list):
        
        #self.args = args
        self.masks = []
        self.smiles_batch = []
        self.mol_graphs = mol_graphs

        self.atom_fdim = ATOM_FDIM
        self.bond_fdim = BOND_FDIM 

        self.n_atoms = 1
        self.n_bonds = 1

        f_atoms = [[0] * self.atom_fdim]
        f_bonds = [[0] * self.bond_fdim]
        self.a_scope = []
        self.b_scope = []

        a2b = [[]]
        b2a = [0]
        b2revb = [0]
        for mol_graph in mol_graphs:
            f_atoms.extend(mol_graph.f_atoms)
            f_bonds.extend(mol_graph.f_bonds)

            for a in range(mol_graph.n_atoms):
                a2b.append([b + self.n_bonds for b in mol_graph.a2b[a]])

            for b in range(mol_graph.n_bonds):
                b2a.append(self.n_atoms + mol_graph.b2a[b])
                b2revb.append(self.n_bonds + mol_graph.b2revb[b])

            self.a_scope.append((self.n_atoms, mol_graph.n_atoms))
            self.b_scope.append((self.n_bonds, mol_graph.n_bonds))
            self.n_atoms += mol_graph.n_atoms
            self.n_bonds += mol_graph.n_bonds
            self.smiles_batch.append(mol_graph.smiles)

        self.max_num_bonds = max(len(in_bonds) for in_bonds in a2b)

        self.f_atoms = torch.FloatTensor(f_atoms)
        self.f_bonds = torch.FloatTensor(f_bonds)
        self.a2b = torch.LongTensor([a2b[a] + [0] * (self.max_num_bonds - len(a2b[a])) for a in range(self.n_atoms)])

        self.b2a = torch.LongTensor(b2a)
        self.b2revb = torch.LongTensor(b2revb)
        self.b2b = None
        self.a2a = None
        

    def get_components(self):
        """
        Returns the components of the BatchMolGraph.
        :return: A tuple containing PyTorch tensors with the atom features, bond features, and graph structure
        and two lists indicating the scope of the atoms and bonds (i.e. which molecules they belong to).
        """
        return self.f_atoms, self.f_bonds, self.a2b, self.b2a, self.b2revb, self.a_scope, self.b_scope

    def get_b2b(self) -> torch.LongTensor:
        """
        Computes (if necessary) and returns a mapping from each bond index to all the incoming bond indices.
        :return: A PyTorch tensor containing the mapping from each bond index to all the incoming bond indices.
        """

        if self.b2b is None:
            b2b = self.a2b[self.b2a]

            revmask = (b2b != self.b2revb.unsqueeze(1).repeat(1, b2b.size(1))).long()
            self.b2b = b2b * revmask

        return self.b2b

    def get_a2a(self) -> torch.LongTensor:
        """
        Computes (if necessary) and returns a mapping from each atom index to all neighboring atom indices.
        :return: A PyTorch tensor containing the mapping from each bond index to all the incodming bond indices.
        """
        get_b2a = self.b2a.detach().numpy().tolist()

        if self.a2a is None:

            a2neia=[]
            for incoming_bondIdList in self.a2b:
                neia=[]
                for incoming_bondId in incoming_bondIdList:
                    neia.append(get_b2a[incoming_bondId])
                a2neia.append(neia)
            self.a2a=torch.LongTensor(a2neia)

        return self.a2a
    
class MPNEncoder(nn.Module):
    """A message passing neural network for encoding a molecule."""

    def __init__(self, args: dict, weight_seed=1):
        """Initializes the MPNEncoder.
        :param args: Arguments.
        :param atom_fdim: Atom features dimension.
        :param bond_fdim: Bond features dimension.
        """
        torch.cuda.manual_seed(weight_seed)
        torch.manual_seed(weight_seed)
        random.seed(weight_seed)
        np.random.seed(weight_seed)
        
        super(MPNEncoder,self).__init__()
        
        self.args     = args
        self.act_func = nn.ReLU()
        self.depth    = args['ConvNum']
        self.dim      = int(args['dim'])
        self.W_i      = nn.Linear(ATOM_FDIM, self.dim)
        self.W_o      = nn.Linear(self.dim*2, self.dim)
        
        w_h_input_size = self.dim + BOND_FDIM
        modulList      = [self.act_func, nn.Linear(w_h_input_size, self.dim)]
        
        for d in range(args['agg_depth']):
            modulList.extend([self.act_func, nn.Linear(self.dim, self.dim)])
       
        for i in range(args['ConvNum']):
            exec(f"self.W_h{i} = nn.Sequential(*modulList)")
            
        self_module = [nn.Linear(ATOM_FDIM, self.dim), self.act_func]
        for d in range(args['agg_depth']):
            self_module.extend([nn.Linear(self.dim, self.dim), self.act_func])
            
        self.W_ah = nn.Sequential(*self_module)


    def forward(self, mol_graph):
        """
        Encodes a batch of molecular graphs.
        :param mol_graph: A BatchMolGraph representing a batch of molecular graphs.
        :param features_batch: A list of ndarrays containing additional features.
        :return: A PyTorch tensor of shape (num_molecules, hidden_size) containing the encoding of each molecule.
        """

        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope = mol_graph.get_components()
        a2a = mol_graph.get_a2a()
        
        if self.args['cuda']:
            f_atoms, f_bonds, a2b, b2a, b2revb, a2a = f_atoms.cuda(), f_bonds.cuda(), a2b.cuda(), b2a.cuda(), b2revb.cuda(), a2a.cuda()
            self.W_i, self.W_o = self.W_i.cuda(), self.W_o.cuda()
            for i in range(self.depth-1):
                exec(f"self.W_h{i} = self.W_h{i}.cuda()")
                      
        input = self.act_func(self.W_i(f_atoms))

        self_message, message = input.clone(), input.clone()          
        self_message[0, :], message[0, :] = 0, 0
         
        for depth in range(self.depth):
            
            nei_a_message, nei_f_bonds = index_select_ND(message, a2a), index_select_ND(f_bonds, a2b)
            nei_message = torch.cat([nei_a_message, nei_f_bonds], dim=2)
            message = nei_message.sum(dim=1).float()
            
            message = eval(f"self.W_h{depth}(message)")
            message = self_message + message
            self_message = message.clone()
            message[0 , :] = 0
        
        nei_a_message = index_select_ND(message, a2a)
        a_message = nei_a_message.sum(dim=1).float()
        cc_message = self.W_ah(f_atoms)

        a_input = torch.cat([cc_message, a_message], dim=1)
        out = self.act_func(self.W_o(a_input))

        mol_vecs = []
        for i, (a_start, a_size) in enumerate(a_scope):
            cur_hiddens = out.narrow(0, a_start, a_size)
            mol_vec = cur_hiddens.sum(dim=0)
            mol_vecs.append(mol_vec)
        out = torch.stack(mol_vecs, dim=0)
        
        return out
class Base_wodirection_CGR():
    
    def __init__(self, target, modeltype, dir_log=None, dir_score=None, aconly=False, data_split_metric='LOCO'):
        #self.bd      = "/home/tamura/work/Interpretability"
        #os.chdir(self.bd)
        self.target       = target
        self.mtype        = modeltype
        self.debug        = False        
        self.aconly       = aconly
        self.trtssplit    = data_split_metric
        self.col          = 'CGR'
        self.logdir, self.scoredir, self.modeldir = self._MakeLogDir(dir_log, dir_score)  
        self.main         = self._ReadDataFile(target, acbits=self.aconly)

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


    def _SetParams(self):
        
        if self.trtssplit == 'LOCO':
            # Leave One Core Out
            self.data_split_generator = LeaveOneCoreOut(self.main)
            self.testsetidx           = self.data_split_generator.keys()
            self.del_leak             = True
        
        elif self.trtssplit == 'trtssplit':
            # Stratified Shuffled split
            self.data_split_generator = MultipleTrainTestSplit(self.main, n_dataset=3)
            self.testsetidx           = self.data_split_generator.keys()
            self.del_leak             = False

        
    def _ReadDataFile(self, target, acbits=False):

        if acbits:
            main = pd.read_csv("./Dataset_ACbits/CGR/%s.tsv" %target, sep="\t", index_col=0)
        else:    
            main = pd.read_csv("./Dataset/CGR/%s.tsv" %target, sep="\t", index_col=0)

        return main


    def _GetTrainTest(self, cid, aconly=False):
        '''
        Wrapper function of TrainTestSplit_main, TrainTestSplit_ecfp, Hash2Bits 
        '''
        tr, ts = self._TrainTestSplit(cid, aconly)

        return tr, ts
        
    
    def _TrainTestSplit(self, cid, aconly):
        
        generator = self.data_split_generator[cid]
        tr        = self.main.loc[generator.tridx,:]
        ts        = self.main.loc[generator.tsidx,:]
        
        # Check leak cpds
        if self.del_leak:
            tr = DelTestCpdFromTrain(ts, tr, id1='chembl_cid1', id2='chembl_cid2', deltype="both", biased_check=False)
        
        # Assign pd_sr of fp; [core, sub1, sub2]
        if aconly:
            print(    "Only AC-MMPs are used for making training data.\n")
            tr = tr[tr["class"]==1]
            
        return tr, ts
    
    
    def run(self, target, load_params=False, debug=False, onlyfccalc=False):
        
        self.tname = target
        print("\n----- %s is proceeding -----\n" %target)
        
        if debug:
            self.debug=True
        
        self.main, self.cgr = self._ReadDataFile(target, acbits=self.aconly)
        
        if load_params:
            self._SetParams()
            self._AllMMSPred_Load(target)
            
        else:
            
            if self._IsPredictableSet():
                self._SetParams()
                self.fixed_arg = self._set_fixedargs()
                self._AllMMSPred(target)
                
            else:
                print('    $ %s is skipped because of lack of the actives' %target)
        
        
    def run_parallel(self, target_list, load_params=False, njob=-1):
        
        if load_params:
            run = functools.partial(self.run, load_params=load_params)
            result = joblib.Parallel(n_jobs=njob, backend='loky')(joblib.delayed(run)(target) for target in target_list)
        else:
            result = joblib.Parallel(n_jobs=njob, backend='loky')(joblib.delayed(self.run)(target) for target in target_list)
        
        
    
    def _IsPredictableTarget(self):
        '''
        Check if the given target has...
        - at least 50 sample
        - more than 1 MMS
        - diversity
        '''
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
        '''
        Check if the given set has at least one ac data
        '''
        bool_nac = np.where(self.main['class']==1)[0].shape[0] > 1
        return bool_nac
                
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
        
        '''
        main scripts
        '''
        pass
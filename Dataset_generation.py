import pandas as pd
import numpy as np
import os
import shutil
from Tools.DataFrame import MakeKey
from Tools.pickle import ToPickle
from Chemistry.Property import CountHeavyatoms
from tqdm import tqdm

def GetOriginalFile():
    
    target_list = pd.read_csv('./Dataset/target_list.tsv', sep='\t', index_col=None)
    p = '/Volumes/funatsu-lab/Datasets/ChEMBL29-rdkitCuration/chemblx29-Ki/compounds/'
    
    for i, sr in target_list.iterrows():
        tid = sr['tid']
        tname = sr['target']
        fname = 'tid-%d-actives.txt'%tid
        dir_to = '/Users/tamura/work/ACPredCompare/Dataset/Original/%s/'%tname
        shutil.copy(p+fname, dir_to+'%s.txt'%tname)
        
        original = pd.read_csv(dir_to+'%s.txt'%tname, sep='\t', index_col=None)
        original[['funatsu_lab_id','non_stereo_aromatic_smieles']].to_csv(dir_to+'%s.smi'%tname, sep='\t', index=None)

def DelUnreliableChiral(df, target):

    #BackUp
    df.to_csv("./OriginalMMPFile/%s/%s_original.txt" %(target,target), sep="\t", index=None)

    grouped = df.groupby("non_stereo_aromatic_smieles")
        
    n_Chiral = grouped.size()
    
    df_std = grouped.std()
    df_std = df_std.loc[n_Chiral.index[n_Chiral>=2],:]
    
    delidx = [] 
    
    for smi in df_std.index:
        if df_std.loc[smi,"pot.(log,Ki)"] > 1:
            delidx += list(grouped.get_group(smi).index)
    
    idx = [i for i in df.index if i not in delidx]
    df = df.loc[idx,:]
    
#    os.makedirs("./%s" %target, exist_ok=True)
#    df.to_csv("./%s/%s.txt" %(target,target), sep="\t")
    
    #Make log file
    f = open("./OriginalMMPFile/delID_%s.txt" %target, "w")
    for x in delidx:
        f.write(str(x) + "\n")
    f.close()

    return df

def RunMartinMMP(path):
    """
    - Execute MMS generation.
    - Parameter c, n, and m are set. 
        cut type (-c) : single
        number of cuts (-n) : 1
        mode (-m) : series
    - The remaining parameters are same as default
    - If you want to try other parameters, change the settings in commands using Java grammar
    """
    fpath = os.path.abspath(path)
    
    d = os.getcwd()
    os.chdir("Z:/MMP_v8.1_refactored/")
    command = "java -jar MMP.jar %s -c single -n 1 -m mmp" %fpath
#    subprocess.run([command], shell=True)
    os.system(command)
    os.chdir(d)

def GenerateSmiAndGetSmiPath(df, target, fdir):
    
    op =  "%s/%s.smi"%(fdir,target)
    
    df = df[["funatsu_lab_id", "non_stereo_aromatic_smieles"]]
    df.to_csv(op, index=False, sep = "\t")
    
    return op

#def MoveMMPFile(target, pot_kind):
#
#    pathfrom = "Z:/MMP_v8.1_refactored/data/ChEMBLE24/%s/MMP/%s-%s.unique.mmp-1.single.txt"%(target, target, pot_kind)
#    pathto   = "Z:/ActivityCliff/DataSets/OriginalMMP-%s/%s-%s.unique.mmp-1.single.txt"%(pot_kind, target, pot_kind)
#    
#    copy(pathfrom, pathto)
    
    
def DelNonActive(df, thres):
    """
    - Drop structures whose potencies are less than 10 micro M
    """
    df = df[df["pot.(nMol,Ki)"]<thres]
    return df

def main(targetid, target, pot_kind, filedir, pot_thres=None):
    
    df = pd.read_table("%s/%s.txt" %(filedir,target))
    df = DelUnreliableChiral(df, target)

    if pot_thres!=None:
        df = DelNonActive(df, pot_thres)

    path_smi = GenerateSmiAndGetSmiPath(df, target, filedir)
    RunMartinMMP(path_smi)      
    


def SplitandJoin(df0):
    
    """
    -Edit the original MMP file given by Martin's program.
    1. Split the whole strs with dilimeter of '|', then get df including sub_change, core, and funatsu_lab_ids
    2. Split strs in sub_change columns with dilimeter of '>>', then get df of sub_smiles
    3. Concatenate these df and drop the sub_change column.
    """
    
    df = df0.iloc[:,0].str.split("|", expand=True)
    df = df.rename(columns={0:"a"})
    
    df = pd.concat([df, df.iloc[:,0].str.split('>>', expand=True)], axis=1).iloc[:,1:]
    df.columns = ["core","funatsu_lab_id1","funatsu_lab_id2","sub1","sub2"]
    
    df = df.loc[:,["funatsu_lab_id1","funatsu_lab_id2","core","sub1","sub2"]]
    df.iloc[:,:2] = df.iloc[:,:2].applymap(lambda x:int(x))
    
    df = MakeKey(df, df.columns[0], df.columns[1])
    
    return df

def SortbyPot(df):
    
    """
    - Sort each pair of info by logKi values in descending order.(high pot one will be assigned in left column)
    - ex) pair of info : sub1/sub2, funatsu_lab_id1/funatsu_lab_id2
    """
    
    potcol = [c for c in df.columns if "pot" in c] # pot1/pot2
    idcol = [c for c in df.columns if "funatsu" in c] # id1/id2
    subcol = [c for c in df.columns if "sub" in c] # sub1/sub2
    
    poslist = np.argsort(df[potcol], axis=1) #compare the potency values of potcol
    
    for i in tqdm(range(df.shape[0])):
    
        for cols in [idcol, subcol, potcol]:
            
            pos = poslist.iloc[i,:].tolist()
            icols = [val for val in range(len(df.columns)) if df.columns[val] in cols]
            
            df.iloc[i,icols] = df[cols].iloc[i,pos].values
    
    return df

def DelOnlyRatom(df, return_delidx = False):
    
    """
    - Delete MMPs including a substructure whose smiles is only Ratom
    - After deleting, New index is assigned
    - if 'return_index' is true, this function returns df and dropped index (for dubuging)
    """
    
    smiles = df[["sub1", "sub2"]]
    
    smiles['sub1-sub2'] = smiles.min(axis=1) + " - " + smiles.max(axis=1)
    
    idx = [i for i in df.index[np.where(smiles["sub1-sub2"].str.startswith("[R1] -"))[0]]]
    
    df = df.drop(idx)
    df.index = range(df.shape[0])
    
    if return_delidx:
        return df, idx
    
    else:
        return df

def Assignpot(df, df_pot, pot):
    
    """
    - Search potency values, then assign to correspondng cell.
    """
    
    for i,col in enumerate(["pot1","pot2"]):
        
        idx = list()
        
        for j in range(df.shape[0]):
            
            idcol = "funatsu_lab_id%s" %(i+1)
            ID    = df[idcol][j]
            idx.append(df_pot.index[np.where(df_pot["funatsu_lab_id"]==ID)[0]][0])
        
        potcol = "pot.(log,%s)" %pot
        df[col] = df_pot.loc[idx, potcol].values
    
    return df

def AssignClassAndDelOthers(df, thres):
    
    """
    - Calculate potency difference as dpot according to pot1 and pot2
    - assigne each MMP to AC/NAC, following the cliff definision
    - Finally, MMPs with dpots between 1 and threshold are dropped.
    """
    
    col = "dpot"
    df[col] = abs(df["pot1"] - df["pot2"])
    
    clscol = "class" 
    df[clscol] = 0
    
    idx_c = df.index[np.where(df[col]>thres)[0]]
    df.loc[idx_c,clscol] = 1
    
    idx_nc = df.index[np.where(df[col]<1)[0]]
    df.loc[idx_nc,clscol] = -1
    
    df = df[df[clscol]!=0]
    df.index = range(df.shape[0])
    
    return df

def DropSmallMolecule(df):
    
    df_H = CountHeavyatoms(df,"core")
    idx = df_H[df_H>=10].index
    
    df = df.loc[idx,:]
    
    df.index = range(df.shape[0])
    
    return df

def AssignCoreid(df,thres):
    
    """
    - Assign core id to each MMP
    - MMPs which have same core id have a common core structure.
    - The smaller core id, the larger the number of cliffs in the MMP with the same core.
    """
    
    l_core = df["core"].drop_duplicates().tolist()
    
    rank = GetRankByCliff(df, l_core, thres)
    
    df["core_id"] = np.nan
    coreid = 0
    
    for l in rank:
        core = l_core[l]
        
        idx = df.index[np.where(df["core"]==core)[0]]
        df.loc[idx, "core_id"] = coreid
        
        coreid+=1
    
    return df

def GetRankByCliff(df, l_core, thres):
    
    n_c  = list()
    
    for core in l_core:
        
        df_c = df[df["core"]==core]
        df_counts = df_c["class"].value_counts()
        
        if df_counts.shape[0]==2:
            
            n_c.append(df_counts[1])
        
        elif df_counts.index[0]==1:
            n_c.append(df_counts[1])
        
        else:
            n_c.append(0)
    
    rank = np.argsort(n_c)[::-1]
    
    return rank

def main(df0, df_pot, pot, threshold):
    
    df = SplitandJoin(df0)
    df = DelOnlyRatom(df)
    #df, idx = DelOnlyRatom(df, return_delidx=True)
    df = Assignpot(df, df_pot, pot)
    df = SortbyPot(df)
    df = AssignClassAndDelOthers(df, threshold)
    df = DropSmallMolecule(df)
    df = AssignCoreid(df,threshold)
    
    return df #,idx

    
if __name__ == '__main__':
    GetOriginalFile()
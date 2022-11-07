import pandas as pd
import numpy as np
import os
from functools                         import partial
from sklearn.neighbors                 import KNeighborsClassifier
from collections                       import defaultdict
from BaseFunctions       import Hash2Bits, FindBitLength, funcTanimotoKernel_MMPKernel

def distance_func(x, y, len_c):
    
    return 1 - funcTanimotoKernel_MMPKernel(x, y, len_c=len_c)

class KNN():
    
    def __init__(self, target, ref_model, log_dir, distance=5):
        
        self.target  = target
        self.ref     = ref_model
        self.log_dir = log_dir
        self.k       = distance
        
        
    def ReadFiles(self, trial):
        data = pd.read_csv('./Dataset/Data/%s.tsv'%(self.target), sep='\t', index_col='id')
        fps  = pd.read_csv('./Dataset/ECFP/%s.tsv'%(self.target), sep='\t', index_col='id')
        
        if 'NN' in self.ref:
            log  = pd.read_csv('./Log_wodirection_trtssplit/%s/%s_trial%d_test.tsv'%(self.ref, self.target, trial), sep='\t', index_col='ids')
        else:    
            log  = pd.read_csv('./Log_wodirection_trtssplit/%s/%s_trial%d.tsv'%(self.ref, self.target, trial), sep='\t', index_col='ids')
        
        return data, fps, log
    
    
    def SingleTrial(self, trial):
        
        data, fps, log = self.ReadFiles(trial)
        len_c = FindBitLength(fps, cols=['core'])
        len_s = FindBitLength(fps, cols=['sub1', 'sub2', 'overlap'])
        
        # id_ts = log.index[:3]
        id_ts = log.index
        id_tr = np.setdiff1d(fps.index, id_ts)
        
        tr = data.loc[id_tr, :]
        ts = data.loc[id_ts, :]
        
        tr_fps = fps.loc[id_tr, :]
        ts_fps = fps.loc[id_ts, :]
        
        converter = Hash2Bits()
        trX = converter.GetMMPfingerprints_DF_unfold(tr_fps, cols=['core', 'sub1', 'sub2', 'overlap'], nbits=[len_c, len_s], overlap='concat')
        tsX = converter.GetMMPfingerprints_DF_unfold(ts_fps, cols=['core', 'sub1', 'sub2', 'overlap'], nbits=[len_c, len_s], overlap='concat')
        trY = data.loc[id_tr, 'class']
        tsY = data.loc[id_ts, 'class']
        
        
        dist_func = partial(distance_func, len_c=len_c)
        ml        = KNeighborsClassifier(n_neighbors=self.k, metric=dist_func, n_jobs=-1)
        print("    $  fit.\n") 
        ml.fit(trX, trY)
        # print("    $  predit for training set.\n") 
        # predY_tr  = ml.predict(trX)
        # proba_tr  = ml.predict_proba(trX)
        
        print("    $  predit for test set.\n") 
        predY_ts   = ml.predict(tsX)
        proba_ts   = ml.predict_proba(tsX)
        dist_neigh, idx_neigh = ml.kneighbors(tsX, return_distance=True)
        id_neigh   = ['; '.join([id_tr[i] for i in idx]) for idx in idx_neigh]
        dist_neigh = ['; '.join(d.astype(str)) for d in dist_neigh]        
        
        # Write & save log
        # log_tr = self.WriteLog_tr(trial, tr, trY, predY_tr, proba_tr)
        log_tr = None
        log_ts = self.WriteLog_ts(trial, tr, ts, tsY, predY_ts, proba_ts, id_neigh, dist_neigh)  
        self.Save(self.target, trial, log_tr, log_ts)
        print("    $  Log is out.\n")      
            
            
            
    def WriteLog_tr(self, loop, tr, trY, predY, proba):
        
        log = defaultdict(list)
        
        # Write log
        trids         = tr.index.tolist()
        log["ids"]   += trids
        log["loop"]  += [loop] * len(trids)
        log['#tr']   += [tr.shape[0]] * len(trids)
        log['#ac_tr']+= [tr[tr['class']==1].shape[0]] * len(trids)
        log["trueY"] += trY.tolist()
        log["predY"] += predY
        log["prob"]  += proba
        
        return log
    
    
    def WriteLog_ts(self, loop, tr, ts, tsY, predY, proba, id_neigh, dist_neigh):
        
        log = defaultdict(list)
        
        # Write log
        tsids            = ts.index.tolist()
        log["ids"]      += tsids
        log["loop"]     += [loop] * len(tsids)
        log['#tr']      += [tr.shape[0]] * len(tsids)
        log['#ac_tr']   += [tr[tr['class']==1].shape[0]] * len(tsids)
        log["trueY"]    += tsY.tolist()
        log["predY"]    += predY.tolist()
        log["prob"]     += [p[1] for p in proba]
        log['neighbor'] += id_neigh
        log['distance'] += dist_neigh
        
        return log
        
        
    def Save(self, target, loop, log_tr, log_ts):
        # path_log_tr = os.path.join(self.log_dir, "%s_trial%d_train.tsv" %(target, loop))
        # self.log_tr = pd.DataFrame.from_dict(log_tr)
        # self.log_tr.to_csv(path_log_tr, sep="\t")
        
        path_log_ts = os.path.join(self.log_dir, "%s_trial%d_test.tsv" %(target, loop))
        self.log_ts = pd.DataFrame.from_dict(log_ts)
        self.log_ts.to_csv(path_log_ts, sep="\t")
        
        
    def Run(self):
        
        for i in range(3):
            self.SingleTrial(i)
            

def main(ref_model, distance=5):
    
    df      = pd.read_csv('./Dataset/target_list.tsv', sep='\t', index_col=0)
    targets = df['chembl_tid'].loc[df['predictable_trtssplit']]
    log_dir = './Log_wodirection_trtssplit/%dNN/%s/'%(distance, ref_model)
    os.makedirs(log_dir, exist_ok=True)
    
    print('ref_model: %s\n'%ref_model)
    print('distance : %d\n'%distance)
    
    for target in targets:
        print('\n--- %s ---\n'%target)
        p = KNN(target, ref_model, log_dir, distance)
        p.Run()
        
        
def debug(ref_model, distance=5):
    
    df      = pd.read_csv('./Dataset/target_list.tsv', sep='\t', index_col=0)
    targets = df['chembl_tid'].loc[df['predictable_trtssplit']]
    log_dir = './Log_wodirection_trtssplit/%dNN/%s/'%(distance, ref_model)
    os.makedirs(log_dir, exist_ok=True)
    
    
    p = KNN(targets.iloc[0], ref_model, log_dir, distance)
    p.Run()            
            
if __name__ == '__main__':
    
    import sys
    args = sys.argv
    ref_model = args[1]
    distance  = int(args[2])
    # debug(ref_model, distance)
    main(ref_model, distance)
        
        
        
        
        
    

        
    
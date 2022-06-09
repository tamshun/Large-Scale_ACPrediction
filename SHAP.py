import pandas as pd
import numpy as np
import os
import shap
from Tools.ReadWrite import LoadJson
from Kernels.Kernel                    import funcTanimotoKernel_MMPKernel
from Fingerprint.Hash2BitManager       import Hash2Bits, FindBitLength
from sklearn.svm import SVC
from collections                         import defaultdict
from functools import partial
#from sklearnex                           import patch_sklearn

#patch_sklearn()


class SHAP():
    
    def __init__(self, target:str, model:str, trial:int) -> None:
        self.target = target
        self.model  = model
        self.trial  = trial
        
        self.data_dir  = './Dataset/Data/'
        self.fp_dir    = './Dataset/ECFP/'
        self.log_dir   = './Log_wodirection_trtssplit/'
        self.param_dir = self.log_dir + model + '/Models/' 
        
        self.col = ['core', 'sub1', 'sub2', 'overlap']       
     
        
    def _load(self):
        
        log   = pd.read_csv('./Log_wodirection_trtssplit/%s/%s_trial%d.tsv' %(self.model, self.target, self.trial), sep='\t', index_col='ids')
        
        data  = pd.read_csv(self.data_dir+'%s.tsv'%self.target, sep='\t', index_col='id')
        ecfp  = pd.read_csv(self.fp_dir+'%s.tsv'%self.target, sep='\t', index_col='id')
        
        return log, data, ecfp
    
    
    def _load_param(self):
        return LoadJson(self.param_dir + '%s_trial%d.json'%(self.target, self.trial))
    
    
    def _load_nn(self):
        return torch.load(self.param_dir + '%s_trial%d.pth'%(self.target, self.trial))
    
    
    def _bitlength(self):
        
        nbits_c = FindBitLength(self.ecfp, [self.col[0]])
        nbits_s = FindBitLength(self.ecfp, self.col[1:] )
        
        return nbits_c, nbits_s
    
    
    def _trtssplit(self):
        
        tsidx = self.log.index
        tridx = np.setdiff1d(self.data.index, tsidx)
        
        return tridx, tsidx
    
    
    def analyze(self):
        
        tridx, tsidx  = self._trtssplit()
        self.trX, self.trY, self.tsX = self._make_input(tridx, tsidx)
        ml            = self._reload_ml(self.trX, self.trY)
        ml_score      = self._select_scorefunc(ml)
        explainer     = self._init_shap(ml_score, self.trX)
        return explainer


class SHAP_SVM(SHAP):
    
    def __init__(self, target: str, trial: int) -> None:
        
        super().__init__(target=target, model='SVM', trial=trial)
        
        self.log, self.data, self.ecfp = self._load()
        self.param                     = self._load_param()
        self.len_c, self.len_s         = self._bitlength()
    
    
    def _make_input(self, tridx, tsidx):
        
        df_trX = self.ecfp.loc[tridx, :]
        df_trY = self.data.loc[tridx, 'class']
        df_tsX = self.ecfp.loc[tsidx, :]

        forward  = Hash2Bits(subdiff=False, sub_reverse=False)
        trX, trY = forward.GetMMPfingerprints_DF_unfold(df=df_trX, Y=df_trY, cols=self.col, nbits=[self.len_c, self.len_s], overlap="concat")
        tsX      = forward.GetMMPfingerprints_DF_unfold(df=df_tsX, cols=self.col, nbits=[self.len_c, self.len_s], overlap="concat")

        return trX, trY, tsX
    
    
    def _reload_ml(self, trX, trY):
        
        kernelf = partial(funcTanimotoKernel_MMPKernel, len_c=self.len_c)
        ml = SVC(kernel=kernelf, **self.param)
        ml.fit(trX, trY)
        
        return ml
    
    def _select_scorefunc(self, ml):
        return ml.decision_function
        
    def _init_shap(self, ml_score, trdata, proportion="average"):
        
        model = self.model.lower()
        
        # Kernel explaner will be used for non-tree-based model 
        explainer = shap.KernelExplainer(model = ml_score,
                                         data  = trdata,
                                         link  = "identity"
                                        )
                        
        return explainer
    
    
class SHAP_TreeBase(SHAP):
    
    def __init__(self, target: str, trial: int) -> None:
        
        super().__init__(target=target, model='XGBoost', trial=trial)
        
        self.log, self.data, self.ecfp = self._load()
        self.param                     = self._load_param()
        self.len_c, self.len_s         = self._bitlength()
    
    
    def _make_input(self, tridx, tsidx):
        
        df_trX = self.ecfp.loc[tridx, :]
        df_trY = self.data.loc[tridx, 'class']
        df_tsX = self.ecfp.loc[tsidx, :]

        forward  = Hash2Bits(subdiff=False, sub_reverse=False)
        trX, trY = forward.GetMMPfingerprints_DF_unfold(df=df_trX, Y=df_trY, cols=self.col, nbits=[self.len_c, self.len_s], overlap="concat")
        tsX      = forward.GetMMPfingerprints_DF_unfold(df=df_tsX, cols=self.col, nbits=[self.len_c, self.len_s], overlap="concat")

        return trX, trY, tsX
    
    
    def _reload_ml(self, trX, trY):
        
        if self.model == 'XGBoost':
            import xgboost as xgb
            ml = xgb.XGBClassifier(**self.param)
            ml.fit(trX, trY)

        elif self.model == 'Random_Forest':
            from sklearn.ensemble import RandomForestClassifier as rf
            ml = rf(**self.param)
            ml.fit(trX, trY)
        
        return ml
    
    
    def _select_scorefunc(self, ml):
        return ml.predict_proba
    
        
    def _init_shap(self, ml_score, trdata):
        
        # Kernel explaner will be used for non-tree-based model 
        explainer = shap.TreeExplainer(model         = ml_score,
                                       data          = trdata,
                                       model_output  = "probability"
                                      )
                        
        return explainer
    
    
from torch import nn
import torch

class FullyConnectedNN(nn.Module):
    
    def __init__(self, arg, random_seed=0):
        
        super(FullyConnectedNN, self).__init__()
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        np.random.seed(random_seed)
        
        hidden_list  = arg['node_list']
        W_list       = [nn.Linear(hidden_list[0], hidden_list[1], bias=True)]
        self.dropout = nn.Dropout(p=arg['drop_rate'])
        self.active_function = nn.ReLU()
        
        for num in range(len(hidden_list)-2):
            W_list.extend([self.active_function, self.dropout, nn.Linear(hidden_list[num+1], hidden_list[num+2], bias=True)])
                
        modulelist  =  nn.ModuleList(W_list) 
        self.W      =  nn.Sequential(*modulelist)

    def forward(self, x):
        return self.W(x)  
            
            
class Dataset(torch.utils.data.Dataset):
    
    def __init__(self, fpset, label):
        self.X = fpset
        self.y = label.reshape([label.shape[0],1])
        
    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        tensor_X = torch.FloatTensor(self.X[idx,:])
        tensor_y = torch.FloatTensor(self.y[idx])
        return tensor_X, tensor_y 
    
                
class SHAP_NN(SHAP):
    
    def __init__(self, target: str, trial: int) -> None:
        
        super().__init__(target=target, model='FCNN', trial=trial)
        
        self.log, self.data, self.ecfp = self._load()
        self.param                     = self._load_param()
        self.nn                        = self._load_nn()
        self.len_c, self.len_s         = self._bitlength()
    
    
    def _make_input(self, tridx, tsidx):
        
        df_trX = self.ecfp.loc[tridx, :]
        df_trY = self.data.loc[tridx, 'class']
        df_tsX = self.ecfp.loc[tsidx, :]

        forward  = Hash2Bits(subdiff=False, sub_reverse=False)
        trX, trY = forward.GetMMPfingerprints_DF_unfold(df=df_trX, Y=df_trY, cols=self.col, nbits=[self.len_c, self.len_s], overlap="concat")
        tsX      = forward.GetMMPfingerprints_DF_unfold(df=df_tsX, cols=self.col, nbits=[self.len_c, self.len_s], overlap="concat")

        return trX, trY, tsX
    
    
    def _reload_ml(self, trX, trY):
        
        ml = FullyConnectedNN(**self.param)
        ml.load_state_dict(self.nn)
        ml.eval()
        
        return ml
    
    
    def _select_scorefunc(self, ml):
        return ml.predict_proba
    
        
    def _init_shap(self, ml_score, trdata):
        
        # Kernel explaner will be used for non-tree-based model 
        explainer = shap.KernelExplainer(model         = ml_score,
                                         data          = trdata,
                                         model_output  = "identity"
                                        )
                        
        return explainer
        
        
    
    
                  
if __name__ == '__main__':
    main()
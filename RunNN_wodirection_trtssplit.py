# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 14:09:59 2020

@author: Tamura
"""
import pandas as pd
import numpy as np
import os
import optuna
import torch
import json
from joblib.externals.loky.backend.context import get_context
from functools                         import partial
from torch                             import nn
from torch.utils.data                  import DataLoader, Subset
from collections                       import OrderedDict, defaultdict
from collections                       import defaultdict, OrderedDict
from sklearn.model_selection           import StratifiedKFold
from sklearn.metrics                   import roc_auc_score
from BaseFunctions_NN                  import Base_wodirection

def torch2numpy(x):
    return x.to("cpu").detach().numpy().copy()

def ToJson(obj, path):
    
    f = open(path, 'w')
    json.dump(obj,f)

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
class FullyConnectedNN(nn.Module):
    
    def __init__(self, nbits, hidden_nodes, drop_rate):
        super(FullyConnectedNN, self).__init__()
        
        self.nbits = nbits
        self.nlayer = len(hidden_nodes)
        self.linear_relu_stack = nn.Sequential(OrderedDict([
                                                ('in-l1', nn.Linear(self.nbits, hidden_nodes[0])),
                                                ('relu' , nn.ReLU()),                                                
                                                ]))
        
        for i in range(1,self.nlayer):
            self.linear_relu_stack.add_module('l%d-l%d'%(i, i+1) , nn.Linear(hidden_nodes[i-1], hidden_nodes[i]))
            self.linear_relu_stack.add_module('relu%d'%(i), nn.ReLU())
            self.linear_relu_stack.add_module('dropout%d'%(i), nn.Dropout(drop_rate))
                
        self.linear_relu_stack.add_module('l%d-out'%self.nlayer, nn.Linear(hidden_nodes[i], 1))
        # self.linear_relu_stack.add_module('l%d-out'%self.nlayer, nn.Linear(hidden_nodes[i], 2))
        #self.linear_relu_stack.add_module('softmax', nn.Softmax(dim=1))
        
    def forward(self, x):
        signal = self.linear_relu_stack(x)
        return signal    
    
    
def train(model, device, loss_fn, optimizer, dataloader, verbose=10):
    model.to(device)
    loss_fn.to(device)
    
    model.train()
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        # calculate loss
        pred = model(X)
        loss = loss_fn(pred, y)
        
        # back propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #verbose
        if isinstance(verbose, int) & (batch % verbose == 0):
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            

def test(model, device, loss_fn, dataloader):
    model.to(device)
    loss_fn.to(device)

    size = len(dataloader.dataset)
    model.eval()
    threshold = torch.tensor([0.5], device=device)
    test_loss, correct = 0, 0
    
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            score      = model(X)
            pred       = (score>threshold).float()*1
            test_loss += loss_fn(score, y).item()
            correct   += (pred.flatten() == y.flatten()).sum().item()
            
            if batch == 0:
                predY_score = score.cpu()
                predY       = pred.cpu()
            else:
                predY_score = np.vstack([predY_score, score.cpu()])
                predY       = np.vstack([predY, pred.cpu()])
            
    test_loss /= size
    correct /= size
    #print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
    return predY_score, predY

        
def objective(trial, trX, trY, nfold, nepoch):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # parameter space
    EPOCH        = nepoch
    nbits        = trX.shape[1]
    n_layer      = trial.suggest_int('n_layer', 2, 3)
    hidden_nodes = [int(trial.suggest_discrete_uniform("num_filter_"+str(i), 250, 2000, 250)) for i in range(n_layer)]
    drop_rate    = trial.suggest_discrete_uniform('drop_rate', 0.25, 0.50, 0.25)
    adam_lr      = trial.suggest_loguniform('adam_lr', 1e-4, 1e-2)
    batch_size   = trial.suggest_categorical('batch_size', [64, 128, 256])
    
    # Create NN instance
    model     = FullyConnectedNN(nbits, hidden_nodes, drop_rate).to(device)
    w_pos     = int(np.where(trY==0)[0].shape[0] / np.where(trY==1)[0].shape[0])
    loss_fn   = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([w_pos]))#nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=adam_lr)
    
    # Set up dataloader
    dataset       = Dataset(fpset=trX, label=trY)
    kfold         = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=0)
    score_cv      = []
    for _fold, (idx_tr, idx_vl) in enumerate(kfold.split(trX, trY)):
        dataset_tr    = Subset(dataset, idx_tr)
        dataloader_tr = DataLoader(dataset_tr, batch_size, shuffle=True)
        dataset_vl    = Subset(dataset, idx_vl)
        dataloader_vl = DataLoader(dataset_vl, batch_size, shuffle=False)
    
    # X_tr, X_vl, y_tr, y_vl = train_test_split(trX, trY, test_size=0.2, random_state=42, shuffle=True, stratify=trY)
    # dataset_tr    = Dataset(fpset=X_tr, label=y_tr)
    # dataset_vl    = Dataset(fpset=X_vl, label=y_vl)
    # dataloader_tr = DataLoader(dataset_tr, shuffle=True, batch_size=batch_size, num_workers=2)
    # dataloader_vl = DataLoader(dataset_vl, shuffle=False, batch_size=batch_size, num_workers=2)
    
    # training
        for step in range(EPOCH):
            train(model, device, loss_fn, optimizer, dataloader_tr)
            predY_score, predY = test(model, device, loss_fn, dataloader_vl)
            score = roc_auc_score(y_true=trY[idx_vl], y_score=predY_score)
            #print(f"AUCROC: {(score):>0.5f}\n")
            
        score_cv.append(score)
            
    return np.mean(score_cv)

    
class Classification(Base_wodirection):

    def __init__(self, modeltype, model, dir_log, dir_score):

        super().__init__(modeltype, dir_log=dir_log, dir_score=dir_score, data_split_metric='trtssplit')
        self.model_name = model
        self.pred_type  = "classification"
        self.nfold      = 3
        
    
    def _fit_bestparams(self, params, trX, trY):
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        EPOCH         = self.nepoch[1]
        nbits         = trX.shape[1]
        model         = FullyConnectedNN(nbits=nbits,
                                         hidden_nodes=[int(params['num_filter_%d'%i]) for i in range(params['n_layer'])],
                                         drop_rate=params['drop_rate']
                                         )
        dataloader_tr = DataLoader(Dataset(fpset=trX, label=trY), shuffle=True, batch_size=params['batch_size'], num_workers=2, multiprocessing_context=get_context('loky'))
        w_pos     = int(np.where(trY==0)[0].shape[0] / np.where(trY==1)[0].shape[0])
        self.loss_fn   = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([w_pos]))
        optimizer     = torch.optim.Adam(model.parameters(), lr=params['adam_lr'])
        
        for step in range(EPOCH):
            train(model, device, self.loss_fn, optimizer, dataloader_tr)
            
        return model

    def _predict_bestparams(self, model, tsX, tsY):
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        dataloader_ts = DataLoader(Dataset(fpset=tsX, label=tsY), shuffle=False, num_workers=2, multiprocessing_context=get_context('loky'))
        
        pred_score, pred = test(model, device, self.loss_fn, dataloader_ts)
        
        return pred_score, pred
    
    def _AllMMSPred(self, target):
        
        if self._IsPredictableTarget():
            # Initialize
            log = defaultdict(list) 

            for trial in self.testsetidx:    
                
                if self.debug:
                    if trial>2:
                        break
                
                # Train test split
                print("    $ Prediction for cid%d is going on.\n" %trial)
                tr, ts, df_trX, df_trY, df_tsX, df_tsY, trX, trY, tsX, tsY = self._GetMatrices(trial)
                trY[np.where(trY==-1)[0]] = 0
                tsY[np.where(tsY==-1)[0]] = 0
                
                if self._IsPredictableSeries(tr, ts, min_npos=self.nfold):
                    # Fit and Predict
                    pruner = optuna.pruners.MedianPruner()
                    study = optuna.create_study(pruner=pruner, direction='maximize')
                    objective_partial = partial(objective, trX=trX, trY=trY, nfold=self.nfold, nepoch=self.nepoch[0])
                    study.optimize(objective_partial, n_trials=self.nepoch[2]) #NOTE!!!!
                    
                    ml = self._fit_bestparams(study.best_params, trX, trY)
                    score, predY = self._predict_bestparams(ml, tsX, tsY)
                    print("    $ Prediction Done.\n")

                    # Write & save log
                    log = self._WriteLog(log, trial, tr, ts, tsY, predY, score)           
                    self._Save(target, trial, log, ml, study)
                    print("    $  Log is out.\n")      
            
            
            
    def _WriteLog(self, log, trial, tr, ts, tsY, predY, score):
        
        predY = predY
        score = score

        # Write log
        tsids          = ts["id"].tolist()
        log["ids"]    += tsids
        log["trial"]  += [trial] * len(tsids)
        log['#tr']    += [tr.shape[0]] * len(tsids)
        log['#ac_tr'] += [tr[tr['class']==1].shape[0]] * len(tsids)
        log["trueY"]  += tsY.tolist()
        log["predY"]  += predY.flatten().astype(int).tolist()
        log["prob"]   += score.flatten().tolist()
        
        return log

    def _Save(self, target, trial, log, ml, study):
        
        path_log = os.path.join(self.logdir, "%s_%s_trial%d.tsv" %(target, self.mtype, trial))
        self.log = pd.DataFrame.from_dict(log)
        self.log.to_csv(path_log, sep="\t")

        ToJson(study.best_params, self.modeldir+"/params_%s_trial%d.json"%(target, trial))
        torch.save(ml.to('cpu').state_dict(), self.modeldir+"/Model_%s_trial%d.pth"%(target, trial))
        print("    $  Log is out.\n")  
        
        
    def ConvertRowScoreToProb(self, target):
        
        func = nn.Sigmoid()
        
        for trial in range(3):
            path_log     = os.path.join(self.logdir, "%s_%s_trial%d.tsv" %(target, self.mtype, trial))
            log          = pd.read_csv(path_log, sep='\t', index_col=0)
            log['score'] = log['prob'].copy()
            score        = torch.FloatTensor(log['score'].to_numpy())
            prob         = torch2numpy(func(score))
            log['prob']  = prob

            log.to_csv(path_log, sep='\t')



if __name__ == "__main__":
    import platform
    from tqdm import tqdm

    if platform.system() == 'Darwin':
        bd    = "/Users/tamura/work/ACPredCompare/"
    else:
        bd    = "/home/tamuras0/work/ACPredCompare/"
        
    #bd    = "/work/ta-shunsuke/ACPredCompare"
    model = "FCNN"
    mtype = "wodirection_trtssplit"
    os.chdir(bd)
    os.makedirs("./Log_%s"%mtype, exist_ok=True)
    os.makedirs("./Score_%s"%mtype, exist_ok=True)
    
    tlist = pd.read_csv('./Dataset/target_list.tsv', sep='\t', index_col=0)
    tlist = tlist.loc[tlist['predictable_trtssplit'], :]
    
    p = Classification(modeltype  = mtype,
                       model      = model,
                       dir_log    = './Log_%s/%s' %(mtype, model),
                       dir_score  = './Score_%s/%s' %(mtype, model)
                       )
    p.run_parallel(tlist['chembl_tid'])
    
    # for i, sr in tqdm(tlist.iterrows()):
        
    #     target = sr['chembl_tid']
        
    #     p.run(target=target, debug=True)
        # p.GetScore(t="Thrombin")
        
        # p.ConvertRowScoreToProb(target=target)
        
        

    
 

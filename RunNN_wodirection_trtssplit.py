# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 14:09:59 2020

@author: Tamura
"""
from operator import neg
import pandas as pd
import numpy as np
import sys
import os
import platform
import matplotlib.pyplot as plt
import optuna
import torch
from torch                             import nn
from torch.utils.data                  import DataLoader
from torch.nn.modules.container        import Sequential
from tqdm                              import tqdm
from MMP.make_input                    import LeaveOneCoreOut, GetDiverseCore, DelTestCpdFromTrain
from Tools.ReadWrite                   import ReadDataAndFingerprints, ToPickle, LoadPickle, ToJson
from Tools.utils                       import BasicInfo
from Evaluation.Labelling              import AddLabel
from Evaluation.Score                  import ScoreTable_wDefinedLabel
from collections                       import OrderedDict, defaultdict
from Plots.Barplot                     import MakeBarPlotsSeaborn as bar
from Plots.Lineplot                    import MakeLinePlotsSeaborn as line
from Fingerprint.Hash2BitManager       import Hash2Bits, FindBitLength
from functools                         import partial
from collections                       import defaultdict, OrderedDict
from sklearn.model_selection           import train_test_split
from sklearn.metrics                   import roc_auc_score
from BaseFunctions                    import Base_wodirection


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
                                                ('in-l1'    , nn.Linear(self.nbits,hidden_nodes[0])),
                                                ('relu'   , nn.ReLU()),                                                
                                                ]))
        
        for i in range(1,self.nlayer):
            self.linear_relu_stack.add_module('l%d-l%d'%(i, i+1) , nn.Linear(hidden_nodes[i-1], hidden_nodes[i]))
            self.linear_relu_stack.add_module('relu%d'%(i), nn.ReLU())
            self.linear_relu_stack.add_module('dropout%d'%(i), nn.Dropout(drop_rate))
                
        self.linear_relu_stack.add_module('l%d-out'%self.nlayer, nn.Linear(hidden_nodes[i], 1))
        self.linear_relu_stack.add_module('sigmoid', nn.Sigmoid())
        
    def forward(self, x):
        signal = self.linear_relu_stack(x)
        return signal    
    
    
def train(model, device, loss_fn, optimizer, dataloader, verbose=10):
    model.train()
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        # 損失誤差を計算
        pred = model(X)
        loss = loss_fn(pred, y)
        
        # バックプロパゲーション
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #verbose
        if isinstance(verbose, int) & (batch % verbose == 0):
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            

def test(model, device, loss_fn, dataloader):
    
    size = len(dataloader.dataset)
    model.eval()
    threshold = torch.tensor([0.5])
    test_loss, correct = 0, 0
    
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            score      = model(X)
            pred       = (score>threshold).float()*1
            test_loss += loss_fn(score, y).item()
            correct   += (pred.flatten() == y.flatten()).sum().item()
            
            if batch == 0:
                predY_score = score
                predY       = pred
            else:
                predY_score = np.vstack([predY_score, score])
                predY       = np.vstack([predY, pred])
            
    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
    return predY_score, predY

        
def objective(trial, trX, trY):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # parameter space
    EPOCH        = 50
    nbits        = trX.shape[1]
    n_layer      = trial.suggest_int('n_layer', 2, 3)
    hidden_nodes = [int(trial.suggest_discrete_uniform("num_filter_"+str(i), 250, 2000, 250)) for i in range(n_layer)]
    drop_rate    = trial.suggest_discrete_uniform('drop_rate', 0.25, 0.50, 0.25)
    adam_lr      = trial.suggest_loguniform('adam_lr', 1e-4, 1e-2)
    batch_size   = trial.suggest_categorical('batch_size', [64, 128, 256])
    
    # Create NN instance
    model     = FullyConnectedNN(nbits, hidden_nodes, drop_rate).to(device)
    loss_fn   = nn.BCELoss()#nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=adam_lr)
    
    # Set up dataloader
    X_tr, X_vl, y_tr, y_vl = train_test_split(trX, trY, test_size=0.2, random_state=42, shuffle=True, stratify=trY)
    dataset_tr = Dataset(fpset=X_tr, label=y_tr)
    dataset_vl = Dataset(fpset=X_vl, label=y_vl)
    dataloader_tr = DataLoader(dataset_tr, shuffle=True, batch_size=batch_size, num_workers=2)
    dataloader_vl = DataLoader(dataset_vl, shuffle=False, batch_size=batch_size, num_workers=2)
    
    # training
    for step in range(EPOCH):
        train(model, device, loss_fn, optimizer, dataloader_tr)
        predY_score, predY = test(model, device, loss_fn, dataloader_vl)
        score = roc_auc_score(y_true=y_vl, y_score=predY_score)
        print(f"AUCROC: {(score):>0.5f}\n")
            
        trial.report(score, step)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
    return score

    
class Classification(Base_wodirection):

    def __init__(self, modeltype, model, dir_log, dir_score):

        super().__init__(modeltype, model_name=model, dir_log=dir_log, dir_score=dir_score, data_split_metric='trtssplit')

        self.pred_type = "classification"
    
    def _fit_bestparams(params, trX, trY):
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        EPOCH         = 200
        nbits         = trX.shape[1]
        model         = FullyConnectedNN(nbits=nbits, **params)
        dataloader_tr = DataLoader(Dataset(fpset=trX, label=trY), shuffle=True ,batch_size=params['batch_size'], num_workers=2)
        loss_fn       = nn.CrossEntropyLoss()
        optimizer     = torch.optim.Adam(lr=params['adam_lr'])
        
        for step in range(EPOCH):
            train(model, device, loss_fn, optimizer, dataloader_tr)
            
        return model

    def _predict_bestparams(model, tsX, tsY):
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        dataloader_ts = DataLoader(Dataset(fpset=tsX, label=tsY), shuffle=False, num_workers=2)
        loss_fn       = nn.CrossEntropyLoss()
        
        pred_score, pred = test(model, device, loss_fn, dataloader_ts)
        
        return pred_score.to('cpu').detach().numpy(), pred.to('cpu').detach().numpy()
    
    def _AllMMSPred(self, target, path_log):
        
        # Initialize
        log = defaultdict(list) 

        for cid in self.testsetidx:    
            
            if self.debug:
                if cid>2:
                    break
            
            # Train test split
            print("    $ Prediction for cid%d is going on.\n" %cid)
            tr, ts, df_trX, df_trY, df_tsX, df_tsY, trX, trY, tsX, tsY = self._GetMatrices(cid)
            trY[np.where(trY==-1)[0]] = 0
            tsY[np.where(tsY==-1)[0]] = 0
            
            # Fit and Predict
            pruner = optuna.pruners.MedianPruner()
            study = optuna.create_study(pruner=pruner, direction='maximize')
            objective_partial = partial(objective, trX=trX, trY=trY)
            study.optimize(objective_partial, n_trials=100)
            
            ml = self._fit_bestparams(study.best_params, trX, trY)
            score, predY = self._predict_bestparams(ml, tsX, tsY)
            print("    $ Prediction Done.\n")

            # Write log
            tsids         = ts["id"].tolist()
            log["ids"]   += tsids
            log["cid"]   += [cid] * len(tsids)
            log["trueY"] += tsY.tolist()
            log["predY"] += predY.tolist()
            log["prob"]  += score
            
            # Save            
            path_log = os.path.join(self.logdir, "%s_%s_trial%d.tsv" %(target, self.mtype, cid))
            self.log = pd.DataFrame.from_dict(log)
            self.log.to_csv(path_log, sep="\t")

            ToJson(ml.get_params(), self.modeldir+"/params_trial%d.json"%cid)
            torch.save(ml.to('cpu').state_dict(), self.modeldir+"/model_trial%d.pth"%cid)
            print("    $  Log is out.\n")


if __name__ == "__main__":
    
    if platform.system() == 'Darwin':
        bd    = "/Users/tamura/work/ACPredCompare/"
    else:
        bd    = "/home/tamuras0/work/ACPredCompare/"
    
    model = "FCNN"
    mtype = "wodirection"
    os.chdir(bd)
    os.makedirs("./Log_trtssplit", exist_ok=True)
    os.makedirs("./Score_trtssplit", exist_ok=True)
    
    tlist = pd.read_csv('./Dataset/target_list.tsv', sep='\t', index_col=0)
    
    for i, sr in tlist.iterrows():
        
        target = sr['target']
        
        p = Classification(
                           modeltype   = mtype,
                           model       = model,
                           dir_log     = "./Log_trtssplit/%s" %(model+'_'+mtype),
                           dir_score   = "./Score_trtssplit/%s" %(model+'_'+mtype),
                          )

        p.run(target=target, debug=True)
        # p.GetScore(t="Thrombin")
        
        #TODO
        #function for fw should be independent.
        #This file should concentrate on fit/predict
        #Make another scriptto calculate fw.
        #Also funcs for calc score should be independent.
        

    
 
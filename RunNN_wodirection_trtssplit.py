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
from sklearn.metrics                   import roc_auc_score, accuracy_score
from BaseFunctions                     import Base_wodirection

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
       
       
def train(model, device, loss_fn, optimizer, dataloader, verbose=5):
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
            loss, current = loss.item(), batch * 128
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            

def test(model, device, loss_fn, dataloader):
    model.to(device)
    loss_fn.to(device)

    size = len(dataloader.dataset)
    model.eval()
    threshold = torch.tensor([0.5], device=device)
    test_loss, correct = 0, 0
    pred_score_all, pred_all, proba_all, y_all = [], [], [], []
    
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            score      = model(X)
            proba      = predict_proba(score)
            pred       = (proba>threshold).float()*1
            test_loss += loss_fn(score, y).item()
            correct   += (pred.flatten() == y.flatten()).sum().item()
            
            pred_score_all += torch2numpy(score).reshape(-1).tolist()
            pred_all       += torch2numpy(pred).reshape(-1).tolist()
            proba_all      += torch2numpy(proba).reshape(-1).tolist()
            y_all          += torch2numpy(y).reshape(-1).tolist() 
            
            test_loss += loss_fn(score, y.to(device))
            
           
    test_loss /= size
    acc    = accuracy_score(pred_all, y_all)
    aucroc = roc_auc_score(y_all, proba_all)
    print(f"Test Error: \n Accuracy: {(100*acc):>0.1f}%, AUCROC: {(aucroc):>0.3f}, Avg loss: {test_loss:>8f} \n")
    
    return pred_score_all, pred_all, proba_all


def predict_proba(X):
    output_act = nn.Sigmoid()
    out = output_act(X)
    
    return out

        
def objective(trial, trX, trY, nfold, nepoch):
    
    # parameter space
    EPOCH        = nepoch
    args = dict(
                train_num    = nepoch,
                batch_size   = 128,
                cuda         = False,#torch.cuda.is_available(),
                device       = 'cpu',#'cuda' if torch.cuda.is_available() else 'cpu'
                gamma        = 0.1,
                nbits        = trX.shape[1]
                )
    
    optarg = dict(
                  drop_rate    = trial.suggest_discrete_uniform('drop_rate', 0.1, 0.3, 0.05),
                  step_num     = trial.suggest_int("step_num", 1, 3),
                  DNNLayerNum  = trial.suggest_int('DNNLayerNum', 2, 8),  
                  adam_lr      = trial.suggest_loguniform('adam_lr', 1e-4, 1e-2),              
                )
    
    args.update(optarg)
    args['step_size']  = int(args['train_num']/args['step_num'])
    args['grad_node']  = int(args['nbits']/ args['DNNLayerNum'])
    args['node_list']  = [int(args['nbits'] - args['grad_node']*num) for num in range(args['DNNLayerNum'])] + [1]
    
    device = args['device']
    
    # Set up dataloader
    dataset       = Dataset(fpset=trX, label=trY)
    kfold         = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=0)
    score_cv      = []
    for _fold, (idx_tr, idx_vl) in enumerate(kfold.split(trX, trY)):
        dataset_tr    = Subset(dataset, idx_tr)
        dataloader_tr = DataLoader(dataset_tr, args['batch_size'], shuffle=True)
        dataset_vl    = Subset(dataset, idx_vl)
        dataloader_vl = DataLoader(dataset_vl, args['batch_size'], shuffle=False)

        # Create NN instance
        model     = FullyConnectedNN(args).to(device)
        w_pos     = int(np.where(trY[idx_tr]==0)[0].shape[0] / np.where(trY[idx_tr]==1)[0].shape[0])
        loss_fn   = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([w_pos]))#nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args['adam_lr'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args['step_size'], gamma=args['gamma'])
    
    # training
        for step in range(EPOCH):
            train(model, device, loss_fn, optimizer, dataloader_tr)
            predY_score, predY, proba = test(model, device, loss_fn, dataloader_vl)
            score = roc_auc_score(y_true=trY[idx_vl], y_score=proba)
            scheduler.step()
            
        score_cv.append(score)
            
    return np.mean(score_cv)

    
class Classification(Base_wodirection):

    def __init__(self, modeltype, model, dir_log, dir_score):

        super().__init__(modeltype, dir_log=dir_log, dir_score=dir_score, data_split_metric='trtssplit')
        self.model_name = model
        self.pred_type  = "classification"
        self.nfold      = 3
        
        
    def _Setnepoch(self):
        
        if self.debug:
            nepoch = [2, 2, 2]
        else:
            nepoch = [50, 100, 100]
            
        return nepoch
    
    
    def _fit_bestparams(self, params, trX, trY, loss_fn):
        
        device = params['device']
        
        EPOCH = self.nepoch[1]
        nbits = trX.shape[1]
        model = FullyConnectedNN(params)
        
        dataloader_tr = DataLoader(Dataset(fpset=trX, label=trY),
                                   shuffle=True,
                                   batch_size=params['batch_size'],
                                   )
        
        optimizer     = torch.optim.Adam(model.parameters(), lr=params['adam_lr'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=params['step_size'], gamma=params['gamma'])
        
        for step in range(EPOCH):
            train(model, device, loss_fn, optimizer, dataloader_tr)
            scheduler.step()
            
        return model


    def _predict_bestparams(self, params, model, tsX, tsY, loss_fn):
        
        device = params['device']
        
        dataloader_ts = DataLoader(Dataset(fpset=tsX, label=tsY),
                                   batch_size=params['batch_size'],
                                   shuffle=False,
                                   num_workers=2,
                                   multiprocessing_context=get_context('loky'),
                                   )
        
        pred_score, pred, proba = test(model, device, loss_fn, dataloader_ts)
        
        return pred_score, pred, proba
    
    
    def _AllMMSPred(self, target):
        
        if self._IsPredictableTarget():
            # Initialize
            log = defaultdict(list) 

            for trial in [2]:#self.testsetidx:    
                
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
                    
                    best_args = dict(
                                train_num    = self.nepoch[0],
                                batch_size   = 128,
                                cuda         = False,#torch.cuda.is_available(),
                                device       = 'cpu',#'cuda' if torch.cuda.is_available() else 'cpu'
                                gamma        = 0.1,
                                nbits        = trX.shape[1]
                                )
                    best_args.update(study.best_params)
                    best_args['step_size']  = int(best_args['train_num']/best_args['step_num'])
                    best_args['grad_node']  = int(best_args['nbits']/ best_args['DNNLayerNum'])
                    best_args['node_list']  = [int(best_args['nbits'] - best_args['grad_node']*num) for num in range(best_args['DNNLayerNum'])] + [1]

                    w_pos     = int(np.where(trY==0)[0].shape[0] / np.where(trY==1)[0].shape[0])
                    loss_fn   = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([w_pos]))
                    ml        = self._fit_bestparams(best_args, trX, trY, loss_fn)
                    score_tr, predY_tr, proba_tr = self._predict_bestparams(best_args, ml, trX, trY, loss_fn)
                    score_ts, predY_ts, proba_ts = self._predict_bestparams(best_args, ml, tsX, tsY, loss_fn)
                    print("    $ Prediction Done.\n")

                    # Write & save log
                    log_tr = self.WriteLog_tr(trial, tr, trY, predY_tr, proba_tr) 
                    log_ts = self.WriteLog_ts(trial, tr, ts, tsY, predY_ts, proba_ts)  
                    self.Save(target, trial, log_tr, log_ts, best_args, ml)
                    print("    $  Log is out.\n")      
            
            
            
    def WriteLog_tr(self, loop, tr, trY, predY, proba):
        
        log = defaultdict(list)
        
        # Write log
        trids         = tr["id"].tolist()
        log["ids"]   += trids
        log["loop"]  += [loop] * len(trids)
        log['#tr']   += [tr.shape[0]] * len(trids)
        log['#ac_tr']+= [tr[tr['class']==1].shape[0]] * len(trids)
        log["trueY"] += trY.tolist()
        log["predY"] += predY
        log["prob"]  += proba
        
        return log
    
    
    def WriteLog_ts(self, loop, tr, ts, tsY, predY, proba):
        
        log = defaultdict(list)
        
        # Write log
        tsids         = ts["id"].tolist()
        log["ids"]   += tsids
        log["loop"]  += [loop] * len(tsids)
        log['#tr']   += [tr.shape[0]] * len(tsids)
        log['#ac_tr']+= [tr[tr['class']==1].shape[0]] * len(tsids)
        log["trueY"] += tsY.tolist()
        log["predY"] += predY
        log["prob"]  += proba
        
        return log
        
        
    def Save(self, target, cid, log_tr, log_ts, args, dnn):
        path_log_tr = os.path.join(self.logdir, "%s_trial%d_train.tsv" %(target, cid))
        self.log_tr = pd.DataFrame.from_dict(log_tr)
        self.log_tr.to_csv(path_log_tr, sep="\t")
        
        path_log_ts = os.path.join(self.logdir, "%s_trial%d_test.tsv" %(target, cid))
        self.log_ts = pd.DataFrame.from_dict(log_ts)
        self.log_ts.to_csv(path_log_ts, sep="\t")

        ToJson(args, self.modeldir+"/params_%s_trial%d.json" %(target, cid))
        torch.save(dnn.to('cpu').state_dict(), self.modeldir+"/dnn_%s_trial%d.pth" %(target, cid))



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
    
    p.run(target='CHEMBL244', debug=False)
    # p.run_parallel(tlist['chembl_tid'])
    
    # for i, sr in tqdm(tlist.iterrows()):
        
    #     target = sr['chembl_tid']
        
    #     p.run(target=target, debug=False)
        # p.GetScore(t="Thrombin")
        
        # p.ConvertRowScoreToProb(target=target)
        
        

    
 

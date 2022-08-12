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
        self.X    = fpset
        self.X_c  = self.X['core']
        self.X_s1 = self.X['sub1']
        self.X_s2 = self.X['sub2']
        self.y = label.reshape([label.shape[0],1])
        
    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        tensor_X_c  = torch.FloatTensor(self.X_c[idx,:])
        tensor_X_s1 = torch.FloatTensor(self.X_s1[idx,:])
        tensor_X_s2 = torch.FloatTensor(self.X_s2[idx,:])
        tensor_y = torch.FloatTensor(self.y[idx])
        return tensor_X_c, tensor_X_s1, tensor_X_s2, tensor_y    
    
 
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
    
    
def train(dnn_c, dnn_s1, dnn_s2, dnn_cat, dataloader, optimizer, loss_fn, cuda, verbose=1):

    device  = 'cuda' if cuda else 'cpu'  
    
    if cuda:
        dnn_c   = dnn_c.cuda()
        dnn_s1  = dnn_s1.cuda()
        dnn_s2  = dnn_s2.cuda()
        dnn_cat = dnn_cat.cuda()
        loss_fn = loss_fn.cuda()
          
        
    size = len(dataloader.dataset)
    
    n_usedtr = 0
    for batch, (X_c, X_s1, X_s2, y) in enumerate(dataloader):
        
        X_c, X_s1, X_s2, y = X_c.to(device), X_s1.to(device), X_s2.to(device), y.to(device)
        
        score = predict(dnn_c, dnn_s1, dnn_s2, dnn_cat, X_c, X_s1, X_s2)
        loss  = loss_fn(score, y)
        
        # back propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #verbose
        n_usedtr += y.shape[0]
        if isinstance(verbose, int) & (batch % verbose == 0):
            loss, current = loss.item(), batch * 128
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
    return dnn_c, dnn_s1, dnn_s2, dnn_cat


def test(dnn_c, dnn_s1, dnn_s2, dnn_cat, dataloader, loss_fn, cuda):
    
    device = 'cuda' if cuda else 'cpu'
    
    dnn_c.eval()
    dnn_s1.eval()
    dnn_s2.eval()
    dnn_cat.eval()
    
    size      = len(dataloader.dataset)
    threshold = torch.tensor([0.5]).to(device)
    test_loss = 0
    pred_score_all, pred_all, proba_all, y_all = [], [], [], []
    
    with torch.no_grad():
        for X_c, X_s1, X_s2, y in dataloader:
            
            X_c, X_s1, X_s2, y = X_c.to(device), X_s1.to(device), X_s2.to(device), y.to(device)
            
            pred_score  = predict(dnn_c, dnn_s1, dnn_s2, dnn_cat, X_c, X_s1, X_s2)
            proba       = predict_proba(pred_score)
            pred        = (proba>threshold).float()*1
            
            pred_score_all += torch2numpy(pred_score).reshape(-1).tolist()
            pred_all       += torch2numpy(pred).reshape(-1).tolist()
            proba_all      += torch2numpy(proba).reshape(-1).tolist()
            y_all          += torch2numpy(y).reshape(-1).tolist() 
            
            test_loss += loss_fn(pred_score, y.to(device))
            
    test_loss /= size
    acc    = accuracy_score(pred_all, y_all)
    aucroc = roc_auc_score(y_all, proba_all)
    print(f"Test Error: \n Accuracy: {(100*acc):>0.1f}%, AUCROC: {(aucroc):>0.3f}, Avg loss: {test_loss:>8f} \n")
    
    return pred_score_all, pred_all, proba_all


def predict(dnn_c, dnn_s1, dnn_s2, dnn_cat, X_c, X_s1, X_s2):
        
    out_c  = dnn_c(X_c)
    out_s1 = dnn_s1(X_s1)
    out_s2 = dnn_s2(X_s2)
    out    = torch.cat((out_c, out_s1, out_s2), dim=1)
    pred   = dnn_cat(out)
    
    return pred


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
                cuda         = torch.cuda.is_available(),
                device       = 'cuda' if torch.cuda.is_available() else 'cpu',
                gamma        = 0.1,
                nbits_c      = trX['core'].shape[1],
                nbits_s      = trX['sub1'].shape[1],
                )
    
    optarg = dict(
                  drop_rate       = trial.suggest_discrete_uniform('drop_rate', 0.1, 0.3, 0.05),
                  step_num        = trial.suggest_int("step_num", 1, 3),
                  DNNLayerNum_c   = trial.suggest_int("DNNLayerNum_c", 1, 4),
                  DNNLayerNum_s   = trial.suggest_int("DNNLayerNum_s", 1, 4),
                  DNNLayerNum_cat = trial.suggest_int("DNNLayerNum_cat", 2, 8),
                  dim_c           = trial.suggest_int("dim_c", 70, 100, 10),
                  dim_s           = trial.suggest_int("dim_s", 70, 100, 10),
                  adam_lr         = trial.suggest_loguniform('adam_lr', 1e-4, 1e-2),              
                )
    
    args.update(optarg)
    args['step_size']  = int(args['train_num']/args['step_num'])
    
    args_c = args.copy()
    args_c['grad_node']  = int((args['nbits_c'] - args['dim_c'])/ args['DNNLayerNum_c'])
    args_c['node_list']  = [int(args['nbits_c'] - args_c['grad_node']*num) for num in range(args['DNNLayerNum_c'])] + [args['dim_c']]
    
    args_s = args.copy()
    args_s['grad_node']  = int((args['nbits_s'] - args['dim_s'])/ args['DNNLayerNum_s'])
    args_s['node_list']  = [int(args['nbits_s'] - args_s['grad_node']*num) for num in range(args['DNNLayerNum_s'])] + [args['dim_s']]
    
    args_cat = args.copy()
    args_cat['grad_node']  = int((args['dim_c'] + 2*args['dim_s']) / args['DNNLayerNum_cat'])
    args_cat['node_list']  = [int((args['dim_c'] + 2*args['dim_s']) - args_cat['grad_node']*num) for num in range(args['DNNLayerNum_cat'])] + [1]
    
    device = args['device']
    
    # Set up dataloader
    dataset       = Dataset(fpset=trX, label=trY)
    kfold         = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=0)
    score_cv      = []
    for _fold, (idx_tr, idx_vl) in enumerate(kfold.split(trX['core'], trY)):
        dataset_tr    = Subset(dataset, idx_tr)
        dataloader_tr = DataLoader(dataset_tr, args['batch_size'], shuffle=True)
        dataset_vl    = Subset(dataset, idx_vl)
        dataloader_vl = DataLoader(dataset_vl, args['batch_size'], shuffle=False)

        # Create NN instance
        dnn_c     = FullyConnectedNN(args_c).to(device)
        dnn_s1    = FullyConnectedNN(args_s).to(device)
        dnn_s2    = FullyConnectedNN(args_s).to(device)
        dnn_cat   = FullyConnectedNN(args_cat).to(device)
        w_pos     = int(np.where(trY[idx_tr]==0)[0].shape[0] / np.where(trY[idx_tr]==1)[0].shape[0])
        loss_fn   = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([w_pos]))#nn.CrossEntropyLoss()
        opt_list  = list(dnn_c.parameters()) + list(dnn_s1.parameters()) + list(dnn_s2.parameters()) +list(dnn_cat.parameters())
        optimizer = torch.optim.Adam(opt_list, lr=args['adam_lr'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args['step_size'], gamma=args['gamma'])
    
    # training
        for step in range(EPOCH):
            train(dnn_c, dnn_s1, dnn_s2, dnn_cat, dataloader_tr, optimizer, loss_fn, args['cuda'])
            predY_score, predY, proba = test(dnn_c, dnn_s1, dnn_s2, dnn_cat, dataloader_vl, loss_fn, args['cuda'])
            score = roc_auc_score(y_true=trY[idx_vl], y_score=proba)
            scheduler.step()
            
        score_cv.append(score)
            
    return np.mean(score_cv)

    
class Classification(Base_wodirection):

    def __init__(self, modeltype, model, dir_log, dir_score):

        super().__init__(modeltype, dir_log=dir_log, dir_score=dir_score, data_split_metric='axv')
        self.model_name = model
        self.pred_type  = "classification"
        self.nfold      = 3
        
        
    def _Setnepoch(self):
        
        if self.debug:
            nepoch = [3, 3, 3]
        else:
            nepoch = [50, 100, 100]
            
        return nepoch
    
    def _fit_bestparams(self, args_c, args_s, args_cat, trX, trY, loss_fn):
    
        dataloader_tr = DataLoader(Dataset(fpset=trX, label=trY),
                                    shuffle=True,
                                    batch_size=args_cat['batch_size'],
                                    )
        
        dnn_c     = FullyConnectedNN(args_c).to(args_cat['device'])
        dnn_s1    = FullyConnectedNN(args_s).to(args_cat['device'])
        dnn_s2    = FullyConnectedNN(args_s).to(args_cat['device'])
        dnn_cat   = FullyConnectedNN(args_cat).to(args_cat['device'])
        
        opt_list  = list(dnn_c.parameters()) + list(dnn_s1.parameters()) + list(dnn_s2.parameters()) + list(dnn_cat.parameters())
        optimizer = torch.optim.Adam(opt_list, lr=args_cat['adam_lr'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args_cat['step_size'], gamma=args_cat['gamma'])
        
        EPOCH = args_cat['train_num']
        for step in range(EPOCH):
            dnn_c, dnn_s1, dnn_s2, dnn_cat = train(dnn_c, dnn_s1, dnn_s2, dnn_cat, dataloader_tr, optimizer, loss_fn, args_cat['cuda'])
            scheduler.step()
            
        return dnn_c, dnn_s1, dnn_s2, dnn_cat
            

    def _predict_bestparams(self, dnn_c, dnn_s1, dnn_s2, dnn_cat, args, tsX, tsY, loss_fn):
        
        dataloader_ts = DataLoader(Dataset(fpset=tsX, label=tsY),
                                   batch_size=args['batch_size'],
                                   shuffle=False,
                                   num_workers=2,
                                   multiprocessing_context=get_context('loky'),
                                   )
        
        pred_score, pred, proba = test(dnn_c, dnn_s1, dnn_s2, dnn_cat, dataloader_ts, loss_fn, args['cuda'])        
        return pred_score, pred, proba
    
    
    def _AllMMSPred(self, target):
        
        if self._IsPredictableTarget():
            # Initialize
            self.nepoch = self._Setnepoch()

            for trial in self.testsetidx:    
                
                if self.debug:
                    if trial>2:
                        break
                
                # Train test split
                print("    $ Prediction for cid%d is going on.\n" %trial)
                tr, cpdout, bothout, df_trX, df_trY, df_cpdoutX, df_cpdoutY, df_bothoutX, df_bothoutY, trX, trY, cpdoutX, cpdoutY, bothoutX, bothoutY = self._GetMatrices(trial, separated_input=True)
                trY[np.where(trY==-1)[0]] = 0
                cpdoutY[np.where(cpdoutY==-1)[0]] = 0
                bothoutY[np.where(bothoutY==-1)[0]] = 0
                
                flag_predictable = self._IsPredictableSeries(tr, cpdout, min_npos=self.nfold) * self._IsPredictableSeries(tr, bothout, min_npos=self.nfold)
                if flag_predictable:
                    # Fit and Predict
                    pruner = optuna.pruners.MedianPruner()
                    study = optuna.create_study(pruner=pruner, direction='maximize')
                    objective_partial = partial(objective, trX=trX, trY=trY, nfold=self.nfold, nepoch=self.nepoch[0])
                    study.optimize(objective_partial, n_trials=self.nepoch[2]) #NOTE!!!!
                    
                    args = dict(
                                train_num    =self.nepoch[0],
                                batch_size   = 128,
                                cuda         = torch.cuda.is_available(),
                                device       = 'cuda' if torch.cuda.is_available() else 'cpu',
                                gamma        = 0.1,
                                nbits_c      = trX['core'].shape[1],
                                nbits_s      = trX['sub1'].shape[1],
                                )
                    
                    args.update(study.best_params)
                    args['step_size']  = int(args['train_num']/args['step_num'])

                    args_c = args.copy()
                    args_c['grad_node']  = int((args['nbits_c'] - args['dim_c'])/ args['DNNLayerNum_c'])
                    args_c['node_list']  = [int(args['nbits_c'] - args_c['grad_node']*num) for num in range(args['DNNLayerNum_c'])] + [args['dim_c']]

                    args_s = args.copy()
                    args_s['grad_node']  = int((args['nbits_s'] - args['dim_s'])/ args['DNNLayerNum_s'])
                    args_s['node_list']  = [int(args['nbits_s'] - args_s['grad_node']*num) for num in range(args['DNNLayerNum_s'])] + [args['dim_s']]

                    args_cat = args.copy()
                    args_cat['grad_node']  = int((args['dim_c'] + 2*args['dim_s']) / args['DNNLayerNum_cat'])
                    args_cat['node_list']  = [int((args['dim_c'] + 2*args['dim_s']) - args_cat['grad_node']*num) for num in range(args['DNNLayerNum_cat'])] + [1]

                    w_pos     = int(np.where(trY==0)[0].shape[0] / np.where(trY==1)[0].shape[0])
                    loss_fn   = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([w_pos]))
                    dnn_c, dnn_s1, dnn_s2, dnn_cat = self._fit_bestparams(args_c, args_s, args_cat, trX, trY, loss_fn)
                    score_tr, predY_tr, proba_tr   = self._predict_bestparams(dnn_c, dnn_s1, dnn_s2, dnn_cat, args, trX, trY, loss_fn)
                    score_cpdout , predY_cpdout , proba_cpdout  = self._predict_bestparams(dnn_c, dnn_s1, dnn_s2, dnn_cat, args, cpdoutX, cpdoutY, loss_fn)
                    score_bothout, predY_bothout, proba_bothout = self._predict_bestparams(dnn_c, dnn_s1, dnn_s2, dnn_cat, args, bothoutX, bothoutY, loss_fn)
                    print("    $ Prediction Done.\n")

                    # Write & save log
                    log_tr = self.WriteLog_tr(trial, tr, trY, predY_tr, proba_tr) 
                    log_cpdout  = self.WriteLog_ts(trial, tr, cpdout, cpdoutY, predY_cpdout, proba_cpdout)
                    log_bothout = self.WriteLog_ts(trial, tr, bothout, bothoutY, predY_bothout, proba_bothout)  
                    self.Save(target, trial, log_tr, log_cpdout, log_bothout, args_c, args_s, args_cat, dnn_c, dnn_s1, dnn_s2, dnn_cat)
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
        
        
    def Save(self, target, seed, log_tr, log_cpdout, log_bothout, args_c, args_s, args_cat, dnn_c, dnn_s1, dnn_s2, dnn_cat):
        path_log_tr = os.path.join(self.logdir, "%s_Seed%d_train.tsv" %(target, seed))
        self.log_tr = pd.DataFrame.from_dict(log_tr)
        self.log_tr.to_csv(path_log_tr, sep="\t")
        
        path_log_cpdout = os.path.join(self.logdir, "%s_Seed%d_cpdout.tsv" %(target, seed))
        self.log_cpdout = pd.DataFrame.from_dict(log_cpdout)
        self.log_cpdout.to_csv(path_log_cpdout, sep="\t")
        
        path_log_bothout = os.path.join(self.logdir, "%s_Seed%d_bothout.tsv" %(target, seed))
        self.log_bothout = pd.DataFrame.from_dict(log_bothout)
        self.log_bothout.to_csv(path_log_bothout, sep="\t")

        
        args = dict(dnn_c = args_c, dnn_s = args_s, dnn_cat=args_cat)
        ToJson(args, self.modeldir+"/params_%s_Seed%d.json" %(target, seed))
        
        torch.save(dnn_c.to('cpu').state_dict() ,  self.modeldir+"/dnn_c_%s_Seed%d.pth" %(target, seed))
        torch.save(dnn_s1.to('cpu').state_dict(),  self.modeldir+"/dnn_s1_%s_Seed%d.pth" %(target, seed))
        torch.save(dnn_s2.to('cpu').state_dict(),  self.modeldir+"/dnn_s2_%s_Seed%d.pth" %(target, seed))
        torch.save(dnn_cat.to('cpu').state_dict(), self.modeldir+"/dnn_cat_%s_Seed%d.pth" %(target, seed))



def main(bd):
    
    #Initialize   
    model = "FCNN_separated"
    mtype = "axv"
    
    tlist = pd.read_csv('./Dataset/target_list.tsv', sep='\t', index_col=0)
    #tlist = tlist.loc[tlist['machine1'],:]
    
    os.chdir(bd)
    os.makedirs("./Log_%s"%mtype, exist_ok=True)
    os.makedirs("./Score_%s"%mtype, exist_ok=True)
    
    p = Classification(modeltype  = mtype,
                       model      = model,
                       dir_log    = './Log_%s/%s' %(mtype, model),
                       dir_score  = './Score_%s/%s' %(mtype, model),
                       )
                    
    p.run_parallel(tlist['chembl_tid'], njob=6)
    

def debug(bd):
    
    #Initialize   
    model = "FCNN_separated"
    mtype = "axv"
    mtype +='_debug'
    
    tlist = pd.read_csv('./Dataset/target_list.tsv', sep='\t', index_col=0)
    #tlist = tlist.loc[tlist['machine1'],:]
        
    os.chdir(bd)
    os.makedirs("./Log_%s"%mtype, exist_ok=True)
    os.makedirs("./Score_%s"%mtype, exist_ok=True)
    
    p = Classification(modeltype  = mtype,
                       model      = model,
                       dir_log    = './Log_%s/%s' %(mtype, model),
                       dir_score  = './Score_%s/%s' %(mtype, model),
                       )
                    
    p.run('CHEMBL204', debug=True)
                    
            
if __name__ == '__main__':    
    
    #bd = '/home/bit/tamuras0/ACPredCompare'
    bd = '/home/tamuras0/work/ACPredCompare'
    
    main(bd)
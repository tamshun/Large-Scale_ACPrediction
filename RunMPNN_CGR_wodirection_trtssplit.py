from importlib_metadata import functools
import pandas as pd
import numpy as np
import os
import joblib
from collections import namedtuple
from sklearn.model_selection import StratifiedShuffleSplit

import pandas as pd
import numpy as np
import sys
import os
import optuna
import torch
from torch                             import nn
from torch.utils.data                  import dataloader, Subset, DataLoader
#from torch_geometric.data              import DataLoader
from Tools.ReadWrite                   import ToJson, LoadJson
from collections                       import defaultdict
from sklearn.metrics                   import roc_auc_score
from sklearn.model_selection           import StratifiedKFold
from sklearn.metrics                   import roc_auc_score, accuracy_score
from BaseFunctions_NN                  import Base_wodirection_CGR, torch2numpy, MolGraph, BatchMolGraph, MPNEncoder
import random
from functools import partial
from rdkit import Chem
from argparse import Namespace

class Dataset(torch.utils.data.Dataset):
    
    def __init__(self, smi_list, label_list):
        
        if not isinstance(smi_list, np.ndarray):
            smi_list = smi_list.to_numpy()
        if not isinstance(label_list, np.ndarray):
            label_list = label_list.to_numpy()
        
        # Negative label is converted into 0 (originally -1)
        if -1 in label_list:    
            label_list[np.where(label_list==-1)[0]] = 0
            
        self.smi_list   = smi_list
        self.label_list = label_list
        
    def __len__(self):
        return self.label_list.shape[0]

    def __getitem__(self, idx):
        return self.smi_list[idx], self.label_list[idx]
  
    
def mycollate_fn(batch):
    
    batch_list = list(zip(batch))
    mol_graphs = []
    labels     = []
    
    for smi, label in batch:
        mol_graphs.append(MolGraph(smi))
        labels.append(label)
    
    labels = torch.FloatTensor(labels).reshape([-1, 1])
        
    return mol_graphs, labels


class DeepNeuralNetwork(nn.Module):
    
    def __init__(self, arg, random_seed=0):
        
        super(DeepNeuralNetwork, self).__init__()
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        hidden_list  = arg['node_list']
        W_list       = [nn.Linear(hidden_list[0], hidden_list[1], bias=True)]
        self.dropout = nn.Dropout(p=arg['dropout'])
        self.active_function = nn.ReLU()
        
        for num in range(len(hidden_list)-2):
            W_list.extend([self.active_function, self.dropout, nn.Linear(hidden_list[num+1], hidden_list[num+2], bias=True)])
                
        modulelist  =  nn.ModuleList(W_list) 
        self.W      =  nn.Sequential(*modulelist)

    def forward(self, x):
        return self.W(x)  
    
    
class Classification(Base_wodirection_CGR):

    def __init__(self, target, modeltype, model, dir_log, dir_score, data_split_metric='trtssplit', debug=False):

        super().__init__(target, modeltype, dir_log=dir_log, dir_score=dir_score, data_split_metric=data_split_metric)

        self.pred_type = "classification"
        self.mname     = model
        self.debug     = debug
        self.nfold     = 3
        torch.autograd.set_detect_anomaly(True)
    
    
    def _SetML(self):
        
        return super()._SetML()
    
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
        
        
    def Save(self, target, cid, log_tr, log_ts, args, mpn, dnn):
        path_log_tr = os.path.join(self.logdir, "%s_trial%d_train.tsv" %(target, cid))
        self.log_tr = pd.DataFrame.from_dict(log_tr)
        self.log_tr.to_csv(path_log_tr, sep="\t")
        
        path_log_ts = os.path.join(self.logdir, "%s_trial%d_test.tsv" %(target, cid))
        self.log_ts = pd.DataFrame.from_dict(log_ts)
        self.log_ts.to_csv(path_log_ts, sep="\t")

        ToJson(args, self.modeldir+"/params_%s_trial%d.json" %(target, cid))
        torch.save(mpn.to('cpu').state_dict(), self.modeldir+"/mpn_%s_trial%d.pth" %(target, cid))
        torch.save(dnn.to('cpu').state_dict(), self.modeldir+"/dnn_%s_trial%d.pth" %(target, cid))
    
    
class ACPredictionModel():
    
    def __init__(self, target, debug=False) -> None:
        
        self.target     = target
        self.debug      = debug
        self.output_act = nn.Sigmoid()
        self.cuda       = torch.cuda.is_available()
        self.device     = 'cuda' if self.cuda else 'cpu'
        self.nepoch     = self._Setnepoch()
        self.fixed_arg  = self._set_fixedargs()
        self.nfold      = 3
        
        if self.cuda:
            self.output_act = self.output_act.cuda()

    def _Setnepoch(self):
        
        if self.debug:
            nepoch = [10, 10, 2]
        else:
            nepoch = [50, 100, 100]
            
        return nepoch
    
    def _set_fixedargs(self):
        
        args = dict(
                    train_num    = self.nepoch[0],
                    batch_size   = 10, #trial.suggest_categorical('batch_size', [64, 128, 256])
                    #lr           = 0.0001, #trial.suggest_log_uniform('adam_lr', 1e-4, 1e-2)                   
                    agg_depth    = 1,
                    cuda         = self.cuda,
                    gamma        = 0.1,
                    )
        
        return args
     
    def _set_arg_dict(self, opt_args):
        
        args = self.fixed_arg.copy()
        args.update(opt_args)
        
        args['batch_size'] = 128
            
        args['step_size']  = int(args['train_num']/args['step_num'])
        args['grad_node']  = int(args['dim'] / args['DNNLayerNum'])
        args['node_list']  = [int(args['dim'] - args['grad_node']*num) for num in range(args['DNNLayerNum'])] + [1]
        
        return args


    
    def _AllMMSPred_Load(self, target):
        
        if self._IsPredictableTarget():
            # Initialize
            log = defaultdict(list) 

            for cid in self.testsetidx:    
                
                if self.debug:
                    if cid>2:
                        break
                
                # Train test split
                print("    $ Prediction for cid%d is going on.\n" %cid)
                tr, ts, df_trX, df_tsX = self._GetTrainTest(cid)
                trX, trY = df_trX[self.col], tr['class'].to_numpy()
                tsX, tsY = df_tsX[self.col], ts['class'].to_numpy()
                
                if self._IsPredictableSeries(tr, ts, min_npos=self.nfold):
                    # Fit and Predict
                    p_args = './Log_wodirection_trtssplit/MPNN/Models/params_%s_trial%d.json' %(target, cid)
                    best_args   = LoadJson(p_args)
                    best_args['cuda'] = False
                    
                    self.w_pos = int(np.where(trY==-1)[0].shape[0] / np.where(trY==1)[0].shape[0])
                    self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([self.w_pos, 1]))
                    
                    if best_args['cuda']:
                        self.loss_fn = self.loss_fn.cuda()
                        
                    self.output_act = nn.Sigmoid()
                    
                    self._fit_bestparams(best_args, trX, trY)
                    score, predY, proba = self._predict_bestparams(best_args, tsX, tsY)
                    print("    $ Prediction Done.\n")
                    
                    # Write & save log
                    log = self._WriteLog(log, cid, tr, ts, tsY, predY, proba)           
                    self._Save(target, cid, log, best_args)
                    print("    $  Log is out.\n")


def predict(mpn, dnn, X):
        
    out  = mpn(X)
    pred = dnn(out)
    
    return pred


def predict_proba(X):
    output_act = nn.Sigmoid()
    out = output_act(X)
    
    return out

    
def objective(trial, info, trX, trY):
    
    optarg = dict(
                ConvNum      = trial.suggest_int('ConvNum', 2, 4), #depth
                dropout      = trial.suggest_discrete_uniform('dropout', 0.1, 0.3, 0.05),
                dim          = int(trial.suggest_discrete_uniform("dim", 70, 100, 10)),#hidden_dim
                step_num     = trial.suggest_int("step_num", 1, 3),
                DNNLayerNum  = trial.suggest_int('DNNLayerNum', 2, 8),  
                lr           = trial.suggest_loguniform('lr', 1e-4, 1e-2),              
                )
    
    args = info._set_arg_dict(optarg)
    
    # Set up dataloader
    dataset       = Dataset(smi_list=trX, label_list=trY)
    kfold         = StratifiedKFold(n_splits=info.nfold, shuffle=True, random_state=0)
    score_cv      = []
    
    # Cross validation
    for _fold, (idx_tr, idx_vl) in enumerate(kfold.split(trX, trY)):
        dataset_tr    = Subset(dataset, idx_tr)
        dataloader_tr = DataLoader(dataset_tr, args['batch_size'], shuffle=True,  collate_fn=mycollate_fn)
        dataset_vl    = Subset(dataset, idx_vl)
        dataloader_vl = DataLoader(dataset_vl, args['batch_size'], shuffle=False, collate_fn=mycollate_fn)
        
        # training
        mpn = MPNEncoder(args)
        dnn = DeepNeuralNetwork(args)
        
        opt_list  = list(mpn.parameters()) + list(dnn.parameters())
        optimizer = torch.optim.Adam(opt_list, lr=args['lr'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args['step_size'], gamma=args['gamma'])
        w_pos     = int(np.where(trY[idx_tr]==0)[0].shape[0] / np.where(trY[idx_tr]==1)[0].shape[0])
        loss_fn   = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([w_pos]))

        EPOCH = info.nepoch[0]
        for step in range(EPOCH):
            mpn, dnn = train(info, mpn, dnn, dataloader_tr, optimizer, loss_fn)
            predY_score, predY, proba = test(info, mpn, dnn, dataloader_vl, loss_fn)
            score = roc_auc_score(y_true=trY[idx_vl], y_score=proba)
            scheduler.step()
            
        score_cv.append(score)
    
    return np.mean(score_cv)
    
                        
def train(info, mpn, dnn, dataloader, optimizer, loss_fn, verbose=1):
        
    device   = info.device

    if info.cuda:
        mpn = mpn.cuda()
        dnn = dnn.cuda()
        loss_fn = loss_fn.cuda()
        
    size = len(dataloader.dataset)
    
    n_usedtr = 0
    for batch, (X, y) in enumerate(dataloader):
            
        # X, y = X.to(device), y.to(device)
        X = BatchMolGraph(X)
        
        score = predict(mpn, dnn, X)
        loss  = loss_fn(score, y.to(device))
        
        # back propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #verbose
        n_usedtr += y.shape[0]
        if isinstance(verbose, int) & (batch % verbose == 0):
            loss, current = loss.item(), batch * 128
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]", 'target: %s'%info.target)
            
    return mpn, dnn
                

def test(info, mpn, dnn, dataloader, loss_fn):
    
    device = info.device
    
    mpn.eval()
    dnn.eval()
    
    size      = len(dataloader.dataset)
    threshold = torch.tensor([0.5]).to(device)
    test_loss = 0
    pred_score_all, pred_all, proba_all, y_all = [], [], [], []
    
    with torch.no_grad():
        for X, y in dataloader:
            X = BatchMolGraph(X)
            
            pred_score  = predict(mpn, dnn, X)
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


def fit_bestparams(info, args, trX, trY, loss_fn):
    
    dataloader_tr = DataLoader(Dataset(smi_list=trX, label_list=trY),
                                batch_size = args['batch_size'],
                                shuffle    = True, 
                                collate_fn = mycollate_fn
                                )
    
    mpn   = MPNEncoder(args)
    dnn   = DeepNeuralNetwork(args)
    
    opt_list  = list(mpn.parameters()) + list(dnn.parameters())
    optimizer = torch.optim.Adam(opt_list, lr=args['lr'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args['step_size'], gamma=args['gamma'])
    
    EPOCH = info.nepoch[1]
    for step in range(EPOCH):
        mpn, dnn = train(info, mpn, dnn, dataloader_tr, optimizer, loss_fn)
        scheduler.step()
        
    return mpn, dnn
        

def predict_bestparams(info, mpn, dnn, args, tsX, tsY, loss_fn):
    
    dataloader_ts = DataLoader(Dataset(smi_list=tsX, label_list=tsY),
                                args['batch_size'],
                                shuffle=False,
                                collate_fn=mycollate_fn
                                )
    
    pred_score, pred, proba = test(info, mpn, dnn, dataloader_ts, loss_fn)
    
    return pred_score, pred, proba

    
def main(target, bd, debug=False):
    
    #Initialize   
    model = "MPNN"
    mtype = "wodirection_trtssplit"
    
    if debug:
        mtype +='_debug'
    
    os.chdir(bd)
    os.makedirs("./Log_%s"%mtype, exist_ok=True)
    os.makedirs("./Score_%s"%mtype, exist_ok=True)
    
    p = Classification(target     = target, 
                       modeltype  = mtype,
                       model      = model,
                       dir_log    = './Log_%s/%s' %(mtype, model),
                       dir_score  = './Score_%s/%s' %(mtype, model),
                       )
    
    info = ACPredictionModel(target=target, debug=debug)
    
    if not p._IsPredictableSet():
        print('    $ %s is skipped because of lack of the actives' %target)
    
    else:    
        if p._IsPredictableTarget():
            p._SetParams()
            for loop in p.testsetidx:    
                
                if p.debug:
                    if loop > 2:
                        break
                
                # Train test split
                print("    $ Prediction for loop%d is going on.\n" %loop)
                tr, ts = p._GetTrainTest(loop)
                trX, trY = tr[p.col], tr['class'].to_numpy()
                tsX, tsY = ts[p.col], ts['class'].to_numpy()
                
                if p._IsPredictableSeries(tr, ts, min_npos=p.nfold):
                    
                    # Fit and Predict
                    pruner = optuna.pruners.MedianPruner()
                    study  = optuna.create_study(pruner=pruner, direction='maximize')
                    obj    = partial(objective, info=info, trX=trX, trY=trY)
                    study.optimize(obj, n_trials=info.nepoch[2])
                    
                    best_args = info._set_arg_dict(study.best_params)
                    w_pos     = int(np.where(trY==0)[0].shape[0] / np.where(trY==1)[0].shape[0])
                    loss_fn   = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([w_pos]))
                    mpn, dnn  = fit_bestparams(info, best_args, trX, trY, loss_fn)
                    score_tr, predY_tr, proba_tr = predict_bestparams(info, mpn, dnn, best_args, trX, trY, loss_fn)
                    score_ts, predY_ts, proba_ts = predict_bestparams(info, mpn, dnn, best_args, tsX, tsY, loss_fn)
                    print("    $ Prediction Done.\n")
                    
                    # Write & save log
                    log_tr = p.WriteLog_tr(loop, tr, trY, predY_tr, proba_tr) 
                    log_ts = p.WriteLog_ts(loop, tr, ts, tsY, predY_ts, proba_ts)  
                    p.Save(target, loop, log_tr, log_ts, best_args, mpn, dnn)
                    print("    $  Log is out.\n")
                    
            
if __name__ == '__main__':    
    
    bd = '/home/bit/tamuras0/ACPredCompare'#'/home/tamuras0/work/ACPredCompare'
    tlist = pd.read_csv('./Dataset/target_list.tsv', sep='\t', index_col=0)
    tlist = tlist.loc[tlist['machine1'],:]
    
    debug = False
    
    # main('CHEMBL4072', bd, debug)
    
    # for i, sr in tlist.iterrows():
    #     target = sr['chembl_tid']   
    
    #     print("\n----- %s is proceeding -----\n" %target)
    #     main(target, bd, debug)
    
    obj = partial(main, bd=bd, debug=debug)
    joblib.Parallel(n_jobs=-1, backend='loky')(joblib.delayed(obj)(target) for target in tlist['chembl_tid'])
from importlib_metadata import functools
import pandas as pd
import numpy as np
import os
import joblib
import re
from collections import namedtuple
from sklearn.model_selection import StratifiedShuffleSplit
import sys
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

def ReplaceRgroups2Astarisk(smi):
    smi = re.compile(r'R\d').sub('*', smi)
    smi = smi.replace('[*]', '*')

    return smi


class Dataset(torch.utils.data.Dataset):
    
    def __init__(self, df_smi, label_list):
        
        # Negative label is converted into 0 (originally -1)
        if -1 in label_list:    
            label_list[np.where(label_list==-1)[0]] = 0
            
        self.smi_list   = df_smi.values
        self.label_list = label_list
        self.core       = df_smi['core'].values
        self.sub1       = df_smi['sub1'].values
        self.sub2       = df_smi['sub2'].values
        
    def __len__(self):
        return self.label_list.shape[0]

    def __getitem__(self, idx):
        return self.core[idx], self.sub1[idx], self.sub2[idx], self.label_list[idx]
  
    
def mycollate_fn(batch):
    
    batch_list = list(zip(batch))
    mol_c  = []
    mol_s1 = []
    mol_s2 = []
    labels = []
    
    for c, s1, s2, label in batch:
        mol_c.append(MolGraph(ReplaceRgroups2Astarisk(c)))
        mol_s1.append(MolGraph(ReplaceRgroups2Astarisk(s1)))
        mol_s2.append(MolGraph(ReplaceRgroups2Astarisk(s2)))
        labels.append(label)
    
    labels = torch.FloatTensor(labels).reshape([-1, 1])
        
    return mol_c, mol_s1, mol_s2, labels


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

    def __init__(self, target, modeltype, model, dir_log, dir_score, data_split_metric='axv', debug=False):

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
        
        
    def Save(self, target, seed, log_tr, log_cpdout, log_bothout, args_mpn_c, args_mpn_s, args_dnn, mpn_c, mpn_s1, mpn_s2, dnn):
        path_log_tr = os.path.join(self.logdir, "%s_Seed%d_train.tsv" %(target, seed))
        self.log_tr = pd.DataFrame.from_dict(log_tr)
        self.log_tr.to_csv(path_log_tr, sep="\t")
        
        path_log_cpdout = os.path.join(self.logdir, "%s_Seed%d_cpdout.tsv" %(target, seed))
        self.log_cpdout = pd.DataFrame.from_dict(log_cpdout)
        self.log_cpdout.to_csv(path_log_cpdout, sep="\t")
        
        path_log_bothout = os.path.join(self.logdir, "%s_Seed%d_bothout.tsv" %(target, seed))
        self.log_bothout = pd.DataFrame.from_dict(log_bothout)
        self.log_bothout.to_csv(path_log_bothout, sep="\t")

        
        args = dict(mpn_c = args_mpn_c, mpn_s = args_mpn_s, dnn=args_dnn)
        ToJson(args, self.modeldir+"/params_%s_Seed%d.json" %(target, seed))
        
        torch.save(mpn_c.to('cpu').state_dict() , self.modeldir+"/mpn_c_%s_Seed%d.pth" %(target, seed))
        torch.save(mpn_s1.to('cpu').state_dict(), self.modeldir+"/mpn_s1_%s_Seed%d.pth" %(target, seed))
        torch.save(mpn_s2.to('cpu').state_dict(), self.modeldir+"/mpn_s2_%s_Seed%d.pth" %(target, seed))
        torch.save(dnn.to('cpu').state_dict()   , self.modeldir+"/dnn_%s_Seed%d.pth" %(target, seed))
        
        
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
                    batch_size   = 128, #trial.suggest_categorical('batch_size', [64, 128, 256])
                    #lr           = 0.0001, #trial.suggest_log_uniform('adam_lr', 1e-4, 1e-2)                   
                    agg_depth    = 1,
                    cuda         = self.cuda,
                    gamma        = 0.1,
                    )
        
        return args
     
    def _set_arg_dict(self, opt_args):
        
        args = self.fixed_arg.copy()
        args.update(opt_args)
        
        return args
    

def objective(trial, info, trX, trY):
        
    args_mpn_c = dict(
                      dim          = int(trial.suggest_discrete_uniform("dim", 70, 100, 10)),#hidden_dim
                      ConvNum      = trial.suggest_int('ConvNum', 2, 4), #depth
                      )
    
    args_mpn_c = info._set_arg_dict(args_mpn_c)
    
    args_mpn_s = dict(
                      dim          = int(trial.suggest_discrete_uniform("dim", 70, 100, 10)),#hidden_dim
                      ConvNum      = trial.suggest_int('ConvNum', 2, 4), #depth
                      )
    
    args_mpn_s = info._set_arg_dict(args_mpn_s)
    
    args_dnn = dict(
                    dropout      = trial.suggest_discrete_uniform('dropout', 0.1, 0.3, 0.05),
                    step_num     = trial.suggest_int("step_num", 1, 3),
                    DNNLayerNum  = trial.suggest_int('DNNLayerNum', 2, 8),
                    lr           = trial.suggest_loguniform('lr', 1e-4, 1e-2)
                    )
    args_dnn = info._set_arg_dict(args_dnn)
    args_dnn['dim']        = int(args_mpn_c['dim']+2*args_mpn_s['dim'])
    args_dnn['step_size']  = int(args_dnn['train_num']/args_dnn['step_num'])
    args_dnn['grad_node']  = int(args_dnn['dim'] / args_dnn['DNNLayerNum'])
    args_dnn['node_list']  = [int(args_dnn['dim'] - args_dnn['grad_node']*num) for num in range(args_dnn['DNNLayerNum'])] + [1]
    
    # Set up dataloader
    dataset       = Dataset(df_smi=trX, label_list=trY)
    kfold         = StratifiedKFold(n_splits=info.nfold, shuffle=True, random_state=0)
    score_cv      = []
    
    # Cross validation
    for _fold, (idx_tr, idx_vl) in enumerate(kfold.split(trX, trY)):
        dataset_tr    = Subset(dataset, idx_tr)
        dataloader_tr = DataLoader(dataset_tr, args_dnn['batch_size'], shuffle=True,  collate_fn=mycollate_fn)
        dataset_vl    = Subset(dataset, idx_vl)
        dataloader_vl = DataLoader(dataset_vl, args_dnn['batch_size'], shuffle=False, collate_fn=mycollate_fn)
        
        # training
        mpn_c  = MPNEncoder(args_mpn_c)
        mpn_s1 = MPNEncoder(args_mpn_s)
        mpn_s2 = MPNEncoder(args_mpn_s)
        dnn    = DeepNeuralNetwork(args_dnn)
        
        opt_list  = list(mpn_c.parameters()) + list(mpn_s1.parameters()) + list(mpn_s2.parameters()) +list(dnn.parameters())
        optimizer = torch.optim.Adam(opt_list, lr=args_dnn['lr'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args_dnn['step_size'], gamma=args_dnn['gamma'])
        w_pos     = int(np.where(trY[idx_tr]==0)[0].shape[0] / np.where(trY[idx_tr]==1)[0].shape[0])
        loss_fn   = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([w_pos]))

        EPOCH = info.nepoch[0]
        for step in range(EPOCH):
            mpn_c, mpn_s1, mpn_s2, dnn = train(info, mpn_c, mpn_s1, mpn_s2, dnn, dataloader_tr, optimizer, loss_fn)
            predY_score, predY, proba  = test(info, mpn_c, mpn_s1, mpn_s2, dnn, dataloader_vl, loss_fn)
            score = roc_auc_score(y_true=trY[idx_vl], y_score=proba)
            scheduler.step()
            
        score_cv.append(score)
    
    return np.mean(score_cv)


def train(info, mpn_c, mpn_s1, mpn_s2, dnn, dataloader, optimizer, loss_fn, verbose=1):
        
    device   = info.device

    if info.cuda:
        mpn_c   = mpn_c.cuda()
        mpn_s1  = mpn_s1.cuda()
        mpn_s2  = mpn_s2.cuda()
        dnn     = dnn.cuda()
        loss_fn = loss_fn.cuda()
        
    size = len(dataloader.dataset)
    
    n_usedtr = 0
    for batch, (X_c, X_s1, X_s2, y) in enumerate(dataloader):
            
        # X, y = X.to(device), y.to(device)
        X_c  = BatchMolGraph(X_c)
        X_s1 = BatchMolGraph(X_s1)
        X_s2 = BatchMolGraph(X_s2)
        
        score = predict(mpn_c, mpn_s1, mpn_s2, dnn, X_c, X_s1, X_s2)
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
            
    return mpn_c, mpn_s1, mpn_s2, dnn


def test(info, mpn_c, mpn_s1, mpn_s2, dnn, dataloader, loss_fn):
    
    device = info.device
    
    mpn_c.eval()
    mpn_s1.eval()
    mpn_s2.eval()
    dnn.eval()
    
    size      = len(dataloader.dataset)
    threshold = torch.tensor([0.5]).to(device)
    test_loss = 0
    pred_score_all, pred_all, proba_all, y_all = [], [], [], []
    
    with torch.no_grad():
        for X_c, X_s1, X_s2, y in dataloader:
            X_c  = BatchMolGraph(X_c)
            X_s1 = BatchMolGraph(X_s1)
            X_s2 = BatchMolGraph(X_s2)
            
            pred_score  = predict(mpn_c, mpn_s1, mpn_s2, dnn, X_c, X_s1, X_s2)
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


def fit_bestparams(info, args_dnn, args_mpn_c, args_mpn_s, trX, trY, loss_fn):
    
    dataloader_tr = DataLoader(Dataset(df_smi=trX, label_list=trY),
                                batch_size = args_dnn['batch_size'],
                                shuffle    = True, 
                                collate_fn = mycollate_fn
                                )
    
    mpn_c  = MPNEncoder(args_mpn_c)
    mpn_s1 = MPNEncoder(args_mpn_s)
    mpn_s2 = MPNEncoder(args_mpn_s)
    dnn    = DeepNeuralNetwork(args_dnn)
    
    opt_list  = list(mpn_c.parameters()) + list(mpn_s1.parameters()) + list(mpn_s2.parameters()) + list(dnn.parameters())
    optimizer = torch.optim.Adam(opt_list, lr=args_dnn['lr'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args_dnn['step_size'], gamma=args_dnn['gamma'])
    
    EPOCH = info.nepoch[1]
    for step in range(EPOCH):
        mpn_c, mpn_s1, mpn_s2, dnn = train(info, mpn_c, mpn_s1, mpn_s2, dnn, dataloader_tr, optimizer, loss_fn)
        scheduler.step()
        
    return mpn_c, mpn_s1, mpn_s2, dnn
        

def predict_bestparams(info, mpn_c, mpn_s1, mpn_s2, dnn, args, tsX, tsY, loss_fn):
    
    dataloader_ts = DataLoader(Dataset(df_smi=tsX, label_list=tsY),
                                args['batch_size'],
                                shuffle=False,
                                collate_fn=mycollate_fn
                                )
    
    pred_score, pred, proba = test(info, mpn_c, mpn_s1, mpn_s2, dnn, dataloader_ts, loss_fn)
    
    return pred_score, pred, proba


def predict(mpn_c, mpn_s1, mpn_s2, dnn, X_c, X_s1, X_s2):
        
    out_c  = mpn_c(X_c)
    out_s1 = mpn_s1(X_s1)
    out_s2 = mpn_s2(X_s2)
    out    = torch.cat((out_c, out_s1, out_s2), dim=1)
    pred   = dnn(out)
    
    return pred


def predict_proba(X):
    
    output_act = nn.Sigmoid()
    out = output_act(X)
    
    return out
      
                
def main(target, bd, debug=False):
    
    #Initialize   
    model = "MPNN_separated"
    mtype = "axv"
    
    if debug:
        mtype +='_debug'
    
    os.chdir(bd)
    os.makedirs("./Log_%s"%mtype, exist_ok=True)
    os.makedirs("./Score_%s"%mtype, exist_ok=True)
    
    cols = ['core', 'sub1', 'sub2']
    
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
            p._SetParams(target)
            for loop in p.testsetidx:    
                
                if p.debug:
                    if loop > 2:
                        break
                
                # Train test split
                print("    $ Prediction for loop%d is going on.\n" %loop)
                tr, cpdout, bothout = p._GetTrainTest(loop)
                trX, trY = tr[cols], tr['class'].to_numpy()
                cpdoutX , cpdoutY  = cpdout[cols], cpdout['class'].to_numpy()
                bothoutX, bothoutY = bothout[cols], bothout['class'].to_numpy()
                
                flag_predictable = p._IsPredictableSeries(tr, cpdout, min_npos=p.nfold) * p._IsPredictableSeries(tr, bothout, min_npos=p.nfold)
                if flag_predictable:
                    
                    # Fit and Predict
                    pruner = optuna.pruners.MedianPruner()
                    study  = optuna.create_study(pruner=pruner, direction='maximize')
                    obj    = partial(objective, info=info, trX=trX, trY=trY)
                    study.optimize(obj, n_trials=info.nepoch[2])
                    
                    best_args = study.best_params
                    
                    args_mpn_c = dict(dim = best_args['dim'], ConvNum = best_args['ConvNum'])
                    args_mpn_c = info._set_arg_dict(args_mpn_c)
                    
                    args_mpn_s = dict(dim = best_args['dim'], ConvNum = best_args['ConvNum'])
                    args_mpn_s = info._set_arg_dict(args_mpn_s)
                    
                    args_dnn = dict(dropout = best_args['dropout'], step_num = best_args["step_num"], DNNLayerNum = best_args['DNNLayerNum'], lr = best_args['lr'])
                    args_dnn = info._set_arg_dict(args_dnn)
                    args_dnn['dim']        = int(args_mpn_c['dim']+2*args_mpn_s['dim'])
                    args_dnn['step_size']  = int(args_dnn['train_num']/args_dnn['step_num'])
                    args_dnn['grad_node']  = int(args_dnn['dim'] / args_dnn['DNNLayerNum'])
                    args_dnn['node_list']  = [int(args_dnn['dim'] - args_dnn['grad_node']*num) for num in range(args_dnn['DNNLayerNum'])] + [1]
                                    
                    w_pos     = int(np.where(trY==0)[0].shape[0] / np.where(trY==1)[0].shape[0])
                    loss_fn   = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([w_pos]))
                    mpn_c, mpn_s1, mpn_s2, dnn  = fit_bestparams(info, args_dnn, args_mpn_c, args_mpn_s, trX, trY, loss_fn)
                    score_tr     , predY_tr     , proba_tr      = predict_bestparams(info, mpn_c, mpn_s1, mpn_s2, dnn, args_dnn, trX, trY, loss_fn)
                    score_cpdout , predY_cpdout , proba_cpdout  = predict_bestparams(info, mpn_c, mpn_s1, mpn_s2, dnn, args_dnn, cpdoutX, cpdoutY, loss_fn)
                    score_bothout, predY_bothout, proba_bothout = predict_bestparams(info, mpn_c, mpn_s1, mpn_s2, dnn, args_dnn, bothoutX, bothoutY, loss_fn)
                    print("    $ Prediction Done.\n")
                    
                    # Write & save log
                    log_tr = p.WriteLog_tr(loop, tr, trY, predY_tr, proba_tr) 
                    log_cpdout = p.WriteLog_ts(loop, tr, cpdout, cpdoutY, predY_cpdout, proba_cpdout)
                    log_bothout = p.WriteLog_ts(loop, tr, bothout, bothoutY, predY_bothout, proba_bothout)  
                    p.Save(target, loop, log_tr, log_cpdout, log_bothout, args_mpn_c, args_mpn_s, args_dnn, mpn_c, mpn_s1, mpn_s2, dnn)
                    print("    $  Log is out.\n")
                    
if __name__ == '__main__':    
    
    #bd = '/home/bit/tamuras0/ACPredCompare'
    bd = '/home/tamuras0/work/ACPredCompare'
    tlist = pd.read_csv('./Dataset/target_list.tsv', sep='\t', index_col=0)
    #tlist = tlist.loc[tlist['machine2'],:]
    
    debug = False
    
    # main('CHEMBL4072', bd, debug)
    
    # for i, sr in tlist.iterrows():
    #     target = sr['chembl_tid']   
    
    #     print("\n----- %s is proceeding -----\n" %target)
    #     main(target, bd, debug)
    
    obj = partial(main, bd=bd, debug=debug)
    joblib.Parallel(n_jobs=10, backend='loky')(joblib.delayed(obj)(target) for target in tlist['chembl_tid'])
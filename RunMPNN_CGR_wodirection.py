import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import optuna
import torch
from torch                             import nn
from torch.utils.data                  import dataloader
from torch_geometric.data              import Data
from torch_geometric.nn                import MessagePassing, Sequential, MFConv
from torch_geometric.utils             import add_self_loops, degree
from Tools.ReadWrite                   import ToJson
from collections                       import OrderedDict, defaultdict
from functools                         import partial
from sklearn.model_selection           import train_test_split
from sklearn.metrics                   import roc_auc_score
from GCN.MolGraph                      import MolGraph
from .BaseFunctions                    import Base_wodirection

class Dataset(torch.utils.data.Dataset):
    
    def __init__(self, smi_list, label_list):
        self.smi_list   = smi_list
        self.label_list = label_list
        
    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return self.smi_list[idx], self.label_list[idx]
  
    
def mycollate_fn(batch):
    
    batch_list = list(zip(batch))
    pyg_data   = []
    
    for smi, label in batch_list:
        mol_inf = MolGraph(smi)
        data    = Data(x          = mol_inf.f_atoms, 
                       edge_index = mol_inf.edge_index,
                       edge_attr  = mol_inf.f_bonds,
                       y          = label
                       )
        pyg_data.append(data)
        
    return pyg_data

   
class FullyConnectedNN(nn.Module):
    
    def __init__(self, nbits, hidden_nodes, drop_rate):
        super(FullyConnectedNN, self).__init__()
        
        self.nbits = nbits
        self.nlayer = len(hidden_nodes)
        self.linear_relu_stack = nn.Sequential(OrderedDict([
                                                ('input'    , nn.Linear(self.nbits,hidden_nodes[0])),
                                                ('relu'   , nn.ReLU()),                                                
                                                ]))
        
        for i in range(1, self.nlayer):
            self.linear_relu_stack.add_module('hidden_layer%d'%(i) , nn.Linear(hidden_nodes[i-1], hidden_nodes[i]))
            self.linear_relu_stack.add_module('relu'   , nn.ReLU())
            self.linear_relu_stack.add_module('dropout', nn.Dropout(drop_rate))
                
        self.linear_relu_stack.add_module('output', nn.Linear(hidden_nodes[i], 1))
        self.linear_relu_stack.add_module('sigmoid', nn.Sigmoid())
        
    def forward(self, x):
        signal = self.linear_relu_stack(x)
        return signal    
    
    
class Classification(Base_wodirection):

    def __init__(self, modeltype, model, dir_log, dir_score, debug=False):

        super().__init__(modeltype, dir_log=dir_log, dir_score=dir_score)

        self.pred_type = "classification"
        self.mname     = model
        self.debug     = True
        self.n_epoch   = self._set_nepoch()
    
        
    def _set_nepoch(self):
        
        if self.debug:
            epoch = [2, 2]
        else:
            epoch = [50, 200]
            
        return epoch
    
    
    def train(self, model, device, loss_fn, optimizer, dataloader, verbose=10):
        
        model.train()
        size = len(dataloader.dataset)
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            
            # Calculate loss
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
                

    def test(self, model, device, loss_fn, dataloader):
        
        size = len(dataloader.dataset)
        model.eval()
        threshold = torch.tensor([0.5])
        test_loss, correct = 0, 0
        
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                pred_score = model(X)
                pred       = (pred_score>threshold).float()*1
                test_loss += loss_fn(pred, y).item()
                correct   += (pred.argmax(1) == y).type(torch.float).sum().item()
                
        test_loss /= size
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        
        return pred_score.cpu().detach().numpy(), pred.cpu().detach().numpy()

            
    def objective(self, trial, trX, trY):
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        EPOCH        = self.n_epoch[0]
        nbits        = trX.shape[1]
        n_layer      = trial.suggest_int('n_layer', 4, 8)
        hidden_nodes = [int(trial.suggest_discrete_uniform("num_filter_"+str(i), 250, 2000, step=250)) for i in range(n_layer)]
        drop_rate    = trial.suggest_discrete_uniform('drop_rate', 0.25, 0.50, step=0.25)
        adam_lr      = trial.suggest_log_uniform('adam_lr', 1e-4, 1e-2)
        batch_size   = trial.suggest_categorical('batch_size', [64, 128, 256])
        
        loss_fn   = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(lr=adam_lr)
        model     = nn.Sequential('x, edge_index', [GCNConv(in_channel, out_channel), FullyConnectedNN(nbits, hidden_nodes, drop_rate)]).to(device)
        
        X_tr, X_vl, y_tr, y_vl = train_test_split(trX, trY, test_size=0.2, random_state=42, shuffle=True, stratify=trY)
        dataloader_tr = dataloader(Dataset(fpset=X_tr, label=y_tr), batch_size=batch_size)
        dataloader_vl = dataloader(Dataset(fpset=X_vl, label=y_vl))
        
        for step in range(EPOCH):
            self.train(model, device, loss_fn, optimizer, dataloader_tr)
            predY_score, predY = self.test(model, device, loss_fn, dataloader_vl)
            score = roc_auc_score(y_true=y_vl, y_score=predY_score)
                
        return score

    def _fit_bestparams(self, params, trX, trY):
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        EPOCH         = self.n_epoch[1]
        nbits         = trX.shape[1]
        model         = FullyConnectedNN(nbits=nbits, **params)
        dataloader_tr = dataloader(Dataset(fpset=trX, label=trY), batch_size=params['batch_size'])
        loss_fn       = nn.CrossEntropyLoss()
        optimizer     = torch.optim.Adam(lr=params['adam_lr'])
        
        for step in range(EPOCH):
            self.train(model, device, loss_fn, optimizer, dataloader_tr)
            
        return model

    def _predict_bestparams(self, model, tsX, tsY):
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        dataloader_ts = dataloader(Dataset(fpset=tsX, label=tsY))
        loss_fn       = nn.CrossEntropyLoss()
        
        pred_score, pred = self.test(model, device, loss_fn, dataloader_ts)
        
        return pred_score.cpu().detach().numpy(), pred.cpu().detach().numpy()
    
    
    def _AllMMSPred(self, target, path_log):
        
        # Initialize
        log = defaultdict(list) 

        for cid in self.testsetidx:    
            
            if self.debug:
                if cid>2:
                    break
            
            # Train test split
            print("    $ Prediction for cid%d is going on.\n" %cid)
            tr, ts = self._TrainTestSplit_main(cid)
            
            # Fit and Predict
            study = optuna.create_study()
            study.optimize(self.objective, n_trials=100)
            
            ml = self._fit_bestparams(study.best_params, trX, trY)
            score, predY = self._predict_bestparams(ml, tsX, tsY)
            print("    $ Prediction Done.\n")
            
            # Write & save log
            log = self._WriteLog(log, ml, cid, ts, tsX, tsY, predY, score)           
            self._Save(target, cid, log, ml, study)
            print("    $  Log is out.\n")
            
            
    def _WriteLog(self, log, ml, cid, ts, tsX, tsY, predY, score):
        
        # Write log
        tsids         = ts["id"].tolist()
        log["ids"]   += tsids
        log["cid"]   += [cid] * len(tsids)
        log["trueY"] += tsY.tolist()
        log["predY"] += predY.tolist()
        log["prob"]  += score
        
        
    def _Save(self, target, cid, log, ml, study):
        path_log = os.path.join(self.logdir, "%s_trial%d.tsv" %(target, cid))
        self.log = pd.DataFrame.from_dict(log)
        self.log.to_csv(path_log, sep="\t")

        ToJson(study.best_params, self.modeldir+"/params_%s_trial%d.json" %(target, cid))
        torch.save(ml.to('cpu').state_dict(), self.modeldir+"/model_%s_trial%d.pth" %(target, cid))
        print("    $  Log is out.\n")


if __name__ == "__main__":
    
    bd    = "/home/tamuras0/work/ACPredCompare/"
    model = "FCNN"
    mtype = "wodirection"
    os.chdir(bd)
    os.makedirs("./Log", exist_ok=True)
    os.makedirs("./Score", exist_ok=True)
    
    tlist = pd.read_csv('./Dataset/target_list.tsv', sep='\t', index_col=0)
    
    for i, sr in tlist.iterrows():
        
        target = sr['target']
        
        p = Classification(modeltype   = mtype,
                        model       = model,
                        dir_log     = "./Log/%s" %(model+'_'+mtype),
                        dir_score   = "./Score/%s" %(model+'_'+mtype),
                        interpreter = "shap",
                        aconly      = False,
                        )

        p.run(target=target, debug=True)

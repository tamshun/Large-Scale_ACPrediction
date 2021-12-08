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
import matplotlib.pyplot as plt
import torch
from torch                             import nn
from torch.nn.modules.container        import Sequential
from tqdm                              import tqdm
from MMP.make_input                    import LeaveOneCoreOut, GetDiverseCore, DelTestCpdFromTrain
from Tools.ReadWrite                   import ReadDataAndFingerprints, ToPickle, LoadPickle
from Tools.utils                       import BasicInfo
#from Metrix.scores                     import ScoreTable,RenderDF2Fig
from Evaluation.Labelling              import AddLabel
from Evaluation.Score                  import ScoreTable_wDefinedLabel
from collections                       import OrderedDict, defaultdict
from Plots.Barplot                     import MakeBarPlotsSeaborn as bar
from Plots.Lineplot                    import MakeLinePlotsSeaborn as line
from Fingerprint.Hash2BitManager       import Hash2Bits, FindBitLength
from functools                         import partial
from collections                       import defaultdict, OrderedDict


device = 'cpu' 
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam()

class FullyConnectedNN(nn.Module):
    
    def __init__(self, nbits, hidden_nodes, drop_rate):
        super(FullyConnectedNN, self).__init__()
        
        self.nbits = nbits
        self.nlayer = len(hidden_nodes)
        self.linear_relu_stack = nn.Sequential(OrderedDict([
                                                ('fc1'    , nn.Linear(self.nbits,hidden_nodes[0])),
                                                ('relu'   , nn.ReLU()),
                                                ('dropout', nn.Dropout(drop_rate)),
                                                ('fc2'    , nn.Linear(hidden_nodes[0], hidden_nodes[1])),
                                                ('relu'   , nn.ReLU()), 
                                                ('dropout', nn.Dropout(drop_rate)),
                                                ]))
        
        if self.nlayer > 2:
            for i in range(2, self.nlayer):
                self.linear_relu_stack.add_module('fc%d'%i , nn.Linear(hidden_nodes[i-1], hidden_nodes[i]))
                self.linear_relu_stack.add_module('relu'   , nn.ReLU())
                self.linear_relu_stack.add_module('dropout', nn.Dropout(drop_rate))
                
        self.linear_relu_stack.add_module('fc%d'%self.nlayer, nn.Linear(hidden_nodes[-1], 1))
        self.linear_relu_stack.add_module('sigmoid', nn.Sigmoid())
        
    def forward(self, x):
        signal = self.linear_relu_stack(x)
        return signal    
    
    
def train(dataloader, model, loss_fn, optimizer):
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

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            

def test(dataloader, model):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

class Base_ECFPECFP():
    
    def __init__(self, modeltype, model_name, dir_log=None, dir_score=None, interpreter=None, aconly=False, kernel_type="product"):
        #self.bd      = "/home/tamura/work/Interpretability"
        #os.chdir(self.bd)
        self.mtype        = modeltype
        self.mname        = model_name.lower()
        self.logdir, self.scoredir, self.modeldir = self._MakeLogDir(dir_log, dir_score)
        self.debug        = False
        self.interpreter  = interpreter
        self.aconly       = aconly
        self.kernel_type  = kernel_type
        print("$  Decision function will be applied.\n")

        self.col  = ["core", "sub1", "sub2", "overlap"]

        
    def _MakeLogDir(self, logdir, scoredir):
        
        if logdir is None:
            logdir   = "./Results/"
        
        if scoredir is None:
            scoredir = "./Scores/"
        
        modeldir = os.path.join(logdir, "Models")

        os.makedirs(logdir  , exist_ok=True)
        os.makedirs(scoredir, exist_ok=True)
        os.makedirs(modeldir, exist_ok=True)
        
        return logdir, scoredir, modeldir


    def _SetML(self):
    
        
        return model


    def _SetParams(self):

        self.nbits_c = FindBitLength(self.ecfp, [self.col[0]])
        self.nbits_s = FindBitLength(self.ecfp, self.col[1:] )

        # Leave One Core Out
        self.LOCO_generator = LeaveOneCoreOut(self.main)
        self.testidx        = self.LOCO_generator.keys()

        # Leave One Core Out
        if self.debug: 
            self.nsample = 1

        else:
            self.nsample = 50


    def _ReadDataFile(self, target):
  
        main = pd.read_csv("./Dataset/Data/%s.tsv" %target, sep="\t", index_col=0) # Single class MMSs were removed in curation.py.
        ecfp = pd.read_csv("./Dataset/ECFP/%s.tsv" %target, sep="\t", index_col=0)

        return main, ecfp


    def _GetMatrices(self, cid, aconly=False):

        loco     = self.LOCO_generator[cid]
        tr       = self.main.loc[loco.tridx,:]
        ts       = self.main.loc[loco.tsidx,:]
        
        # Check overlap
        tr = DelTestCpdFromTrain(ts, tr, deltype="both")
        
        # Assign pd_sr of fp; [core, sub1, sub2]
        if aconly:
            print(    "Only AC-MMPs are used for making training data.\n")
            tr = tr[tr["class"]==1]

        df_trX = self.ecfp.loc[tr.index, :]
        df_tsX = self.ecfp.loc[ts.index, :]
        df_trY = tr["class"]
        df_tsY = ts["class"]

        forward  = Hash2Bits(subdiff=False, sub_reverse=False)
        trX, trY = forward.GetMMPfingerprints_DF_unfold(df=df_trX, cols=self.col, Y=df_trY, nbits=[self.nbits_c, self.nbits_s], overlap="concat")
        tsX, tsY = forward.GetMMPfingerprints_DF_unfold(df=df_tsX, cols=self.col, Y=df_tsY, nbits=[self.nbits_c, self.nbits_s], overlap="concat")

        return tr, ts, df_trX, df_trY, df_tsX, df_tsY, trX, trY, tsX, tsY
        
    
    def run(self, target, debug=False):
        
        print("\n----- %s is proceeding -----\n" %target)
        path_log = os.path.join(self.logdir, "%s_%s.tsv" %(target, self.mtype))
        
        if debug:
            self.debug=True
        
        self.main, self.ecfp = self._ReadDataFile(target)
        self._SetParams()

        self._AllMMSPred(target, path_log)
        

class Classification(Base_ECFPECFP):

    def __init__(self, modeltype, model, dir_log, dir_score, interpreter, aconly, kernel_type="product"):

        super().__init__(modeltype, model_name=model, dir_log=dir_log, dir_score=dir_score, interpreter=interpreter, aconly=aconly, kernel_type=kernel_type)

        self.pred_type = "classification"
        
    def _AllMMSPred(self, t, path_log):
        
        fcs_log_path = path_log[:-4] + "_all.npy"
        fcs_log = None
            
        # Initialize
        log = defaultdict(list) #MakeLogDict(type=self.pred_type)

        for cid in self.testidx:    
            
            if self.debug:
                if cid>2:
                    break
            
            # Train test split
            print("    $ Prediction for cid%d is going on.\n" %cid)
            tr, ts, df_trX, df_trY, df_tsX, df_tsY, trX, trY, tsX, tsY = self._GetMatrices(cid)
            
            # Fit and Predict
            ml = self._SetML(self.mname, kernel_type=self.kernel_type)
            
            ml.fit(trX, trY)
            predY = ml.predict(tsX)
            print("    $ Prediction Done.\n")

            # Write log
            tsids            = ts["id"].tolist()
            log["ids"]      += tsids
            log["cid"]      += [cid] * len(tsids)
            log["trueY"]    += tsY.tolist()
            log["predY"]    += predY.tolist()
            log["prob"] += [prob[1] for prob in ml.score(tsX).tolist()]
            
            # Save
            self.log = pd.DataFrame.from_dict(log)
            self.log.to_csv(path_log, sep="\t")

            ToPickle(ml, self.modeldir+"/cid%d_forward.pickle"%cid)
            print("    $  Log is out.\n")

    def _AllMMSPred_SimpleTanimoto(self, t, path_log):
        
        fcs_log_path = path_log[:-4] + "_all.npy"
        fcs_log = None
            
        # Initialize
        log = MakeLogDict(type=self.pred_type)

        for cid in self.testidx:    
            
            if self.debug:
                if cid>2:
                    break
            
            # Train test split
            print("    $ Prediction for cid%d is going on.\n" %cid)
            tr, ts, trX, trY, tsX, tsY, trX_f, trY_f, tsX_f, tsY_f, trX_r, trY_r = self._GetMatrices(cid)
            
            # Fit and Predict
            
            
            ml_f.fit(trX_f, trY_f)
            predY_f = ml_f.predict(tsX_f)
            print("    $ Forward Done.\n")

            ml_r.fit(trX_r, trY_r)
            predY_r = ml_r.predict(tsX_f)
            print("    $ Reverce Done.\n")

            
            
            # Write log
            tsids              = ts["id"].tolist()
            log["ids"]        += tsids
            log["cid"]        += [cid] * len(tsids)
            log["trueY"]      += tsY.tolist()
            log["predY_f"]    += predY_f.tolist()
            log["predY_r"]    += predY_r.tolist()
            log["prob_f"] += ml_f.decision_function(tsX_f).tolist()
            log["prob_r"] += ml_r.decision_function(tsX_f).tolist()


            
            # Save
            df_log   = pd.DataFrame.from_dict(log)
            self.log = AddLabel(df_log) 
            self.log.to_csv(path_log, sep="\t")

            ToPickle(ml_f, self.modeldir+"/cid%d_forward.pickle"%cid)
            ToPickle(ml_r, self.modeldir+"/cid%d_reverse.pickle"%cid)
            
            #Feature contributions
            svector = trX_f[ml_f.support_,:]
            lambda_y = ml_f.dual_coef_

            for i, row in tqdm(enumerate(tsX_f)):
                fc = fc_original(row, svector, lambda_y.ravel(), 'tanimoto')

                if i == 0:
                    fcs = fc
                else:
                    fcs = np.vstack([fcs, fc])
            
            print("    $ Feature contributions have calculated.\n")

            if fcs_log is None:
                fcs_log = fcs
            else:
                fcs_log = np.vstack([fcs_log, fcs])

            np.save(path_log[:-4]+"_cid%d.npy" %cid, fcs)
            np.save(fcs_log_path, fcs_log)

            print("    $  Log is out.\n")

    def _CalcFCs(self, t, path_log, proportion):
        fcs_log_path = path_log[:-4] + "_all.npy"
        fcs_log = None

        for cid in self.testidx:
        # Feature contributions
            model_dir = "/".join(self.logdir.split("/")[:3])
            ml_f = LoadPickle(model_dir+"/TanimotoInterpreter_CV/Models/cid%d_forward.pickle" %cid)
            tr, ts, trX, trY, tsX, tsY, trX_f, trY_f, tsX_f, tsY_f, trX_r, trY_r = self._GetMatrices(cid)

            fcs = self._CalcFeatureImportance(self.mname, ml_f, trdata=trX_f, tsdata=tsX_f, method=self.interpreter, proportion=proportion)
            print("    $ Feature contributions have calculated.\n")

            if fcs_log is None:
                fcs_log = fcs
            else:
                fcs_log = np.vstack([fcs_log, fcs])

            np.save(path_log[:-4]+"_cid%d.npy" %cid, fcs)
            np.save(fcs_log_path, fcs_log)


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
        # p.GetScore(t="Thrombin")
        
        #TODO
        #function for fw should be independent.
        #This file should concentrate on fit/predict
        #Make another scriptto calculate fw.
        #Also funcs for calc score should be independent.
        

    
 
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 14:09:59 2020

@author: Tamura
"""

#%%
from operator import neg
import pandas as pd
import numpy as np
import sys
import os
import platform
import matplotlib.pyplot as plt
#import shap
from tqdm                              import tqdm
from SVM.svmwrappers                   import MMPKernelSVM_sklearn   as svm
from SVM.svrwrappers                   import MMPKernelSVR_sklearn   as svr
from SVM.ocsvmwrappers                 import MMPKernelOCSVM_sklearn as ocsvm
from RandomForest.RandomForestClassifier import RandomForestClassifier_CV as rf
from SVM.svmwrappers                   import MakeInputDict
from MMP.make_input                    import LeaveOneCoreOut, GetDiverseCore, DelTestCpdFromTrain, MultipleTrainTestSplit
from Tools.ReadWrite                   import ReadDataAndFingerprints, ToPickle, LoadPickle, ToJson, LoadJson
from Tools.utils                       import BasicInfo
#from Metrix.scores                     import ScoreTable,RenderDF2Fig
from Evaluation.Labelling              import AddLabel
from Evaluation.Score                  import ScoreTable_wDefinedLabel
from collections                       import OrderedDict, defaultdict
from Plots.Barplot                     import MakeBarPlotsSeaborn as bar
from Plots.Lineplot                    import MakeLinePlotsSeaborn as line
from Fingerprint.Hash2BitManager       import Hash2Bits, FindBitLength
from functools                         import partial
#from svmviz.svmviz.mmpviz              import feature_contributions
from svmviz.svmviz.svmviz              import feature_contributions as fc_original
from sklearn.svm                       import SVC
from Kernels.Kernel                    import funcTanimotoSklearn
#from MKLpy.algorithms                  import EasyMKL, AverageMKL
from Kernels.Kernel                    import funcTanimotoSklearn
from collections                       import defaultdict
from sklearnex                         import patch_sklearn
from GradientBoost.XGBoost             import XGBoost_CV as xgb

patch_sklearn()

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
        self.col  = ["core", "sub1", "sub2", "overlap"]
        
        print("$  Decision function will be applied.\n")    

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


    def _SetML(self, model_name, nu=0.05, kernel_type="product"):
    
        if model_name == "svm":

            if kernel_type == "product":
                print('    $ svm with MMPkernel is used.')
                weight  = False
            else: 
                print('    $ svm with weighted MMPkernel is used.')
                weight = kernel_type

            model = svm(len_c=self.nbits_c, decision_func=True, weight_kernel=weight, cv=True, nfold=3)

        elif model_name == "random_forest":
            print('    $ random forest is used.')
            model = rf(njobs=-1, pos_label=1, neg_label=-1, nf=3)

        elif model_name == "ocsvm":
            print('    $ ocsvm with MMPkernel is used.')
            model = ocsvm(len_c=self.nbits_c, decision_func=True, nu=nu)
            
        elif model_name == 'xgboost':
            print('    $ XGboost is used.')
            model = xgb()
            
        else:
            NotImplementedError("%s has not been implemented. Please put the obj directly." %model_name)

        return model


    def _CalcFeatureImportance(self, model_name, ml, trdata, tsdata, method, link="identity", proportion="average"):
        
        if method == "shap":

            if model_name == "svm":
                shap_kernel = shap.KernelExplainer(
                                                    model = ml.svc_.decision_function,
                                                    data  = trdata,
                                                    link  = link)

                fcs = shap_kernel.shap_values(tsdata, nsamples=self.nsample)

            elif model_name == "random_forest":
                shap_kernel = shap.TreeExplainer(
                                                    model = ml.pmodel,
                                                    data  = trdata)

                fcs = shap_kernel.shap_values(tsdata)[1]

            elif model_name == "ocsvm":
                shap_kernel = shap.KernelExplainer(
                                                    model = ml.ocsvm_.decision_function,
                                                    data  = trdata,
                                                    link  = link)   

                fcs = shap_kernel.shap_values(tsdata, nsamples=self.nsample)                                  
                    
            else:
                NotImplementedError("%s has not been implemented. Please put the obj directly." %model_name)

        elif method == "tanimoto":
            
            svector = trdata[ml.svc_.support_,:]
            lambda_y = ml.svc_.dual_coef_

            for i, row in tqdm(enumerate(tsdata)):
                fc = feature_contributions(row, svector, lambda_y.ravel(), self.nbits_cf, proportion=proportion)

                if i == 0:
                    fcs = fc
                else:
                    fcs = np.vstack([fcs, fc])
            
        return fcs

    def _SetParams(self):

        self.nbits_c = FindBitLength(self.ecfp, [self.col[0]])
        self.nbits_s = FindBitLength(self.ecfp, self.col[1:] )
        
        self.data_split_generator = MultipleTrainTestSplit(self.main, n_dataset=3)
        self.testsetidx           = self.data_split_generator.keys()

        # Leave One Core Out
        if self.debug: 
            self.nsample = 1

        else:
            self.nsample = 50


    def _ReadDataFile(self, target, acbits=False):

        if acbits:
            main = pd.read_csv("./Dataset_ACbits/Data/%s.tsv" %target, sep="\t", index_col=0) # Single class MMSs were removed in curation.py.
            ecfp = pd.read_csv("./Dataset_ACbits/ECFP/%s.tsv" %target, sep="\t", index_col=0)
        else:    
            main = pd.read_csv("./Dataset/Data/%s.tsv" %target, sep="\t", index_col=0) # Single class MMSs were removed in curation.py.
            ecfp = pd.read_csv("./Dataset/ECFP/%s.tsv" %target, sep="\t", index_col=0)

        return main, ecfp


    def _GetMatrices(self, cid, aconly=False):

        generator = self.data_split_generator[cid]
        tr        = self.main.loc[generator.tridx,:]
        ts        = self.main.loc[generator.tsidx,:]
        
        # Check overlap
        #tr = DelTestCpdFromTrain(ts, tr, deltype="both")
        
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
        
    
    def run(self, target, debug=False, kernel_type=False, onlyfccalc=False, proportion="average"):
        
        print("\n----- %s is proceeding -----\n" %target)
        path_log = os.path.join(self.logdir, "%s_%s.tsv" %(target, self.mtype))
        if debug:
            self.debug=True
        
        self.main, self.ecfp = self._ReadDataFile(target, acbits=self.aconly)
        self._SetParams()

        if onlyfccalc:
            self._CalcFCs(target, path_log, proportion)
        
        else:

            if kernel_type == "simpleTanimoto":
                print("    $ %s is conducted\n" %kernel_type)
                self._AllMMSPred_SimpleTanimoto(target, path_log)
                
            elif kernel_type == "MKL":
                print("    $ %s is conducted\n" %kernel_type)
                self._AllMMSPred_MKL(target, path_log)
                
            else:
                self._AllMMSPred(target, path_log)

    
    def GetScore(self, t):

        scorer = ScoreTable_wDefinedLabel(pred_type=self.pred_type)

        path_log = os.path.join(self.logdir, "%s_%s.tsv" %(t, self.mtype))
        log      = pd.read_csv(path_log, sep="\t", index_col=0)
        df       = scorer.GetTable({t:log})

        if self.kernel_type==False:
            path_score = os.path.join(self.scoredir, "%s_%s.tsv" %(t, self.mtype))
        else:   
            path_score = os.path.join(self.scoredir, "%s_%s_%d_%d.tsv" %(t, self.mtype, self.kernel_type[0], self.kernel_type[1]))
        df.to_csv(path_score, sep="\t")
        

class Classification(Base_ECFPECFP):

    def __init__(self, modeltype, model, dir_log, dir_score, interpreter, aconly, kernel_type="product"):

        super().__init__(modeltype, model_name=model, dir_log=dir_log, dir_score=dir_score, interpreter=interpreter, aconly=aconly, kernel_type=kernel_type)

        self.pred_type = "classification"
        
    def _AllMMSPred(self, target, path_log):
        
        fcs_log_path = path_log[:-4] + "_all.npy"
        fcs_log = None

        for cid in self.testsetidx:    
            
            # Initialize
            log = defaultdict(list) #MakeLogDict(type=self.pred_type)
        
            if self.debug:
                if cid>2:
                    break
            
            # Train test split
            print("    $ Prediction for trial%d is going on.\n" %cid)
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

            if self.mname == "svm":
                log["prob"] += ml.dec_func_.tolist()

            elif self.mname == "random_forest":
                log["prob"] += [prob[1] for prob in ml.score(tsX).tolist()]
                
            elif self.mname == 'xgboost':
                log["prob"] += [prob[1] for prob in ml.score(tsX).tolist()]
            
            else:
                raise NotImplementedError('%s is not available so far' %self.mname)
            
            # Save            
            path_log = os.path.join(self.logdir, "%s_%s_trial%d.tsv" %(target, self.mtype, cid))
            self.log = pd.DataFrame.from_dict(log)
            self.log.to_csv(path_log, sep="\t")

            params = {key: val for key, val in ml.get_params().items() if key!='kernel'}
            ToJson(params, self.modeldir+"/%s_trial%d.json"%(target, cid))
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
            ml_f = SVC(C=1.0, class_weight="balanced", kernel=funcTanimotoSklearn)
            ml_r = SVC(C=1.0, class_weight="balanced", kernel=funcTanimotoSklearn)
            
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


class ReCalc_SHAP(Base_ECFPECFP):
    def __init__(self, modeltype, model, dir_log, dir_score, interpreter, aconly, kernel_type="product"):

        super().__init__(modeltype, model_name=model, dir_log=dir_log, dir_score=dir_score, interpreter=interpreter, aconly=aconly, kernel_type=kernel_type)

        self.pred_type = "classification"

    

    def _AllMMSPred(self, t, path_log):
        
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

            ml_f = LoadPickle("./ForThrombin/Log/TanimotoInterpreter_CV/Models/cid%d_forward.pickle" %(cid))
            
            # Feature contributions
            fcs = self._CalcFeatureImportance(self.mname, ml_f, trdata=trX_f, tsdata=tsX_f, method=self.interpreter)
            print("    $ Feature contributions have calculated.\n")

            if fcs_log is None:
                fcs_log = fcs
            else:
                fcs_log = np.vstack([fcs_log, fcs])

            np.save(path_log[:-4]+"_cid%d.npy" %cid, fcs)
            np.save(fcs_log_path, fcs_log)

            print("    $  Log is out.\n")

if __name__ == "__main__":
    
    if platform.system() == 'Darwin':
        bd    = "/Users/tamura/work/ACPredCompare/"
    else:
        bd    = "/home/tamuras0/work/ACPredCompare/"
        
    model = "XGBoost"
    mtype = "wodirection"
    os.chdir(bd)
    os.makedirs("./Log_trtssplit", exist_ok=True)
    os.makedirs("./Score_trtssplit", exist_ok=True)
    
    tlist = pd.read_csv('./Dataset/target_list.tsv', sep='\t', index_col=0)
    
    for i, sr in tlist.iterrows():
        
        target = sr['target']
        
        p = Classification(modeltype   = mtype,
                        model       = model,
                        dir_log     = "./Log_trtssplit/%s" %(model+'_'+mtype),
                        dir_score   = "./Score_trtssplit/%s" %(model+'_'+mtype),
                        interpreter = "shap",
                        aconly      = False,
                        )

        p.run(target=target, debug=False)
        # p.GetScore(t="Thrombin")
        
        #TODO
        #function for fw should be independent.
        #This file should concentrate on fit/predict
        #Make another scriptto calculate fw.
        #Also funcs for calc score should be independent.
        

    
 
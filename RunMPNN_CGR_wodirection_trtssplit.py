import pandas as pd
import numpy as np
import sys
import os
import optuna
import torch
from torch                             import nn
from torch.utils.data                  import dataloader, Subset, DataLoader
#from torch_geometric.data              import DataLoader
from Tools.ReadWrite                   import ToJson
from collections                       import defaultdict
from sklearn.metrics                   import roc_auc_score
from sklearn.model_selection           import StratifiedKFold
from sklearn.metrics                   import roc_auc_score, accuracy_score
from BaseFunctions_NN                  import Base_wodirection_CGR
import random
from functools import partial
from rdkit import Chem
from argparse import Namespace



ELEM_LIST = list(range(1,119))
ATOM_FDIM, BOND_FDIM = len(ELEM_LIST) + 21, 11


def torch2numpy(x):
    return x.to("cpu").detach().numpy().copy()

def onek_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]

def atom_features(atom):
    return onek_encoding_unk(atom.GetAtomicNum() , ELEM_LIST) + onek_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5])+ onek_encoding_unk(atom.GetFormalCharge(), [-1, -2, 1, 2, 0]) + onek_encoding_unk(int(atom.GetChiralTag()), [0, 1, 2, 3])+onek_encoding_unk(int(atom.GetHybridization()),[
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ])+[1 if atom.GetIsAromatic() else 0]  

def bond_features(bond):
    bt = bond.GetBondType()
    stereo = int(bond.GetStereo())
    fbond = [bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC, bond.IsInRing()]
    fstereo = onek_encoding_unk(stereo, [0,1,2,3,4,5])
    fbond=fbond + fstereo
    return fbond

def get_atom(graph, atom_num):
    for atom in graph.GetAtoms():
        if atom.GetIdx() == atom_num:
            return atom
        else:
            continue

def index_select_ND(source: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """
    Selects the message features from source corresponding to the atom or bond indices in index.
    :param source: A tensor of shape (num_bonds, hidden_size) containing message features.
    :param index: A tensor of shape (num_atoms/num_bonds, max_num_bonds) containing the atom or bond
    indices to select from source.
    :return: A tensor of shape (num_atoms/num_bonds, max_num_bonds, hidden_size) containing the message
    features corresponding to the atoms/bonds specified in index.
    """
    # source = source.long()
    source     = source.float()
    index      = index.long()
    index_size = index.size()
    suffix_dim = source.size()[1:]
    final_size = index_size + suffix_dim
    target     = source.index_select(dim=0, index=index.view(-1))
    target     = target.view(final_size)
    
    return target


class MolGraph:
    """
    A MolGraph represents the graph structure and featurization of a single molecule.
    A MolGraph computes the following attributes:
    - smiles: Smiles string.
    - n_atoms: The number of atoms in the molecule.
    - n_bonds: The number of bonds in the molecule.
    - f_atoms: A mapping from an atom index to a list atom features.
    - f_bonds: A mapping from a bond index to a list of bond features.
    - a2b: A mapping from an atom index to a list of incoming bond indices.
    - b2a: A mapping from a bond index to the index of the atom the bond originates from.
    - b2revb: A mapping from a bond index to the index of the reverse bond.
    """

    def __init__(self, smiles: str, args=None, role=None):
        """
        Computes the graph structure and featurization of a molecule.
        :param smiles: A smiles string.
        :param args: Arguments.
        """
        self.smiles = smiles
        self.n_atoms, self.n_bonds = 0, 0
        self.f_atoms, self.f_bonds = [], []
        self.a2b, self.b2a, self.b2revb = [], [], []
        
        mol = Chem.MolFromSmiles(smiles)
        self.n_atoms = mol.GetNumAtoms()
        for i, atom in enumerate(mol.GetAtoms()):
            
            self.f_atoms.append(atom_features(atom))
            
        self.f_atoms = [self.f_atoms[i] for i in range(self.n_atoms)]
        for _ in range(self.n_atoms):
            self.a2b.append([])

        for a1 in range(self.n_atoms):
            for a2 in range(a1 + 1, self.n_atoms):
                bond = mol.GetBondBetweenAtoms(a1, a2)

                if bond is None:
                    continue

                f_bond = bond_features(bond)
                self.f_bonds.append(f_bond)
                self.f_bonds.append(f_bond)
                b1 = self.n_bonds
                b2 = b1 + 1

                self.a2b[a2].append(b1)
                self.b2a.append(a1)
                self.a2b[a1].append(b2)
                self.b2a.append(a2)
                self.b2revb.append(b2)
                self.b2revb.append(b1)
                self.n_bonds += 2
            


class BatchMolGraph:
    """
    A BatchMolGraph represents the graph structure and featurization of a batch of molecules.
    A BatchMolGraph contains the attributes of a MolGraph plus:
    - smiles_batch: A list of smiles strings.
    - n_mols: The number of molecules in the batch.
    - atom_fdim: The dimensionality of the atom features.
    - bond_fdim: The dimensionality of the bond features (technically the combined atom/bond features).
    - a_scope: A list of tuples indicating the start and end atom indices for each molecule.
    - b_scope: A list of tuples indicating the start and end bond indices for each molecule.
    - max_num_bonds: The maximum number of bonds neighboring an atom in this batch.
    - b2b: (Optional) A mapping from a bond index to incoming bond indices.
    - a2a: (Optional): A mapping from an atom index to neighboring atom indices.
    """

    def __init__(self, mol_graphs: list):
        
        #self.args = args
        self.masks = []
        self.smiles_batch = []
        self.mol_graphs = mol_graphs

        self.atom_fdim = ATOM_FDIM
        self.bond_fdim = BOND_FDIM 

        self.n_atoms = 1
        self.n_bonds = 1

        f_atoms = [[0] * self.atom_fdim]
        f_bonds = [[0] * self.bond_fdim]
        self.a_scope = []
        self.b_scope = []

        a2b = [[]]
        b2a = [0]
        b2revb = [0]
        for mol_graph in mol_graphs:
            f_atoms.extend(mol_graph.f_atoms)
            f_bonds.extend(mol_graph.f_bonds)

            for a in range(mol_graph.n_atoms):
                a2b.append([b + self.n_bonds for b in mol_graph.a2b[a]])

            for b in range(mol_graph.n_bonds):
                b2a.append(self.n_atoms + mol_graph.b2a[b])
                b2revb.append(self.n_bonds + mol_graph.b2revb[b])

            self.a_scope.append((self.n_atoms, mol_graph.n_atoms))
            self.b_scope.append((self.n_bonds, mol_graph.n_bonds))
            self.n_atoms += mol_graph.n_atoms
            self.n_bonds += mol_graph.n_bonds
            self.smiles_batch.append(mol_graph.smiles)

        self.max_num_bonds = max(len(in_bonds) for in_bonds in a2b)

        self.f_atoms = torch.FloatTensor(f_atoms)
        self.f_bonds = torch.FloatTensor(f_bonds)
        self.a2b = torch.LongTensor([a2b[a] + [0] * (self.max_num_bonds - len(a2b[a])) for a in range(self.n_atoms)])

        self.b2a = torch.LongTensor(b2a)
        self.b2revb = torch.LongTensor(b2revb)
        self.b2b = None
        self.a2a = None
        

    def get_components(self):
        """
        Returns the components of the BatchMolGraph.
        :return: A tuple containing PyTorch tensors with the atom features, bond features, and graph structure
        and two lists indicating the scope of the atoms and bonds (i.e. which molecules they belong to).
        """
        return self.f_atoms, self.f_bonds, self.a2b, self.b2a, self.b2revb, self.a_scope, self.b_scope

    def get_b2b(self) -> torch.LongTensor:
        """
        Computes (if necessary) and returns a mapping from each bond index to all the incoming bond indices.
        :return: A PyTorch tensor containing the mapping from each bond index to all the incoming bond indices.
        """

        if self.b2b is None:
            b2b = self.a2b[self.b2a]

            revmask = (b2b != self.b2revb.unsqueeze(1).repeat(1, b2b.size(1))).long()
            self.b2b = b2b * revmask

        return self.b2b

    def get_a2a(self) -> torch.LongTensor:
        """
        Computes (if necessary) and returns a mapping from each atom index to all neighboring atom indices.
        :return: A PyTorch tensor containing the mapping from each bond index to all the incodming bond indices.
        """
        get_b2a = self.b2a.detach().numpy().tolist()

        if self.a2a is None:

            a2neia=[]
            for incoming_bondIdList in self.a2b:
                neia=[]
                for incoming_bondId in incoming_bondIdList:
                    neia.append(get_b2a[incoming_bondId])
                a2neia.append(neia)
            self.a2a=torch.LongTensor(a2neia)

        return self.a2a
    
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
        #return [MolGraph(s) for s in self.smi_list[idx]], self.label_list[idx]
  
    
def mycollate_fn(batch):
    
    batch_list = list(zip(batch))
    mol_graphs = []
    labels     = []
    
    for smi, label in batch:
        mol_graphs.append(MolGraph(smi))
        labels.append(label)
    
    labels = torch.FloatTensor(labels).reshape([-1, 1])
        
    return mol_graphs, labels

class MPNEncoder(nn.Module):
    """A message passing neural network for encoding a molecule."""

    def __init__(self, args: dict, weight_seed=1):
        """Initializes the MPNEncoder.
        :param args: Arguments.
        :param atom_fdim: Atom features dimension.
        :param bond_fdim: Bond features dimension.
        """
        torch.cuda.manual_seed(weight_seed)
        torch.manual_seed(weight_seed)
        random.seed(weight_seed)
        np.random.seed(weight_seed)
        
        super(MPNEncoder,self).__init__()
        
        self.args     = args
        self.act_func = nn.ReLU()
        self.depth    = args['ConvNum']
        self.dim      = int(args['dim'])
        self.W_i      = nn.Linear(ATOM_FDIM, self.dim)
        self.W_o      = nn.Linear(self.dim*2, self.dim)
        
        w_h_input_size = self.dim + BOND_FDIM
        modulList      = [self.act_func, nn.Linear(w_h_input_size, self.dim)]
        
        for d in range(args['agg_depth']):
            modulList.extend([self.act_func, nn.Linear(self.dim, self.dim)])
       
        for i in range(args['ConvNum']):
            exec(f"self.W_h{i} = nn.Sequential(*modulList)")
            
        self_module = [nn.Linear(ATOM_FDIM, self.dim), self.act_func]
        for d in range(args['agg_depth']):
            self_module.extend([nn.Linear(self.dim, self.dim), self.act_func])
            
        self.W_ah = nn.Sequential(*self_module)


    def forward(self, mol_graph):
        """
        Encodes a batch of molecular graphs.
        :param mol_graph: A BatchMolGraph representing a batch of molecular graphs.
        :param features_batch: A list of ndarrays containing additional features.
        :return: A PyTorch tensor of shape (num_molecules, hidden_size) containing the encoding of each molecule.
        """

        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope = mol_graph.get_components()
        a2a = mol_graph.get_a2a()
        
        if self.args['cuda']:
            f_atoms, f_bonds, a2b, b2a, b2revb, a2a = f_atoms.cuda(), f_bonds.cuda(), a2b.cuda(), b2a.cuda(), b2revb.cuda(), a2a.cuda()
            self.W_i, self.W_o = self.W_i.cuda(), self.W_o.cuda()
            for i in range(self.depth-1):
                exec(f"self.W_h{i} = self.W_h{i}.cuda()")
                      
        input = self.act_func(self.W_i(f_atoms))

        self_message, message = input.clone(), input.clone()          
        self_message[0, :], message[0, :] = 0, 0
         
        for depth in range(self.depth):
            
            nei_a_message, nei_f_bonds = index_select_ND(message, a2a), index_select_ND(f_bonds, a2b)
            nei_message = torch.cat([nei_a_message, nei_f_bonds], dim=2)
            message = nei_message.sum(dim=1).float()
            
            message = eval(f"self.W_h{depth}(message)")
            message = self_message + message
            self_message = message.clone()
            message[0 , :] = 0
        
        nei_a_message = index_select_ND(message, a2a)
        a_message = nei_a_message.sum(dim=1).float()
        cc_message = self.W_ah(f_atoms)

        a_input = torch.cat([cc_message, a_message], dim=1)
        out = self.act_func(self.W_o(a_input))

        mol_vecs = []
        for i, (a_start, a_size) in enumerate(a_scope):
            cur_hiddens = out.narrow(0, a_start, a_size)
            mol_vec = cur_hiddens.sum(dim=0)
            mol_vecs.append(mol_vec)
        out = torch.stack(mol_vecs, dim=0)
        
        return out

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

    def __init__(self, modeltype, model, dir_log, dir_score, debug=False):

        super().__init__(modeltype, dir_log=dir_log, dir_score=dir_score, data_split_metric='trtssplit')

        self.pred_type = "classification"
        self.mname     = model
        self.debug     = debug
        self.nfold     = 3
        
        torch.autograd.set_detect_anomaly(True)
    
    
    def _set_fixedargs(self):
        
        args = dict(
                    train_num    = self.nepoch[0],
                    batch_size   = 10, #trial.suggest_categorical('batch_size', [64, 128, 256])
                    lr           = 0.0001, #trial.suggest_log_uniform('adam_lr', 1e-4, 1e-2)                   
                    agg_depth    = 1,
                    cuda         = torch.cuda.is_available(),
                    gamma        = 0.1,
                    )
        
        return args
    
    
    def _set_arg_dict(self, opt_args, n_data):
        
        args = self.fixed_arg.copy()
        args.update(opt_args)
        
        bsize = int(n_data * args['batch_prop'])
        args['batch_size'] = 128#bsize if bsize > 5 else 5 # minimum batch size is 5
            
        args['step_size']  = int(args['train_num']/args['step_num'])
        args['grad_node']  = int(args['dim'] / args['DNNLayerNum'])
        args['node_list']  = [int(args['dim'] - args['grad_node']*num) for num in range(args['DNNLayerNum'])] + [1]
        
        return args
    
    
    def _SetML(self):
        
        return super()._SetML()
    
    def predict(self, X):
        
        out  = self.mpn(X)
        pred = self.dnn(out)
        
        return pred
    
    def predict_proba(self, X):
        
        out = self.output_act(X)
        
        return out
        
    def train(self, args, device, dataloader, verbose=1):
        
        self.mpn.train()
        self.dnn.train()
        
        opt_list       = list(self.mpn.parameters()) + list(self.dnn.parameters())
        self.optimizer = torch.optim.Adam(opt_list, lr=args['lr'])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=args['step_size'], gamma=args['gamma'])

        if args['cuda']:
            self.mpn = self.mpn.cuda()
            self.dnn = self.dnn.cuda()
            
        size = len(dataloader.dataset)
        
        n_usedtr = 0
        for batch, (X, y) in enumerate(dataloader):
              
            # X, y = X.to(device), y.to(device)
            X = BatchMolGraph(X)
            
            score = self.predict(X)
            loss  =  self.loss_fn(score, y.to(device))
            
            # back propagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            #verbose
            n_usedtr += y.shape[0]
            if isinstance(verbose, int) & (batch % verbose == 0):
                loss, current = loss.item(), batch * y.shape[0]
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]", 'target: %s'%self.tname)
                

    def test(self, device, dataloader):
        
        self.mpn.eval()
        self.dnn.eval()
        
        size = len(dataloader.dataset)
        threshold = torch.tensor([0.5]).to(device)
        test_loss, correct = 0, 0
        pred_score_all, pred_all, proba_all, y_all = [], [], [], []
        
        with torch.no_grad():
            for X, y in dataloader:
                X = BatchMolGraph(X)
                # X, y = X.to(device), y.to(device)
                pred_score  = self.predict(X)
                proba       = self.predict_proba(pred_score)
                pred        = (proba>threshold).float()*1
                
                pred_score_all += torch2numpy(pred_score).reshape(-1).tolist()
                pred_all       += torch2numpy(pred).reshape(-1).tolist()
                proba_all      += torch2numpy(proba).reshape(-1).tolist()
                y_all          += torch2numpy(y).reshape(-1).tolist() 
                
                test_loss += self.loss_fn(pred_score, y.to(device))
                
        test_loss /= size
        acc = accuracy_score(pred_all, y_all)
        print(f"Test Error: \n Accuracy: {(100*acc):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        
        return pred_score_all, pred_all, proba_all

            
    def objective(self, trial, trX, trY):
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        optarg = dict(
                    ConvNum      = trial.suggest_int('ConvNum', 2, 4), #depth
                    dropout      = trial.suggest_discrete_uniform('dropout', 0.1, 0.3, 0.05),
                    dim          = int(trial.suggest_discrete_uniform("dim", 70, 100, 10)),#hidden_dim
                    step_num     = trial.suggest_int("step_num", 1, 3),
                    DNNLayerNum  = trial.suggest_int('DNNLayerNum', 2, 8),  
                    batch_prop   = trial.suggest_loguniform('batch_prop', 1e-3, 0.3)                  
                    )
        
        args = self._set_arg_dict(optarg, trY.shape[0])
        
        w_pos           = int(np.where(trY==-1)[0].shape[0] / np.where(trY==1)[0].shape[0])
        self.loss_fn    = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([w_pos]))
        self.output_act = nn.Sigmoid()
        
        if args['cuda']:
            self.loss_fn    = self.loss_fn.cuda()
            self.output_act = self.output_act.cuda()  
        
        # Set up dataloader
        dataset       = Dataset(smi_list=trX, label_list=trY)
        kfold         = StratifiedKFold(n_splits=self.nfold, shuffle=True, random_state=0)
        score_cv      = []
        
        for _fold, (idx_tr, idx_vl) in enumerate(kfold.split(trX, trY)):
            dataset_tr    = Subset(dataset, idx_tr)
            dataloader_tr = DataLoader(dataset_tr, args['batch_size'], shuffle=True, collate_fn=mycollate_fn)
            dataset_vl    = Subset(dataset, idx_vl)
            dataloader_vl = DataLoader(dataset_vl, args['batch_size'], shuffle=False, collate_fn=mycollate_fn)
        
            # training
            self.mpn = MPNEncoder(args)
            self.dnn = DeepNeuralNetwork(args)
            EPOCH = self.nepoch[0]
            for step in range(EPOCH):
                self.train(args, device, dataloader_tr)
                predY_score, predY, proba = self.test(device, dataloader_vl)
                score = roc_auc_score(y_true=trY[idx_vl], y_score=predY_score)
                #print(f"AUCROC: {(score):>0.5f}\n")
                
            score_cv.append(score)
            
        return np.mean(score_cv)

    def _fit_bestparams(self, args, trX, trY):
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        EPOCH = self.nepoch[1]
        
        dataloader_tr = DataLoader(Dataset(smi_list=trX, label_list=trY),
                                   batch_size = args['batch_size'],
                                   shuffle    = True, 
                                   collate_fn = mycollate_fn
                                   )
        
        self.mpn = MPNEncoder(args)
        self.dnn = DeepNeuralNetwork(args)
        for step in range(EPOCH):
            self.train(args, device, dataloader_tr)
            

    def _predict_bestparams(self, tsX, tsY):
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        dataloader_ts = DataLoader(Dataset(smi_list=tsX, label_list=tsY),
                                   self.fixed_arg['batch_size'],
                                   shuffle=False,
                                   collate_fn=mycollate_fn
                                   )
        
        pred_score, pred, proba = self.test(device, dataloader_ts)
        
        return pred_score, pred, proba
    
    
    def _AllMMSPred(self, target):
        
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
                    study = optuna.create_study()
                    obj   = partial(self.objective, trX=trX, trY=trY)
                    study.optimize(obj, n_trials=self.nepoch[2])
                    
                    best_args = self._set_arg_dict(study.best_params, n_data=trY.shape[0])
                    self._fit_bestparams(best_args, trX, trY)
                    score, predY, proba = self._predict_bestparams(tsX, tsY)
                    print("    $ Prediction Done.\n")
                    
                    # Write & save log
                    log = self._WriteLog(log, cid, tr, ts, tsY, predY, proba)           
                    self._Save(target, cid, log, best_args)
                    print("    $  Log is out.\n")
            
            
    def _WriteLog(self, log, cid, tr, ts, tsY, predY, proba):
        
        # Write log
        tsids         = ts["id"].tolist()
        log["ids"]   += tsids
        log["cid"]   += [cid] * len(tsids)
        log['#tr']   += [tr.shape[0]] * len(tsids)
        log['#ac_tr']+= [tr[tr['class']==1].shape[0]] * len(tsids)
        log["trueY"] += tsY.tolist()
        log["predY"] += predY
        log["prob"]  += proba
        
        return log
        
        
    def _Save(self, target, cid, log, args):
        path_log = os.path.join(self.logdir, "%s_trial%d.tsv" %(target, cid))
        self.log = pd.DataFrame.from_dict(log)
        self.log.to_csv(path_log, sep="\t")

        ToJson(args, self.modeldir+"/params_%s_trial%d.json" %(target, cid))
        torch.save(self.mpn.to('cpu').state_dict(), self.modeldir+"/mpn_%s_trial%d.pth" %(target, cid))
        torch.save(self.dnn.to('cpu').state_dict(), self.modeldir+"/dnn_%s_trial%d.pth" %(target, cid))
        print("    $  Log is out.\n")


if __name__ == "__main__":
    
    import platform
    if platform.system() == 'Darwin':
        bd    = "/Users/tamura/work/ACPredCompare/"
    else:
        bd    = "/home/tamuras0/work/ACPredCompare/"
        
    model = "MPNN"
    mtype = "wodirection_trtssplit"
    os.chdir(bd)
    os.makedirs("./Log_%s"%mtype, exist_ok=True)
    os.makedirs("./Score_%s"%mtype, exist_ok=True)
    
    tlist = pd.read_csv('./Dataset/target_list.tsv', sep='\t', index_col=0)
    tlist = tlist.loc[tlist['machine1'],:]
    p = Classification(modeltype  = mtype,
                       model      = model,
                       dir_log    = './Log_%s/%s' %(mtype, model),
                       dir_score  = './Score_%s/%s' %(mtype, model)
                       )
    p.run_parallel(tlist['chembl_tid'], njob=8)
    
    # for i, sr in tlist.iterrows():
        
    #     target = sr['chembl_tid']
        
    #     p.run(target=target, debug=True)
    
    # p.run(target='CHEMBL224', debug=True)

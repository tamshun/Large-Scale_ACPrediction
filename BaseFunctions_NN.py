# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 14:09:59 2020

@author: Tamura
"""

#%%

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import StratifiedShuffleSplit
from rdkit import Chem
import torch
from torch import nn
import random
from BaseClass import Initialize, MultipleAXVSplit, MultipleTrainTestSplit
    
    
def torch2numpy(x):
    return x.to("cpu").detach().numpy().copy()

def onek_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]

ELEM_LIST = list(range(1,119))
ATOM_FDIM, BOND_FDIM = len(ELEM_LIST) + 21, 11

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
class Base_wodirection_CGR(Initialize):
    
    def __init__(self, modeltype, dir_log=None, dir_score=None, data_split_metric='trtssplit'):
        
        super().__init__(modeltype=modeltype, dir_log=dir_log, dir_score=dir_score)
        
        self.trtssplit    = data_split_metric
        self.col          = 'CGR'


    def _SetParams(self, target):
        
        if self.trtssplit == 'axv':
            # Leave One Core Out
            all_seeds = pd.read_csv('./Dataset/Stats/axv.tsv', sep='\t', index_col=0).index
            seeds = [int(i.split('-Seed')[1]) for i in all_seeds if target == i.split('-Seed')[0]]
            self.data_split_generator = MultipleAXVSplit(self.main, seeds=seeds)
            self.testsetidx           = self.data_split_generator.keys()
            self.predictable          = True
        
        elif self.trtssplit == 'trtssplit':
            # Stratified Shuffled split
            self.data_split_generator = MultipleTrainTestSplit(self.main, n_dataset=3)
            self.testsetidx           = self.data_split_generator.keys()

        
    def _ReadDataFile(self, target):
  
        self.main = pd.read_csv("./Dataset/CGR/%s.tsv" %target, sep="\t", index_col=0)


    def _GetTrainTest(self, cid):
        '''
        Wrapper function of TrainTestSplit_main, TrainTestSplit_ecfp, Hash2Bits 
        '''
        if self.trtssplit == 'axv':
            tr, cpdout, bothout = self._TrainTestSplit_axv(cid)
            return tr, cpdout, bothout
        
        elif self.trtssplit == 'trtssplit':
            tr, ts = self._TrainTestSplit(cid)
            return tr, ts
        
    
    def _TrainTestSplit(self, cid):
        
        generator = self.data_split_generator[cid]
        tr        = self.main.loc[generator.tridx,:]
        ts        = self.main.loc[generator.tsidx,:]
            
        return tr, ts
    
    
    def _TrainTestSplit_axv(self, cid):
        
        generator = self.data_split_generator[cid]
        tr        = self.main.loc[generator.tridx,        :]
        cpdout    = self.main.loc[generator.compound_out, :]
        bothout   = self.main.loc[generator.both_out,     :]        
            
        return tr, cpdout, bothout
    
    
    
        
        
    

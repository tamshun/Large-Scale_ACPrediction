import pandas as pd
import numpy as np
from collections import defaultdict
from openeye.oechem import *
from chem.transform import SmilesToOEGraphMol, SmartsToOEGraphMol
from tqdm import tqdm
from chem.transform import GetCanonicalSmiles

target = 'melanocortin_receptor_4'
df = pd.read_csv('./Dataset/ECFP/%s.tsv' %target, sep='\t', index_col=0)
df_row = pd.read_csv('./Dataset/ECFP/melanocortin_receptor_4_wosubdiff.tsv', sep='\t', index_col=0)

# all_patternf = np.unique(df_row['sub1_patts_forward'].apply(lambda x: x.split(' ')).sum())
# all_smif = set([OEMolToSmiles(SmartsToOEGraphMol(smi)) for smi in all_patternf])

# all_patternr = df_row['sub1_patts_forward'].apply(lambda x: x.split(' ')).sum()
# all_smir = set([OEMolToSmiles(SmartsToOEGraphMol(smi)) for smi in all_patternr])

# print(all_smif==all_smir)

all_hashf = [int(i) for i in df_row['sub2_forward'].apply(lambda x: x.split(' ')).sum()]
all_hashr = [int(i) for i in df_row['sub2_reverse'].apply(lambda x: x.split(' ')).sum()]

all_pattsf = df_row['sub2_patts_forward'].apply(lambda x: x.split(' ')).sum()
all_pattsr = df_row['sub2_patts_reverse'].apply(lambda x: x.split(' ')).sum()

print(len(all_hashf))
print(len(all_hashr))
print(np.unique(all_hashf).shape)
print(np.unique(all_hashr).shape)

print(len(all_pattsf))
print(len(all_pattsr))
print(np.unique(all_pattsf).shape)
print(np.unique(all_pattsr).shape)

all_smif = [GetCanonicalSmiles(SmartsToOEGraphMol(smi)) for smi in tqdm(all_pattsf)]
all_smir = [GetCanonicalSmiles(SmartsToOEGraphMol(smi)) for smi in tqdm(all_pattsr)]

print(len(all_smif))
print(len(all_smir))

print(set(all_smir)==set(all_smif))

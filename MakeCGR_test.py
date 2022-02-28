import pandas as pd
from MMP.CGR import GetCGR

data = pd.read_csv('./Dataset/CGR/CHEMBL220.tsv', sep='\t', index_col=0)
cols = ['core', 'sub1', 'sub2']

for i, sr in data.iterrows():
    smi = GetCGR(sr[cols[0]], sr[cols[1]], sr[cols[2]])
    
    print('%d: %s' %(i, smi))
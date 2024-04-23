from tqdm import tqdm
import os
import pandas as pd
import pickle
import numpy as np


RAW_DATA_PATH = '/home/datasets/physionet_2012'


def read_ts(raw_data_path, set_name):
    ts = []
    pbar = tqdm(os.listdir(raw_data_path+'/set-'+set_name), 
                desc='Reading time series set '+set_name)
    for f in pbar:
        data = pd.read_csv(raw_data_path+'/set-'+set_name+'/'+f).iloc[1:]
        data = data.loc[data.Parameter.notna()]
        if len(data)<=5:
            continue
        data = data.loc[data.Value>=0] # neg Value indicates missingness.
        data['RecordID'] = f[:-4]
        ts.append(data)
    ts = pd.concat(ts)
    ts.Time = ts.Time.apply(lambda x:int(x[:2])*60
                            +int(x[3:])) # No. of minutes since admission.
    ts.rename(columns={'Time':'minute', 'Parameter':'variable', 
                       'Value':'value', 'RecordID':'ts_id'}, inplace=True)
    return ts


def read_outcomes(raw_data_path, set_name):
    oc = pd.read_csv(raw_data_path+'/Outcomes-'+set_name+'.txt', 
                     usecols=['RecordID', 'Length_of_stay', 'In-hospital_death'])
    oc['subset'] = set_name
    oc.RecordID = oc.RecordID.astype(str)
    oc.rename(columns={'RecordID':'ts_id', 'Length_of_stay':'length_of_stay', 
                       'In-hospital_death':'in_hospital_mortality'}, inplace=True)
    return oc


ts = pd.concat([read_ts(RAW_DATA_PATH, set_name) 
                for set_name in ['a','b','c']])
oc = pd.concat([read_outcomes(RAW_DATA_PATH, set_name) 
                for set_name in ['a','b','c']])
ts_ids = sorted(list(ts.ts_id.unique()))
oc = oc.loc[oc.ts_id.isin(ts_ids)]

# Drop duplicates.
ts = ts.drop_duplicates()

# Convert categorical to numeric.
ii = (ts.variable=='ICUType')
for val in [4,3,2,1]:
    kk = ii&(ts.value==val)
    ts.loc[kk, 'variable'] = 'ICUType_'+str(val)
ts.loc[ii, 'value'] = 1
    
# Generate split.
train_valid_ids = list(oc.loc[oc.subset!='a'].ts_id)
np.random.seed(123)
np.random.shuffle(train_valid_ids)
bp = int(0.8*len(train_valid_ids))
train_ids = train_valid_ids[:bp]
valid_ids = train_valid_ids[bp:]
test_ids = np.array(oc.loc[oc.subset=='a'].ts_id)
oc.drop(columns='subset', inplace=True)

# Store data.
os.makedirs('../data/processed', exist_ok=True)
pickle.dump([ts, oc, train_ids, valid_ids, test_ids], 
            open('../data/processed/physionet_2012.pkl','wb'))
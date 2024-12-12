import pandas as pd
from sklearn.preprocessing import MinMaxScaler


data = pd.read_csv('data/UNSW_NB15_training-set.csv')

# Select useful columns
selected_columns = [
    'dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'sttl', 'dttl',
    'sload', 'dload', 'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit', 'djit',
    'swin', 'dwin', 'tcprtt', 'synack', 'ackdat', 'smean', 'dmean',
    'ct_srv_src', 'ct_state_ttl', 'ct_dst_ltm', 'ct_src_dport_ltm',
    'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'is_ftp_login', 'ct_ftp_cmd',
    'ct_flw_http_mthd', 'ct_src_ltm', 'ct_srv_dst', 'is_sm_ips_ports',
    'label'
]

data = data[selected_columns]

# Handle missing values (if any)
data = data.fillna(0)

# One-hot encode binary categorical features
data['is_ftp_login'] = data['is_ftp_login'].astype(str)
data = pd.get_dummies(data, columns=['is_ftp_login'], drop_first=True)

# Normalize continuous features
scaler = MinMaxScaler()
continuous_features = data.columns.drop(['label'])
data[continuous_features] = scaler.fit_transform(data[continuous_features])

data.to_csv('data/preprocessed_training_UNSW_NB15.csv', index=False)

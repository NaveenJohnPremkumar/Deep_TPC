import os
import datetime
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from data_provider.m4 import M4Dataset, M4Meta
from sklearn.preprocessing import StandardScaler
from utils.tools import convert_tsf_to_dataframe
import warnings

warnings.filterwarnings('ignore')


# class Dataset_ETT_hour(Dataset):
#     def __init__(self, root_path, flag='train', size=None, data_path='ETTh1.csv',
#                  scale=True, seasonal_patterns=None, drop_short=False):
#         self.seq_len = size[0]
#         self.label_len = size[1]
#         self.pred_len = size[2]
#         self.token_len = self.seq_len - self.label_len
#         self.token_num = self.seq_len // self.token_len
#         self.flag = flag
#         # init
#         assert flag in ['train', 'test', 'val']
#         type_map = {'train': 0, 'val': 1, 'test': 2}
#         self.set_type = type_map[flag]

#         self.scale = scale

#         self.root_path = root_path
#         self.data_path = data_path
#         self.__read_data__()
#         self.enc_in = self.data_x.shape[-1]
#         self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

#     def __read_data__(self):
#         self.scaler = StandardScaler()
#         df_raw = pd.read_csv(os.path.join(self.root_path,
#                                           self.data_path))

#         border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
#         border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
#         border1 = border1s[self.set_type]
#         border2 = border2s[self.set_type]

#         # Store the datetime information
#         self.date_stamps = pd.to_datetime(df_raw['date']).values[border1:border2]

#         cols_data = df_raw.columns[1:]
#         df_data = df_raw[cols_data]

#         if self.scale:
#             train_data = df_data[border1s[0]:border2s[0]]
#             self.scaler.fit(train_data.values)
#             data = self.scaler.transform(df_data.values)
#         else:
#             data = df_data.values

#         data_name = self.data_path.split('.')[0]
#         # self.data_stamp = torch.load(os.path.join(self.root_path, f'{data_name}.pt'))
#         self.data_stamp = torch.load('/scratch3/home/fbellos/research/AutoTimesWithMM/ETTh1.pt')
#         self.data_stamp = self.data_stamp[border1:border2]
#         self.data_x = data[border1:border2]
#         self.data_y = data[border1:border2]

#     def __getitem__(self, index):
#         feat_id = index // self.tot_len
#         s_begin = index % self.tot_len
        
#         s_end = s_begin + self.seq_len
#         r_begin = s_end - self.label_len
#         r_end = r_begin + self.label_len + self.pred_len
#         seq_x = self.data_x[s_begin:s_end, feat_id:feat_id+1]
#         seq_y = self.data_y[r_begin:r_end, feat_id:feat_id+1]
#         seq_x_mark = self.data_stamp[s_begin:s_end:self.token_len]
#         seq_y_mark = self.data_stamp[s_end:r_end:self.token_len]

#         # Generate prompt text using the datetime information
#         start_str = str(self.date_stamps[s_begin])
#         end_str = str(self.date_stamps[s_begin + self.token_len - 1])
#         prompt_text = f"This is Time Series from {start_str} to {end_str}"

#         return seq_x, seq_y, seq_x_mark, seq_y_mark, prompt_text

#     def __len__(self):
#         return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

#     def inverse_transform(self, data):
#         return self.scaler.inverse_transform(data)

class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path='ETTh1.csv',
                 scale=True, seasonal_patterns=None, drop_short=False):
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.token_len = self.seq_len - self.label_len
        self.token_num = self.seq_len // self.token_len
        self.flag = flag
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.scale = scale

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # Store the datetime information
        self.date_stamps = pd.to_datetime(df_raw['date']).values[border1:border2]

        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        data_name = self.data_path.split('.')[0]
        self.data_stamp = torch.load(os.path.join(self.root_path, f'{data_name}.pt'))
        self.data_stamp = self.data_stamp[border1:border2]
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len
        
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id+1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id+1]
        seq_x_mark = self.data_stamp[s_begin:s_end:self.token_len]
        seq_y_mark = self.data_stamp[s_end:r_end:self.token_len]

        # Generate prompt text using the datetime information
        start_str = str(self.date_stamps[s_begin])
        end_str = str(self.date_stamps[s_begin + self.token_len - 1])
        prompt_text = f"This is Time Series from {start_str} to {end_str}"

        return seq_x, seq_y, seq_x_mark, seq_y_mark, prompt_text

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path='ETTm1.csv',
                 scale=True, seasonal_patterns=None, drop_short=False):
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.token_len = self.seq_len - self.label_len
        self.token_num = self.seq_len // self.token_len
        self.flag = flag
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.scale = scale

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # Store the datetime information for prompt text generation
        self.date_stamps = pd.to_datetime(df_raw['date']).values[border1:border2]

        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        data_name = self.data_path.split('.')[0]
        self.data_stamp = torch.load(os.path.join(self.root_path, f'{data_name}.pt'))
        self.data_stamp = self.data_stamp[border1:border2]
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len
        
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id+1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id+1]
        seq_x_mark = self.data_stamp[s_begin:s_end:self.token_len]
        seq_y_mark = self.data_stamp[s_end:r_end:self.token_len]

        # Generate prompt text using the datetime information
        start_str = str(self.date_stamps[s_begin])
        end_str = str(self.date_stamps[s_begin + self.token_len - 1])
        prompt_text = f"This is Time Series from {start_str} to {end_str}"

        return seq_x, seq_y, seq_x_mark, seq_y_mark, prompt_text

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

# class Dataset_Custom(Dataset):
#     def __init__(self, root_path, flag='train', size=None, data_path='ETTh1.csv',
#                  scale=True, seasonal_patterns=None, drop_short=False):
#         self.seq_len = size[0]
#         self.label_len = size[1]
#         self.pred_len = size[2]
#         self.token_len = self.seq_len - self.label_len
#         self.token_num = self.seq_len // self.token_len
#         self.flag = flag
#         # init
#         assert flag in ['train', 'test', 'val']
#         type_map = {'train': 0, 'val': 1, 'test': 2}
#         self.set_type = type_map[flag]

#         self.scale = scale

#         self.root_path = root_path
#         self.data_path = data_path
#         self.__read_data__()
#         self.enc_in = self.data_x.shape[-1]
#         self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

#     def __read_data__(self):
#         self.scaler = StandardScaler()
#         df_raw = pd.read_csv(os.path.join(self.root_path,
#                                           self.data_path))
#         num_train = int(len(df_raw) * 0.7)
#         num_test = int(len(df_raw) * 0.2)
#         num_vali = len(df_raw) - num_train - num_test
#         border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
#         border2s = [num_train, num_train + num_vali, len(df_raw)]
#         border1 = border1s[self.set_type]
#         border2 = border2s[self.set_type]
            

#         cols_data = df_raw.columns[1:]
#         df_data = df_raw[cols_data]

#         if self.scale:
#             train_data = df_data[border1s[0]:border2s[0]]
#             self.scaler.fit(train_data.values)
#             data = self.scaler.transform(df_data.values)
#         else:
#             data = df_data.values
#         data_name = self.data_path.split('.')[0]
#         pt_path="/scratch3/home/fbellos/research/AutoTimesWithMM/GptPreprocessed_Data"
#         self.data_stamp = torch.load(os.path.join(pt_path, f'{data_name}.pt'))
#         self.data_stamp = self.data_stamp[border1:border2]
#         self.data_x = data[border1:border2]
#         self.data_y = data[border1:border2]
        

#     def __getitem__(self, index):
#         feat_id = index // self.tot_len
#         s_begin = index % self.tot_len
        
#         s_end = s_begin + self.seq_len
#         r_begin = s_end - self.label_len
#         r_end = r_begin + self.label_len + self.pred_len
#         seq_x = self.data_x[s_begin:s_end, feat_id:feat_id+1]
#         seq_y = self.data_y[r_begin:r_end, feat_id:feat_id+1]
#         seq_x_mark = self.data_stamp[s_begin:s_end:self.token_len]
#         seq_y_mark = self.data_stamp[s_end:r_end:self.token_len]
#         return seq_x, seq_y, seq_x_mark, seq_y_mark

#     def __len__(self):
#         return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

#     def inverse_transform(self, data):
#         return self.scaler.inverse_transform(data)

class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path='ETTh1.csv',
                 scale=True, seasonal_patterns=None, drop_short=False):
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.token_len = self.seq_len - self.label_len
        self.token_num = self.seq_len // self.token_len
        self.flag = flag
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.scale = scale


        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        data_name = self.data_path.split('.')[0]
        self.data_stamp = torch.load(os.path.join(self.root_path, f'{data_name}.pt'))
        self.data_stamp = self.data_stamp[border1:border2]
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        # Assign date_stamps here, just like in Dataset_ETT_hour
        self.date_stamps = pd.to_datetime(df_raw['date']).values[border1:border2]

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len
        
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id+1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id+1]
        seq_x_mark = self.data_stamp[s_begin:s_end:self.token_len]
        seq_y_mark = self.data_stamp[s_end:r_end:self.token_len]
        
        
        # Generate prompt text using the datetime information
        start_str = str(self.date_stamps[s_begin])
        end_str = str(self.date_stamps[s_begin + self.token_len - 1])
        prompt_text = f"This is Time Series from {start_str} to {end_str}"

        return seq_x, seq_y, seq_x_mark, seq_y_mark, prompt_text

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Solar(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path='ETTh1.csv',
                 seasonal_patterns=None, scale=True, drop_short=False):
        # size [seq_len, label_len, pred_len]
        # info
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        
        self.token_len = self.seq_len - self.label_len
        self.token_num = self.seq_len // self.token_len
        self.flag = flag
        
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.scale = scale

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        
        # Read raw solar data
        df_raw = []
        with open(os.path.join(self.root_path, self.data_path), "r", encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip('\n').split(',')
                data_line = np.stack([float(i) for i in line])
                df_raw.append(data_line)
        df_raw = np.stack(df_raw, 0)
        df_raw = pd.DataFrame(df_raw)

        # Use the same border calculation as ETT datasets but adapted for Solar
        # Solar has 137 samples per day, so we use similar proportions
        total_samples = len(df_raw)
        train_samples = int(total_samples * 0.7)
        val_samples = int(total_samples * 0.1)
        test_samples = total_samples - train_samples - val_samples
        
        border1s = [0, train_samples - self.seq_len, total_samples - test_samples - self.seq_len]
        border2s = [train_samples, train_samples + val_samples, total_samples]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # Get all data columns (Solar has multiple features)
        cols_data = df_raw.columns
        df_data = df_raw[cols_data]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # Generate synthetic timestamps for solar data
        # Assuming hourly measurements starting from 2006-01-01 00:00:00
        start_date = pd.Timestamp('2006-01-01 00:00:00')
        all_timestamps = pd.date_range(start=start_date, periods=total_samples, freq='H')
        self.date_stamps = all_timestamps[border1:border2].values

        # Load preprocessed embeddings from Solar.pt file
        data_name = 'Solar'  # Use 'Solar' as the data name for loading .pt file
        self.data_stamp = torch.load(os.path.join(self.root_path, f'{data_name}.pt'))
        self.data_stamp = self.data_stamp[border1:border2]
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len
        
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id+1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id+1]
        
        seq_x_mark = self.data_stamp[s_begin:s_end:self.token_len]
        seq_y_mark = self.data_stamp[s_end:r_end:self.token_len]
        
        # Generate prompt text using the datetime information
        start_str = str(self.date_stamps[s_begin])
        end_str = str(self.date_stamps[s_begin + self.token_len - 1])
        prompt_text = f"This is Solar Time Series from {start_str} to {end_str}"

        return seq_x, seq_y, seq_x_mark, seq_y_mark, prompt_text

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_M4(Dataset):
    def __init__(self, root_path, flag='pred', size=None, data_path='ETTh1.csv',
                 scale=False, inverse=False, seasonal_patterns='Yearly', drop_short=False):
        self.scale = scale
        self.inverse = inverse
        self.root_path = root_path
        self.data_path = data_path

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.token_len = self.seq_len - self.label_len

        self.seasonal_patterns = seasonal_patterns
        self.history_size = M4Meta.history_size[seasonal_patterns]
        self.window_sampling_limit = int(self.history_size * self.pred_len)
        self.flag = flag

        self.__read_data__()

    def __read_data__(self):
        # M4Dataset.initialize()
        if self.flag == 'train':
            dataset = M4Dataset.load(training=True, dataset_file=self.root_path)
        else:
            dataset = M4Dataset.load(training=False, dataset_file=self.root_path)
        training_values = np.array(
            [v[~np.isnan(v)] for v in
             dataset.values[dataset.groups == self.seasonal_patterns]])  # split different frequencies
        self.ids = np.array([i for i in dataset.ids[dataset.groups == self.seasonal_patterns]])
        self.timeseries = [ts for ts in training_values]
        
        # Load preprocessed embeddings from M4.pt file based on flag
        if self.flag == 'train':
            data_name = f'{self.seasonal_patterns}-train'
        else:  # test or val
            data_name = f'{self.seasonal_patterns}-test'
        self.data_stamp = torch.load(os.path.join(self.root_path, f'{data_name}.pt'))
        
        # Generate synthetic timestamps for M4 data (same as Dataset_M4_Preprocess)
        self.date_stamps = []
        
        # TSF uses varied start years that jump around between decades
        # Common years seen in TSF: 1950, 1954, 1959, 1960, 1979, 1986, 1989, 1990
        tsf_years = [1950, 1954, 1959, 1960, 1979, 1986, 1989, 1990]
        
        # For test, use later years to ensure test comes after train
        if self.flag == 'train':
            base_years = tsf_years
        else:  # test or val
            base_years = [year + 20 for year in tsf_years]  # Add 20 years for test
        
        # Generate timestamps following TSF pattern
        for i in range(len(self.timeseries)):
            # Cycle through different base years like TSF does
            base_year = base_years[i % len(base_years)]
            
            # Every 5 datapoints, increment year by 10 (like TSF pattern)
            # Cap the year offset to avoid overflow
            year_offset = ((i // 5) * 10) % 100  # Keep years within reasonable range
            
            current_year = base_year + year_offset
            
            if self.seasonal_patterns == 'Yearly':
                timestamp = f"{current_year}-01-01 00:00:00"
            elif self.seasonal_patterns == 'Quarterly':
                quarter = (i % 4) + 1
                month = quarter * 3
                timestamp = f"{current_year}-{month:02d}-01 00:00:00"
            elif self.seasonal_patterns == 'Monthly':
                month = (i % 12) + 1
                timestamp = f"{current_year}-{month:02d}-01 00:00:00"
            elif self.seasonal_patterns == 'Weekly':
                week = (i % 52) + 1
                timestamp = f"{current_year}-01-{week:02d} 00:00:00"
            elif self.seasonal_patterns == 'Daily':
                day = (i % 365) + 1
                timestamp = f"{current_year}-01-{day:02d} 00:00:00"
            elif self.seasonal_patterns == 'Hourly':
                hour = i % 24
                timestamp = f"{current_year}-01-01 {hour:02d}:00:00"
            else:
                timestamp = f"{current_year}-01-01 00:00:00"
            
            self.date_stamps.append(timestamp)

    def __getitem__(self, index):
        insample = np.zeros((self.seq_len, 1))
        insample_mask = np.zeros((self.seq_len, 1))
        outsample = np.zeros((self.pred_len + self.label_len, 1))
        outsample_mask = np.zeros((self.pred_len + self.label_len, 1))  # m4 dataset

        sampled_timeseries = self.timeseries[index]
        cut_point = np.random.randint(low=max(1, len(sampled_timeseries) - self.window_sampling_limit),
                                      high=len(sampled_timeseries),
                                      size=1)[0]

        insample_window = sampled_timeseries[max(0, cut_point - self.seq_len):cut_point]
        insample[-len(insample_window):, 0] = insample_window
        insample_mask[-len(insample_window):, 0] = 1.0
        outsample_window = sampled_timeseries[
                           cut_point - self.label_len:min(len(sampled_timeseries), cut_point + self.pred_len)]
        outsample[:len(outsample_window), 0] = outsample_window
        outsample_mask[:len(outsample_window), 0] = 1.0

        # Apply mask to the data
        seq_x = insample * insample_mask
        seq_y = outsample * outsample_mask
        
        # Get embeddings for the current time series
        # Use a subset of embeddings based on the time series length
        ts_embeddings = self.data_stamp[index % len(self.data_stamp):(index % len(self.data_stamp)) + len(sampled_timeseries)]
        if len(ts_embeddings) < len(sampled_timeseries):
            # Pad with the last embedding if needed
            padding_needed = len(sampled_timeseries) - len(ts_embeddings)
            ts_embeddings = torch.cat([ts_embeddings, ts_embeddings[-1:].repeat(padding_needed, 1)])
        
        # Extract embeddings for insample and outsample windows
        insample_embeddings = ts_embeddings[max(0, cut_point - self.seq_len):cut_point]
        outsample_embeddings = ts_embeddings[cut_point - self.label_len:min(len(sampled_timeseries), cut_point + self.pred_len)]
        
        # Pad embeddings to match the expected sequence lengths
        if len(insample_embeddings) < self.seq_len:
            padding_needed = self.seq_len - len(insample_embeddings)
            insample_embeddings = torch.cat([torch.zeros(padding_needed, insample_embeddings.shape[1]), insample_embeddings])
        
        if len(outsample_embeddings) < (self.pred_len + self.label_len):
            padding_needed = (self.pred_len + self.label_len) - len(outsample_embeddings)
            outsample_embeddings = torch.cat([outsample_embeddings, torch.zeros(padding_needed, outsample_embeddings.shape[1])])
        
        # Subsample embeddings to match token_len
        seq_x_mark = insample_embeddings[::self.token_len] if len(insample_embeddings) >= self.token_len else insample_embeddings
        seq_y_mark = outsample_embeddings[::self.token_len] if len(outsample_embeddings) >= self.token_len else outsample_embeddings
        
        # Generate prompt text using the datetime information (same as Dataset_M4_Preprocess)
        start_str = self.date_stamps[index % len(self.date_stamps)]
        start = datetime.datetime.strptime(start_str, "%Y-%m-%d %H:%M:%S")
        
        # For M4 data, use appropriate frequency for time calculations
        if self.seasonal_patterns == 'Yearly':
            end = (start + datetime.timedelta(days=365*(self.token_len-1))).strftime("%Y-%m-%d %H:%M:%S")
        elif self.seasonal_patterns == 'Quarterly':
            end = (start + datetime.timedelta(days=90*(self.token_len-1))).strftime("%Y-%m-%d %H:%M:%S")
        elif self.seasonal_patterns == 'Monthly':
            end = (start + datetime.timedelta(days=30*(self.token_len-1))).strftime("%Y-%m-%d %H:%M:%S")
        elif self.seasonal_patterns == 'Weekly':
            end = (start + datetime.timedelta(weeks=self.token_len-1)).strftime("%Y-%m-%d %H:%M:%S")
        elif self.seasonal_patterns == 'Daily':
            end = (start + datetime.timedelta(days=self.token_len-1)).strftime("%Y-%m-%d %H:%M:%S")
        elif self.seasonal_patterns == 'Hourly':
            end = (start + datetime.timedelta(hours=self.token_len-1)).strftime("%Y-%m-%d %H:%M:%S")
        else:
            end = (start + datetime.timedelta(days=self.token_len-1)).strftime("%Y-%m-%d %H:%M:%S")
        
        prompt_text = f"This is Time Series from {start_str} to {end}"

        return seq_x, seq_y, seq_x_mark, seq_y_mark, prompt_text

    def __len__(self):
        return len(self.timeseries)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def last_insample_window(self):
        """
        The last window of insample size of all timeseries.
        This function does not support batching and does not reshuffle timeseries.

        :return: Last insample window of all timeseries. Shape "timeseries, insample size"
        """
        insample = np.zeros((len(self.timeseries), self.seq_len))
        insample_mask = np.zeros((len(self.timeseries), self.seq_len))
        for i, ts in enumerate(self.timeseries):
            ts_last_window = ts[-self.seq_len:]
            insample[i, -len(ts):] = ts_last_window
            insample_mask[i, -len(ts):] = 1.0
        return insample, insample_mask


class Dataset_TSF(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path=None,
                 scale=True, seasonal_patterns=None, drop_short=False):
        
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.token_len = self.pred_len
        self.context_len = 4 * self.token_len
        print(self.seq_len, self.label_len, self.pred_len)
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.root_path = root_path
        self.data_path = data_path
        self.drop_short = drop_short
        self.timeseries = self.__read_data__()


    def __read_data__(self):
        df, _, _, _, _ = convert_tsf_to_dataframe(os.path.join(self.root_path, self.data_path))
        def dropna(x):
            return x[~np.isnan(x)]
        timeseries = [dropna(ts).astype(np.float32) for ts in df.series_value]
        if self.drop_short:
            timeseries = [ts for ts in timeseries if ts.shape[0] > self.context_len]
        self.tot_len = 0
        self.len_seq = []
        self.seq_id = []
        for i in range(len(timeseries)):
            res_len = max(self.pred_len + self.seq_len - timeseries[i].shape[0], 0)
            pad_zeros = np.zeros(res_len)
            timeseries[i] = np.hstack([pad_zeros, timeseries[i]])

            _len = timeseries[i].shape[0]
            train_len = _len-self.pred_len
            border1s = [0,                          train_len - self.seq_len - self.pred_len, train_len-self.seq_len]
            border2s = [train_len - self.pred_len,  train_len,                                _len]
            
            curr_len = border2s[self.set_type] - max(border1s[self.set_type], 0) - self.pred_len - self.seq_len + 1
            curr_len = max(0, curr_len)
            
            self.len_seq.append(np.zeros(curr_len) + self.tot_len)
            self.seq_id.append(np.zeros(curr_len) + i)
            self.tot_len += curr_len
            
        self.len_seq = np.hstack(self.len_seq)
        self.seq_id = np.hstack(self.seq_id)

        return timeseries

    def __getitem__(self, index):
        len_seq = self.len_seq[index]
        seq_id = int(self.seq_id[index])
        index = index - int(len_seq)

        _len = self.timeseries[seq_id].shape[0]
        train_len = _len - self.pred_len
        border1s = [0,train_len - self.seq_len - self.pred_len, train_len-self.seq_len]

        s_begin = index + border1s[self.set_type]
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = s_end + self.pred_len

        data_x = self.timeseries[seq_id][s_begin:s_end]
        data_y = self.timeseries[seq_id][r_begin:r_end]
        data_x = np.expand_dims(data_x, axis=-1)
        data_y = np.expand_dims(data_y, axis=-1)

        return data_x, data_y, data_x, data_y

    def __len__(self):
        return self.tot_len

class Dataset_TSF_ICL(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path=None,
                 scale=True, seasonal_patterns=None, drop_short=True):
        
        self.pred_len = size[2]
        self.token_len = self.pred_len
        self.context_len = 4 * self.token_len

        self.root_path = root_path
        self.data_path = data_path
        self.timeseries = self.__read_data__()

    def __read_data__(self):
        df, _, _, _, _ = convert_tsf_to_dataframe(os.path.join(self.root_path, self.data_path))
        def dropna(x):
            return x[~np.isnan(x)]
        timeseries = [dropna(ts).astype(np.float32) for ts in df.series_value]
        timeseries = [ts for ts in timeseries if ts.shape[0] > self.context_len]
        return timeseries

    # we uniformly adopting the first time points of the time series as the corresponding prompt.
    def __getitem__(self, index):        
        data_x1 = self.timeseries[index][:2*self.token_len]
        data_x2 = self.timeseries[index][-2*self.token_len:-1*self.token_len]
        data_x = np.concatenate((data_x1, data_x2))
        data_y = self.timeseries[index][-1*self.token_len:]
        data_x = np.expand_dims(data_x, axis=-1)
        data_y = np.expand_dims(data_y, axis=-1)
        return data_x, data_y, data_x, data_y

    def __len__(self):
        return len(self.timeseries)

class Dataset_Preprocess(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 data_path='ETTh1.csv', scale=True, seasonal_patterns=None):
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.token_len = self.seq_len - self.label_len
        self.token_num = self.seq_len // self.token_len
        self.flag = flag
        self.data_set_type = data_path.split('.')[0]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.scale = scale

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
        self.tot_len = len(self.data_stamp)

    def __read_data__(self):
        if self.data_set_type == 'solar_AL':
            # For Solar data, read the raw data without date column and generate synthetic timestamps
            df_raw = []
            with open(os.path.join(self.root_path, self.data_path), "r", encoding='utf-8') as f:
                for line in f.readlines():
                    line = line.strip('\n').split(',')
                    data_line = np.stack([float(i) for i in line])
                    df_raw.append(data_line)
            df_raw = np.stack(df_raw, 0)
            
            # Generate synthetic timestamps for solar data
            # Assuming hourly measurements starting from 2006-01-01 00:00:00
            start_date = pd.Timestamp('2006-01-01 00:00:00')
            timestamps = pd.date_range(start=start_date, periods=len(df_raw), freq='H')
            self.data_stamp = [ts.strftime("%Y-%m-%d %H:%M:%S") for ts in timestamps]
        else:
            # For other datasets that have date column
            df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
            df_stamp = df_raw[['date']]
            df_stamp['date'] = pd.to_datetime(df_stamp.date).apply(str)
            self.data_stamp = df_stamp['date'].values
            self.data_stamp = [str(x) for x in self.data_stamp]
        

    def __getitem__(self, index):
        s_begin = index % self.tot_len
        s_end = s_begin + self.token_len
        start = datetime.datetime.strptime(self.data_stamp[s_begin], "%Y-%m-%d %H:%M:%S")
        if self.data_set_type in ['traffic', 'electricity', 'ETTh1', 'ETTh2']:
            end = (start + datetime.timedelta(hours=self.token_len-1)).strftime("%Y-%m-%d %H:%M:%S")
        elif self.data_set_type == 'weather':
            end = (start + datetime.timedelta(minutes=10*(self.token_len-1))).strftime("%Y-%m-%d %H:%M:%S")
        elif self.data_set_type in ['ETTm1', 'ETTm2']:
            end = (start + datetime.timedelta(minutes=15*(self.token_len-1))).strftime("%Y-%m-%d %H:%M:%S")
        elif self.data_set_type == 'solar_AL':
            end = (start + datetime.timedelta(hours=self.token_len-1)).strftime("%Y-%m-%d %H:%M:%S")
        seq_x_mark = f"This is Time Series from {self.data_stamp[s_begin]} to {end}"
        return seq_x_mark

    def __len__(self):
        return len(self.data_stamp)

class Dataset_M4_Preprocess(Dataset):
    def __init__(self, root_path, flag='train', size=None, seasonal_patterns='Daily', data_path=None):
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.token_len = self.seq_len - self.label_len
        self.token_num = self.seq_len // self.token_len
        self.seasonal_patterns = seasonal_patterns
        self.flag = flag
        self.data_path = data_path
        self.root_path = root_path
        
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        # Load M4 dataset metadata and generate timestamps
        self.__read_data__()
        self.tot_len = len(self.data_stamp)

    def __read_data__(self):
        """Generate synthetic timestamps for M4 data based on seasonal pattern"""
        # Read the actual M4 CSV file to get the number of time series
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        num_series = len(df_raw)
        max_len = num_series  # Generate timestamps for all time series in the dataset
        
        # Generate timestamps following TSF pattern: varied start years that jump around
        # Also ensure test dates come after train dates
        self.data_stamp = []
        
        # TSF uses varied start years that jump around between decades
        # Common years seen in TSF: 1950, 1954, 1959, 1960, 1979, 1986, 1989, 1990
        tsf_years = [1950, 1954, 1959, 1960, 1979, 1986, 1989, 1990]
        
        # For test, use later years to ensure test comes after train
        if self.flag == 'train':
            base_years = tsf_years
        else:  # test or val
            base_years = [year + 20 for year in tsf_years]  # Add 20 years for test
        
        # Generate timestamps following TSF pattern
        for i in range(max_len):
            # Cycle through different base years like TSF does
            base_year = base_years[i % len(base_years)]
            
            # Every 5 datapoints, increment year by 10 (like TSF pattern)
            # Cap the year offset to avoid overflow
            year_offset = ((i // 5) * 10) % 100  # Keep years within reasonable range
            
            current_year = base_year + year_offset
            
            if self.seasonal_patterns == 'Yearly':
                timestamp = f"{current_year}-01-01 00:00:00"
            elif self.seasonal_patterns == 'Quarterly':
                quarter = (i % 4) + 1
                month = quarter * 3
                timestamp = f"{current_year}-{month:02d}-01 00:00:00"
            elif self.seasonal_patterns == 'Monthly':
                month = (i % 12) + 1
                timestamp = f"{current_year}-{month:02d}-01 00:00:00"
            elif self.seasonal_patterns == 'Weekly':
                week = (i % 52) + 1
                timestamp = f"{current_year}-01-{week:02d} 00:00:00"
            elif self.seasonal_patterns == 'Daily':
                day = (i % 365) + 1
                timestamp = f"{current_year}-01-{day:02d} 00:00:00"
            elif self.seasonal_patterns == 'Hourly':
                hour = i % 24
                timestamp = f"{current_year}-01-01 {hour:02d}:00:00"
            else:
                timestamp = f"{current_year}-01-01 00:00:00"
            
            self.data_stamp.append(timestamp)

    def __getitem__(self, index):
        s_begin = index % self.tot_len
        s_end = s_begin + self.token_len
        start = datetime.datetime.strptime(self.data_stamp[s_begin], "%Y-%m-%d %H:%M:%S")
        
        # For M4 data, use appropriate frequency for time calculations
        if self.seasonal_patterns == 'Yearly':
            end = (start + datetime.timedelta(days=365*(self.token_len-1))).strftime("%Y-%m-%d %H:%M:%S")
        elif self.seasonal_patterns == 'Quarterly':
            end = (start + datetime.timedelta(days=90*(self.token_len-1))).strftime("%Y-%m-%d %H:%M:%S")
        elif self.seasonal_patterns == 'Monthly':
            end = (start + datetime.timedelta(days=30*(self.token_len-1))).strftime("%Y-%m-%d %H:%M:%S")
        elif self.seasonal_patterns == 'Weekly':
            end = (start + datetime.timedelta(weeks=self.token_len-1)).strftime("%Y-%m-%d %H:%M:%S")
        elif self.seasonal_patterns == 'Daily':
            end = (start + datetime.timedelta(days=self.token_len-1)).strftime("%Y-%m-%d %H:%M:%S")
        elif self.seasonal_patterns == 'Hourly':
            end = (start + datetime.timedelta(hours=self.token_len-1)).strftime("%Y-%m-%d %H:%M:%S")
        else:
            end = (start + datetime.timedelta(days=self.token_len-1)).strftime("%Y-%m-%d %H:%M:%S")
        
        # Create a consistent-length prompt to avoid tokenizer size mismatches
        # Use a standardized format that produces consistent token counts
        start_date_str = self.data_stamp[s_begin][:10]  # Just YYYY-MM-DD part
        end_date_str = end[:10]  # Just YYYY-MM-DD part
        
        # Use the same format as Dataset_Preprocess for consistency
        seq_x_mark = f"This is Time Series from {self.data_stamp[s_begin]} to {end}"
        return seq_x_mark

    def __len__(self):
        return len(self.data_stamp)

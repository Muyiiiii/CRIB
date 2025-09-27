import torch
import numpy as np
import os
import random
import pandas as pd
from torch.utils.data import Dataset, DataLoader

from .dataClass import (
    MyStandardScaler,
    MyDataset,
)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def criterion_mape(y_true, y_pred, eps=1e-8):
    """
    Args:
        y_true (torch.Tensor): Ground truth values, shape (batch_size, ...)
        y_pred (torch.Tensor): Predicted values, shape (batch_size, ...)
        eps (float): Small constant to avoid division by zero
    Returns:
        torch.Tensor: MAPE (scalar)
    """
    # Ensure inputs are torch tensors
    y_true = torch.as_tensor(y_true)
    y_pred = torch.as_tensor(y_pred)
    
    # Calculate absolute percentage error
    ape = torch.abs((y_true - y_pred) / (y_true + eps))
    
    # Take mean
    mape = torch.mean(ape)
    
    return mape

def load_missing_raw_data(args, dataset):

    if dataset=='Metr':
        path = os.path.join('./data/metr_la/', 'metr_la.h5')
        data = pd.read_hdf(path)
        data = np.array(data)
        # data = data[:, :, None]
        data = data[:, :]
        if args.missing_pattern=='point':
            missing_mask=get_missing_mask(data, args.missing_rate)
        elif args.missing_pattern=='col':
            missing_mask=get_col_dropout_mask(data, args.missing_rate)
        elif args.missing_pattern=='block':
            missing_mask=get_block_missing_mask_fixed_size(data, args.missing_rate, block_size=(args.missing_block_width, args.missing_block_height))


    elif dataset=='PEMS':
        path = os.path.join('./data/pems_bay/', 'pems_bay.h5')
        data = pd.read_hdf(path)
        data = np.array(data)
        # data = data[:, :, None]
        data = data[:, :]
        if args.missing_pattern=='point':
            missing_mask = get_missing_mask(data, args.missing_rate)
        elif args.missing_pattern=='col':
            missing_mask=get_col_dropout_mask(data, args.missing_rate)
        elif args.missing_pattern=='block':
            missing_mask=get_block_missing_mask_fixed_size(data, args.missing_rate, block_size=(args.missing_block_width, args.missing_block_height))


    elif dataset=='ETTh1':
        df_raw = pd.read_csv('./data/ETT/ETTh1.csv')
        data=np.array(df_raw)
        data=data[::,1:]
        if args.missing_pattern=='point':
            missing_mask = get_missing_mask(data, args.missing_rate)
        elif args.missing_pattern=='col':
            missing_mask=get_col_dropout_mask(data, args.missing_rate)
        elif args.missing_pattern=='block':
            missing_mask=get_block_missing_mask_fixed_size(data, args.missing_rate, block_size=(args.missing_block_width, args.missing_block_height))
        # data = data[:, :, None].astype('float32')
        data = data[:, :].astype('float32')
        # missing_mask = missing_mask[:, :, None].astype('int32')
        missing_mask = missing_mask[:, :].astype('int32')

    elif dataset == 'Elec':
        data_list = []
        with open('./data/Electricity/electricity.txt', 'r') as f:
            reader = f.readlines()
            for row in reader:
                data_list.append(row.split(','))

        data = np.array(data_list).astype('float')
        if args.missing_pattern=='point':
            missing_mask = get_missing_mask(data, args.missing_rate)
        elif args.missing_pattern=='col':
            missing_mask=get_col_dropout_mask(data, args.missing_rate)
        elif args.missing_pattern=='block':
            missing_mask=get_block_missing_mask_fixed_size(data, args.missing_rate, block_size=(args.missing_block_width, args.missing_block_height))
        # data = data[:, :, None].astype('float32')
        data = data[:, :].astype('float32')
        # missing_mask = missing_mask[:, :, None].astype('int32')
        missing_mask = missing_mask[:, :].astype('int32')

    elif dataset=='BeijingAir_old':
        data = pd.DataFrame(pd.read_hdf('./data/air_quality/small36.h5', 'pm25'))
        data=np.array(data)
        eval_mask=~np.isnan(data)
        if args.missing_pattern=='point':
            missing_mask = get_missing_mask(data, args.missing_rate)
        elif args.missing_pattern=='col':
            missing_mask=get_col_dropout_mask(data, args.missing_rate)
        elif args.missing_pattern=='block':
            missing_mask=get_block_missing_mask_fixed_size(data, args.missing_rate, block_size=(args.missing_block_width, args.missing_block_height))
        data[np.isnan(data)]=0.0
        # data = data[:, :, None].astype('float32')
        data = data[:, :].astype('float32')
        # missing_mask = missing_mask[:, :, None].astype('int32')
        missing_mask = missing_mask[:, :].astype('int32')

    elif dataset=='PEMS08':
        data=np.load('./data/PEMS08/PEMS08.npz')["data"][..., 0]
        data = data[:, :]
        if args.missing_pattern=='point':
            missing_mask = get_missing_mask(data, args.missing_rate)
        elif args.missing_pattern=='col':
            missing_mask=get_col_dropout_mask(data, args.missing_rate)
        elif args.missing_pattern=='block':
            missing_mask=get_block_missing_mask_fixed_size(data, args.missing_rate, block_size=(args.missing_block_width, args.missing_block_height))

    elif dataset=='BeijingAir':
        data=pd.read_excel('./data/BeijingAirQuality/BeijingAirQuality.xlsx')
        data=data.to_numpy()[:,:]
        if args.missing_pattern=='point':
            missing_mask = get_missing_mask(data, args.missing_rate)
        elif args.missing_pattern=='col':
            missing_mask=get_col_dropout_mask(data, args.missing_rate)
        elif args.missing_pattern=='block':
            missing_mask=get_block_missing_mask_fixed_size(data, args.missing_rate, block_size=(args.missing_block_width, args.missing_block_height))
    elif 'imputed' in dataset:
        data=np.load(f"./data/imputed_TimesNet/imputation_0.4_TimesNet_My{dataset}_impu{args.missing_rate}/imputed.npy")
        if args.missing_pattern=='point':
            missing_mask = get_missing_mask(data, args.missing_rate)
        elif args.missing_pattern=='col':
            missing_mask=get_col_dropout_mask(data, args.missing_rate)
        elif args.missing_pattern=='block':
            missing_mask=get_block_missing_mask_fixed_size(data, args.missing_rate, block_size=(args.missing_block_width, args.missing_block_height))
    else:
        print(f'{dataset} is not a valid dataset.')


    # data_torch = torch.from_numpy(data).to(torch.float16)
    data_torch = torch.from_numpy(data).to(torch.float32) # [time_len, var_num]
    missing_mask_torch = torch.from_numpy(missing_mask).to(torch.float32) # [time_len, var_num]

    return data_torch, missing_mask_torch


def get_missing_mask(array,rate=0.2):
    '''
    zero means missing, one means observed
    '''
    zeros_num = int(array.size * rate)
    new_array = np.ones(array.size)
    new_array[:zeros_num] = 0
    np.random.shuffle(new_array)
    re_array = new_array.reshape(array.shape)
    return re_array

def get_col_dropout_mask(array, rate=0.2):
    """
    generate mask for feature/sensor dropout.
    
    Parameters:
    array (np.ndarray): input numpy array, must be 2-dimensional.
    rate (float): the ratio of missing features/sensors.
    
    Returns:
    np.ndarray: mask with the same shape as the input array, 0 means missing, 1 means observed.
    """
    if array.ndim != 2:
        raise ValueError("Input array must be 2-dimensional for feature dropout.")
    if rate == 0:
        return np.ones_like(array)
    
    mask = np.ones_like(array)
    num_cols = array.shape[1]
    
    # calculate the number of missing columns
    num_missing_cols = int(num_cols * rate)
    if num_missing_cols == 0 and rate > 0:
        num_missing_cols = 1 # ensure at least one column is missing
    
    # randomly select the indices of the columns to be missing
    missing_col_indices = np.random.choice(num_cols, size=num_missing_cols, replace=False)
    
    # set all the columns to 0 (missing)
    mask[:, missing_col_indices] = 0
    
    return mask

def get_block_missing_mask(array, rate=0.2, num_blocks=1):
    """
    generate mask for block missing.
    
    Parameters:
    array (np.ndarray): input numpy array, used to get its shape.
    rate (float): missing rate, between 0 and 1.
    num_blocks (int): the number of missing blocks to be generated.
    
    Returns:
    np.ndarray: mask with the same shape as the input array, 0 means missing, 1 means observed.
    """
    if rate == 0:
        return np.ones_like(array)
    if rate == 1:
        return np.zeros_like(array)

    mask = np.ones_like(array)
    total_missing_elements = int(array.size * rate)
    missing_per_block = total_missing_elements // num_blocks
    
    # calculate the approximate side length of each small block
    block_side = int(np.sqrt(missing_per_block))
    if block_side == 0:
        block_side = 1
        
    block_height = block_side
    block_width = block_side

    for _ in range(num_blocks):
        # randomly select the top-left corner of the block
        # ensure that the block does not exceed the array boundary
        start_row = np.random.randint(0, array.shape[0] - block_height + 1)
        start_col = np.random.randint(0, array.shape[1] - block_width + 1)
        
        # set the selected block area to 0 (missing)
        mask[start_row : start_row + block_height, 
             start_col : start_col + block_width] = 0
             
    return mask

def get_block_missing_mask_fixed_size(array, rate=0.2, block_size=(10, 10)):
    """
    generate mask for block missing with fixed size.
    
    Parameters:
    array (np.ndarray): input numpy array, used to get its shape.
    rate (float): missing rate, between 0 and 1.
    block_size (tuple): a tuple containing (height, width), defining the size of each missing block.
    
    Returns:
    np.ndarray: mask with the same shape as the input array, 0 means missing, 1 means observed.
    """
    if rate == 0:
        return np.ones_like(array)
    if rate == 1:
        return np.zeros_like(array)

    mask = np.ones_like(array)
    
    # get the height and width of the block from block_size
    block_height, block_width = block_size
    
    # calculate the total number of elements to be missing
    total_missing_elements = int(array.size * rate)
    
    # calculate the number of elements in a block
    elements_per_block = block_height * block_width
    
    # avoid division by zero error
    if elements_per_block == 0:
        return mask
        
    # calculate the number of blocks needed based on the total number of missing elements and the number of elements in each block
    num_blocks = total_missing_elements // elements_per_block
    
    # if the missing rate is greater than 0, but the number of blocks is 0, then at least generate one block
    if num_blocks == 0 and rate > 0:
        num_blocks = 1
        
    for _ in range(num_blocks):
        # randomly select the top-left corner of the block
        # ensure that the block does not exceed the array boundary
        # +1 is to ensure that the upper bound of randint includes the last possible starting point
        if array.shape[0] - block_height < 0 or array.shape[1] - block_width < 0:
            print("Warning: Block size is larger than array size. Skipping.")
            continue
            
        start_row = np.random.randint(0, array.shape[0] - block_height + 1)
        start_col = np.random.randint(0, array.shape[1] - block_width + 1)
        
        # set the selected block area to 0 (missing)
        mask[start_row : start_row + block_height, 
             start_col : start_col + block_width] = 0
             
    return mask



def Add_Window_Horizon(data, mask_1, window_size, horizon_size):
    '''
    :param data: shape [time_len, var_num] [time_length, variable_number]
    :param window_size:
    :param horizon_size: [prediction_length]
    :return: X is [win_num, win_len, var_num], Y is [win_num, hor_len, var_num]
    '''
    length = len(data)
    end_index = length - horizon_size - window_size + 1
    X = []      #windows
    Y = []      #horizons
    masks_1 = []

    for idx in range(end_index):
        # X.append(data[idx : idx + window_size + horizon_size])
        X.append(data[idx : idx + window_size])

        Y.append(data[idx + window_size : idx + window_size + horizon_size])

        # tmp_mask_1 = torch.concat([mask_1[idx : idx + window_size], horizon_mask])
        # masks_1.append(tmp_mask_1)
        masks_1.append(mask_1[idx : idx + window_size + horizon_size])
        # masks_1.append(mask_1[idx : idx + window_size])

    X = torch.stack(X, dim=0)  # [win_num, seq_len, var_num]
    Y = torch.stack(Y, dim=0)  # [win_num, pred_len, var_num]
    masks_1 = torch.stack(masks_1, dim=0) # [win_num, seq_len+pred_len, var_num]

    return X, Y, masks_1


def patching(args, data, mask_1):
    W, _, N=data.shape # [win_num, seq_len+pred_len / seq_len, var_num]

    data_patch = data.permute(0,2,1).reshape(W, N, -1, args.patch_len).permute(0,2,1,3) # [win_num, patch_num, var_num, patch_len]
    mask_patch_1 = mask_1.permute(0,2,1).reshape(W, N, -1, args.patch_len).permute(0,2,1,3) # [win_num, patch_num, var_num, patch_len]

    return data_patch, mask_patch_1 # [win_num, patch_num, var_num, patch_len]


def split_data_by_ratio(x, y, mask_1, val_ratio, test_ratio, std_flag=True, scaler=None):
    idx = np.arange(x.shape[0])
    # print('idx shape:',idx.shape)
    idx_shuffle = idx.copy()
    # np.random.shuffle(idx_shuffle)
    data_len = x.shape[0]

    test_x = x[idx_shuffle[-int(data_len * test_ratio):]] # [test_num, patch_num, var_num, patch_len]
    test_y = y[idx_shuffle[-int(data_len * test_ratio):]] # [test_num, pred_len, var_num]
    test_mask_1 = mask_1[idx_shuffle[-int(data_len * test_ratio):]] # [test_num, patch_num, var_num, patch_len]

    val_x = x[idx_shuffle[-int(data_len * (test_ratio + val_ratio)):-int(data_len * test_ratio)]] # [val_num, patch_num, var_num, patch_len]
    val_y = y[idx_shuffle[-int(data_len * (test_ratio + val_ratio)):-int(data_len * test_ratio)]] # [val_num, pred_len, var_num]
    val_mask_1 = mask_1[idx_shuffle[-int(data_len * (test_ratio + val_ratio)):-int(data_len * test_ratio)]] # [val_num, patch_num, var_num, patch_len]

    train_x = x[idx_shuffle[:-int(data_len * (test_ratio + val_ratio))]] # [train_num, patch_num, var_num, patch_len]
    train_y = y[idx_shuffle[:-int(data_len * (test_ratio + val_ratio))]] # [train_num, pred_len, var_num]
    train_mask_1 = mask_1[idx_shuffle[:-int(data_len * (test_ratio + val_ratio))]] # [train_num, patch_num, var_num, patch_len]

    if std_flag:
        if scaler is None:
            scaler = MyStandardScaler(mean=(train_x * train_mask_1[:,:train_x.shape[1],...]).mean(), std=(train_x * train_mask_1[:,:train_x.shape[1],...]).std())

        train_x = scaler.transform(train_x)
        train_y = scaler.transform(train_y)
        val_x = scaler.transform(val_x)
        val_y = scaler.transform(val_y)
        test_x = scaler.transform(test_x)
        test_y = scaler.transform(test_y)
    else:
        scaler=None

    return train_x, train_y, train_mask_1, val_x, val_y, val_mask_1, test_x, test_y, test_mask_1, scaler

def load_dataset(args, scaler=None):
    data_raw, missing_mask_raw = load_missing_raw_data(args=args, dataset=args.dataset) # [time_len, var_num]

    print(f'After Initial:\n {data_raw.shape}, {missing_mask_raw.shape}')

    data_raw_win, pred_raw_win, missing_mask_raw_win = Add_Window_Horizon(data=data_raw, mask_1=missing_mask_raw, window_size=args.seq_len, horizon_size=args.pred_len) # [win_num, seq_len+pred_len / seq_len, var_num]

    print(f'After Windowing:\n {data_raw_win.shape}, {pred_raw_win.shape}, {missing_mask_raw_win.shape}')

    data_raw_patch_win, missing_mask_raw_1_patch_win = patching(args=args, data=data_raw_win, mask_1=missing_mask_raw_win) # [win_num, patch_num, var_num, patch_len]

    print(f'After Patching:\n {data_raw_patch_win.shape}, {missing_mask_raw_1_patch_win.shape}')

    train_x, train_y, train_mask_1, val_x, val_y, val_mask_1, test_x, test_y, test_mask_1, scaler = split_data_by_ratio(x=data_raw_patch_win, y=pred_raw_win, mask_1=missing_mask_raw_1_patch_win, val_ratio=args.val_ratio, test_ratio=args.test_ratio, std_flag=args.std_flag, scaler=scaler) # [train/val/test_num, patch_num, var_num, patch_len] [train/val/test_num, pred_len, var_num]

    print(f'After spliting:\n {train_x.shape}, {train_y.shape}, {train_mask_1.shape}, {val_x.shape}, {val_y.shape}, {val_mask_1.shape}, {test_x.shape}, {test_y.shape}, {test_mask_1.shape}')

    train_dataset=MyDataset(args=args, data=train_x, pred=train_y, mask_1=train_mask_1)
    val_dataset=MyDataset(args=args, data=val_x, pred=val_y, mask_1=val_mask_1)
    test_dataset=MyDataset(args=args, data=test_x, pred=test_y, mask_1=test_mask_1)

    return train_dataset, val_dataset, test_dataset, scaler
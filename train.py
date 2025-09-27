import torch
import torch.nn as nn
import pandas as pd
import torch.optim as optim
import numpy as np
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
from contextlib import nullcontext
import argparse


import muyi.utils as mu
from utils import load_dataset, criterion_mape, set_seed
from TSL_models import CRIB
from TSL_models import DLinear, SegRNN, Transformer, iTransformer, PatchTST, TSMixer,WPMixer, PAttn, KANAD, MultiPatchFormer, FreTS # want input [Batch, seq_len, Channels]

def parse_args():
    parser = argparse.ArgumentParser(description='Hyperparameter settings for MyVar.')

    parser.add_argument('--model', type=str, default='CRIB', choices=["CRIB", "STJTransformer", "DLinear", "PAttn", "SegRNN", "Transformer", "iTransformer", "PatchTST", "TSMixer", "WPMixer", "PAttn"], help='Model name')

    parser.add_argument('--dataset', type=str, default='ETTh1', choices=['Elec', 'ETTh1', 'PEMS', 'PEMS08', 'Metr', 'BeijingAir', 'Elec_imputed', 'ETTh1_imputed', 'PEMS_imputed', 'Metr_imputed'], help='Dataset name')
    
    parser.add_argument('--missing_rate', type=float, default=0.7, help='Missing data rate')
    parser.add_argument('--missing_pattern', type=str, default='col', choices=['point', 'block', 'col'], help='Missing data pattern')
    
    parser.add_argument('--loss_type', type=str, default='123', choices=['1', '2', '3', '12', '13', '23', '123'], help='Loss Type')
    parser.add_argument('--IB_weight', type=float, default=1.0, help='IB weight')
    parser.add_argument('--KL_weight', type=float, default=1e-6, help='KL weight')
    parser.add_argument('--Consis_weight', type=float, default=1.0, help='Consistency weight')
    
    parser.add_argument('--missing_block_width', type=int, default=5, help='Missing block width for block pattern')
    parser.add_argument('--missing_block_height', type=int, default=5, help='Missing block height for block pattern')
    
    parser.add_argument('--train_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay')
    parser.add_argument('--use_amp', type=bool, default=True, help='Use automatic mixed precision (AMP)')
    parser.add_argument('--mask_rate', type=float, default=0.5, help='Mask rate')
    parser.add_argument('--std_flag', type=bool, default=True, help='Standardization flag')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--shuffle', type=bool, default=True, help='Shuffle the dataset')
    parser.add_argument('--num_workers', type=int, default=80, help='Number of workers for data loading')
    parser.add_argument('--seq_len', type=int, default=24, help='Input sequence length')
    parser.add_argument('--pred_len', type=int, default=24, help='Prediction sequence length')
    parser.add_argument('--patch_len', type=int, default=8, help='Patch length')
    parser.add_argument('--model_dim', type=int, default=32, help='Model dimension')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--output_attention', type=bool, default=True, help='Output attention flag')
    parser.add_argument('--activation', type=str, default="relu", help='Activation function')
    parser.add_argument('--heads_num', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--enc_num', type=int, default=3, help='Number of encoder layers')
    parser.add_argument('--dec_num', type=int, default=1, help='Number of decoder layers')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Validation set ratio')
    parser.add_argument('--test_ratio', type=float, default=0.2, help='Test set ratio')
    parser.add_argument('--seed', type=int, default=123, help='Random seed')
    parser.add_argument('--iter', type=str, default='1', help='Iteration')
    parser.add_argument('--csv_path', type=str, default='./result_1.csv', help='Result path')
    parser.add_argument('--exp_type', type=str, default='Train', help='Exp result')
    
    # arguments for TSL models
    parser.add_argument('--task_name', type=str, default='long_term_forecast', help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--moving_avg', type=int, default=3, help='window size of moving average')
    parser.add_argument('--seg_len', type=int, default=24, help='the length of segmen-wise iteration of SegRNN')
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--d_ff', type=int, default=128, help='dimension of fcn')
    parser.add_argument('--label_len', type=int, default=0, help='start token length')
    parser.add_argument('--features', type=str, default='M', help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
    parser.add_argument('--channel_independence', type=int, default=0,help='0: channel dependence 1: channel independence for FreTS model')

    
    return parser.parse_args()

args=parse_args()

if args.seq_len % args.patch_len != 0:
    raise ValueError(f"seq_len {args.seq_len} must be divisible by patch_len {args.patch_len}")

args.patch_num = args.seq_len // args.patch_len

args.n_heads = args.heads_num
args.d_model = args.model_dim
args.e_layers = args.enc_num
args.d_layers = args.dec_num

criterion_mae=nn.L1Loss()
criterion_mse=nn.MSELoss()

if args.seed!=-1:
    set_seed(args.seed)


args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if 'Metr' in args.dataset:
    node_number=207
    args.var_num=207
    args.enc_in=207
    args.dec_in=207
    args.c_out=207
elif 'PEMS' in args.dataset:
    node_number=325
    args.var_num=325
    args.enc_in = 325
    args.dec_in = 325
    args.c_out = 325
elif 'ETTh1' in args.dataset:
    node_number=7
    args.var_num=7
    args.enc_in = 7
    args.dec_in = 7
    args.c_out = 7
elif 'Elec' in args.dataset:
    node_number=321
    args.var_num=321
    args.enc_in = 321
    args.dec_in = 321
    args.c_out = 321
elif 'BeijingAir_old' in args.dataset:
    node_number=36
    args.var_num=36
    args.enc_in = 36
    args.dec_in = 36
    args.c_out = 36
elif 'PEMS08' in args.dataset:
    node_number=170
    args.var_num=170
    args.enc_in = 170
    args.dec_in = 170
    args.c_out = 170
elif 'BeijingAir' in args.dataset:
    node_number=7
    args.var_num=7
    args.enc_in = 7
    args.dec_in = 7
    args.c_out = 7
    



train_dataset, val_dataset, test_dataset, data_scaler=load_dataset(args=args, scaler=None)

if 'imputed' in args.dataset:
    print(f"\n\nLoading original raw data for {args.dataset.replace('_imputed','')} ...\n")
    temp_args=parse_args()
    temp_args.dataset=args.dataset.replace('_imputed','')

    train_dataset_ori, val_dataset_ori, test_dataset_ori, _ = load_dataset(args=temp_args, scaler=None)

    train_dataset.data=train_dataset.data*train_dataset.mask_1[:,:3,...]+train_dataset_ori.data*(1-train_dataset.mask_1[:,:3,...])
    val_dataset.data=val_dataset.data*val_dataset.mask_1[:,:3,...]+val_dataset_ori.data*(1-val_dataset.mask_1[:,:3,...])
    test_dataset.data=test_dataset.data*test_dataset.mask_1[:,:3,...]+test_dataset_ori.data*(1-test_dataset.mask_1[:,:3,...])

    train_dataset.pred=train_dataset_ori.pred
    val_dataset.pred=val_dataset_ori.pred
    test_dataset.pred=test_dataset_ori.pred

train_loader=DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers, drop_last=False)
val_loader=DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers, drop_last=False)
test_loader=DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers, drop_last=False)


if args.model=="CRIB":
    model=CRIB(args).to(args.device)
elif args.model=="DLinear":
    model = DLinear(args).to(args.device)
elif args.model=="SegRNN":
    model = SegRNN(args).to(args.device)
elif args.model=="Transformer":
    model = Transformer(args).to(args.device)
elif args.model=="iTransformer":
    model = iTransformer(args).to(args.device)
elif args.model=="PatchTST":
    model = PatchTST(args, patch_len=args.patch_len).to(args.device)
elif args.model=="TSMixer":
    model = TSMixer(args).to(args.device)
elif args.model=="WPMixer":
    model = WPMixer(args).to(args.device)
elif args.model=="PAttn":
    model = PAttn(args).to(args.device)
elif args.model=="KANAD":
    model = KANAD(args).to(args.device)
elif args.model=="MultiPatchFormer":
    model = MultiPatchFormer(args).to(args.device)
elif args.model=="FreTS":
    model = FreTS(args).to(args.device)
else:
    raise ValueError(f"Model-{args.model} not found")

print(f"Model-{args.model} is loaded")

optimizer=optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

if args.use_amp:
    scaler = torch.cuda.amp.GradScaler()

def validation(args, model, val_loader, criterion, val_type="val"):
    total_loss = []
    total_mae = []
    total_mse = []
    total_mape = []
    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y, mask_1) in tqdm(enumerate(val_loader), total=len(val_loader), desc=f"Validation-{val_type}"):
            optimizer.zero_grad()

            batch_x=batch_x.float().to(args.device) # [batch_size, patch_num, var_num, patch_len]
            batch_y=batch_y.to(args.device) # [batch_size, pred_len, var_num]
            mask_1=mask_1.to(args.device) # [batch_size, patch_num, var_num, patch_len]

            B, P, N, L = batch_x.shape

            autocast_context = torch.cuda.amp.autocast() if args.use_amp else nullcontext()
            
            with autocast_context:
                if args.model == "CRIB":
                    # 为 CRIB 模型应用掩码 (mask)
                    batch_x = batch_x * mask_1[:, :P, ...]
                    # 运行 CRIB 模型的前向传播
                    enc_out_1, enc_attns_1, enc_out_2, enc_attns_2, preds, py_z, kl = model(batch_x, x_mark=None, test_flag=True)
                
                else:  # TSL models: DLinear etc.
                    batch_x = batch_x.permute(0, 2, 1, 3).reshape(B, N, P * L).permute(0, 2, 1)
                    mask_1 = mask_1[:, :P, ...].permute(0, 2, 1, 3).reshape(B, N, P * L).permute(0, 2, 1)
                    
                    batch_x_new = batch_x * mask_1 if 'imputed' not in args.dataset else batch_x
                    
                    preds = model(x_enc=batch_x_new, x_mark_enc=None, x_dec=batch_x_new, x_mark_dec=None)
                    
            # outputs = data_scaler.inverse_transform(outputs)
            # batch_y = data_scaler.inverse_transform(batch_y)

            metric=criterion(preds, batch_y)

            mae, mse, mape = criterion_mae(preds, batch_y), criterion_mse(preds, batch_y), criterion_mape(preds, batch_y)

            total_loss.append(metric.cpu())
            total_mae.append(mae.cpu())
            total_mse.append(mse.cpu())
            total_mape.append(mape.cpu())

    total_loss = np.average(total_loss)
    mae=np.average(total_mae)
    mae=(mae, np.var(total_mae))

    mse=np.average(total_mse)
    mse=(mse, np.var(total_mse))

    mape=np.average(total_mape)
    mape=(mape, np.var(total_mape))
    
    model.train()
    
    return total_loss, mae, mse, mape


def run_validation_and_print(args, model, val_loader, test_loader, criterion):
    model.eval()
    with torch.no_grad(): 
        val_loss, val_mae, val_mse, val_mape = validation(args, model, val_loader, criterion, val_type="val")
        test_loss, test_mae, test_mse, test_mape = validation(args, model, test_loader, criterion, val_type="test")
    
    mu.color_print(
        f'val_loss: {val_loss:.4f}, test_loss: {test_loss:.4f}, '
        f'test_mae: {test_mae[0].item():.4f}+{test_mae[1].item():.4f}, '
        f'test_mse: {test_mse[0].item():.4f}+{test_mse[1].item():.4f}, '
        f'test_mape: {test_mape[0].item():.4f}+{test_mape[1].item():.4f}'
    )
    model.train()
    return val_loss, val_mae, val_mse, val_mape, test_loss, test_mae, test_mse, test_mape


mu.color_print(f'{args.exp_type}-{args.dataset}-{args.model}, seed:{args.seed}, missing_pattern:{args.missing_pattern}, missing_rate: {args.missing_rate}, loss_type: {args.loss_type}, seq: {args.seq_len}, pred: {args.pred_len}')

# val_loss, val_mae, val_mse, val_mape, test_loss, test_mae, test_mse, test_mape = run_validation_and_print(args, model, val_loader, test_loader, criterion_mae)
val_loss, val_mae, val_mse, val_mape, test_loss, test_mae, test_mse, test_mape = 0, 0, 0, 0, 0, 0, 0, 0

autocast_context = torch.cuda.amp.autocast() if args.use_amp else nullcontext()

for iter_count, epoch in (enumerate(range(args.train_epochs))):
    model.train()
    train_loss_list=[]
    
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{args.train_epochs}")
    for i, (batch_x, batch_y, mask_1) in pbar:
        optimizer.zero_grad()

        batch_x=batch_x.float().to(args.device) # [batch_size, patch_num, var_num, patch_len]
        batch_y=batch_y.to(args.device) # [batch_size, pred_len, var_num]
        mask_1=mask_1.to(args.device) # [batch_size, patch_num, var_num, patch_len]

        B, P, N, L = batch_x.shape
        # args.var_num = N
        
        with autocast_context:
            if args.model == "CRIB":
                batch_x = batch_x * mask_1[:, :P, ...]
                enc_out_1, enc_attns_1, enc_out_2, enc_attns_2, preds, py_z, kl = model(batch_x, x_mark=None, test_flag=False) # [batch_size, patch_num, var_num, patch_len]
            else:  # TSL models
                batch_x_flat = batch_x.permute(0, 2, 1, 3).reshape(B, N, P * L).permute(0, 2, 1)
                mask_flat = mask_1[:, :P, ...].permute(0, 2, 1, 3).reshape(B, N, P * L).permute(0, 2, 1)
                batch_x_new = batch_x_flat * mask_flat if 'imputed' not in args.dataset else batch_x_flat
                preds = model(x_enc=batch_x_new, x_mark_enc=None, x_dec=batch_x_new, x_mark_dec=None, mask=None)
                
            
            tra_metric = criterion_mae(preds, batch_y)
            
            if args.model=="CRIB":
                kl_norm=kl/kl.detach()*tra_metric.detach()
                behavior_consistency=criterion_mse(enc_out_1,enc_out_2)
                behavior_consistency_norm=behavior_consistency/behavior_consistency.detach()*tra_metric.detach()
            
            
            loss = 0
            if '1' in args.loss_type:
                loss += tra_metric
            if args.model == "CRIB":
                if '2' in args.loss_type:
                    behavior_consistency = criterion_mse(enc_out_1, enc_out_2)
                    loss += args.Consis_weight * behavior_consistency
                if '3' in args.loss_type:
                    loss += args.KL_weight * kl
                    
            train_loss_list.append(loss.item())

        if args.use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        pbar.set_postfix(
            train_loss=f'{tra_metric.item():.4f}', 
            val_loss=f'{val_loss:.4f}', 
            test_loss=f'{test_loss:.4f}', 
            epoch=f' {epoch + 1} / {args.train_epochs}'
        )
        pbar.update(1)
            
    avg_train_loss = np.average(train_loss_list)
    print(f"Epoch {epoch + 1} Average Train Loss: {avg_train_loss:.4f}")
    
    val_loss, val_mae, val_mse, val_mape, test_loss, test_mae, test_mse, test_mape = run_validation_and_print(args, model, val_loader, test_loader, criterion_mae)

print("Final Validation Results:")
val_loss, val_mae, val_mse, val_mape, test_loss, test_mae, test_mse, test_mape = run_validation_and_print(args, model, val_loader, test_loader, criterion_mae)

print(f"test MAE: {test_loss:.4f}")
print(f"test MSE: {test_mse[0].item():.4f}")

total_params = sum(p.numel() for p in model.parameters())
print(f"Total trainable parameters: {total_params}")

res={
    'Setting': [f'{args.exp_type}-{args.dataset}-{args.model}-{args.missing_pattern}-missing{args.missing_rate}-loss{args.loss_type}-model_dim{args.model_dim}-seq{args.seq_len}-pred{args.pred_len}-seed{args.seed}-iter{args.iter}'],
    'Exp_type': [args.exp_type],
    'Dataset': [args.dataset],
    'Model': [args.model],
    'Missing_pattern': [args.missing_pattern],
    'Missing_rate': [args.missing_rate],
    'Loss_type': [args.loss_type],
    'Model_dim': [args.model_dim],
    'Seed': [args.seed],
    'Seq_len': [args.seq_len],
    'Pred_len': [args.pred_len],
    'Test_MAE': [test_mae[0].item()],
    'Test_MAE_var': [test_mae[1].item()],
    'Test_MSE': [test_mse[0].item()],
    'Test_MSE_var': [test_mse[1].item()],
    'Test_MAPE': [test_mape[0].item()],
    'Test_MAPE_var': [test_mape[1].item()],
}

df = pd.DataFrame(res)
csv_file = args.csv_path

try:
    df_existing = pd.read_csv(csv_file)
    df.to_csv(csv_file, mode='a', index=False, header=False)
except FileNotFoundError:
    df.to_csv(csv_file, mode='w', index=False)

print("Results have been appended to:", csv_file)
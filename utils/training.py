import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from contextlib import nullcontext
from utils import criterion_mape


def validating(args, model, val_loader, criterion, optimizer, val_type="val"):
    criterion_mae=nn.L1Loss()
    criterion_mse=nn.MSELoss()
    
    total_loss = []
    total_mae = []
    total_mse = []
    total_mape = []
    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y, mask_1) in tqdm(enumerate(val_loader), total=len(val_loader), desc=f"Validating-{val_type}"):
            optimizer.zero_grad()

            batch_x=batch_x.float().to(args.device) # [batch_size, patch_num, var_num, patch_len]
            batch_y=batch_y.to(args.device) # [batch_size, pred_len, var_num]
            mask_1=mask_1.to(args.device) # [batch_size, patch_num, var_num, patch_len]

            if args.model != "NeuralCDE":
                B, P, N, L = batch_x.shape

            autocast_context = torch.cuda.amp.autocast() if args.use_amp else nullcontext()
            
            with autocast_context:
                if args.model == "CRIB":
                    # apply mask to CRIB model
                    batch_x = batch_x * mask_1[:, :P, ...]
                    # forward
                    enc_out_1, enc_attns_1, enc_out_2, enc_attns_2, preds, py_z, kl = model(batch_x, x_mark=None, test_flag=True)
                    
                    if args.dataset in ['AQI_ori', 'AQI_imputed']:
                        preds=preds * mask_1[:, :P, ...].permute(0, 2, 1, 3).reshape(B, N, P * L).permute(0, 2, 1)
                    
                elif args.model == "NeuralCDE":
                    coeffs = batch_x  # For NeuralCDE, batch_x already contains the CDE coefficients
                    preds = model(coeffs)
                else:  # TSL models: DLinear etc.
                    batch_x = batch_x.permute(0, 2, 1, 3).reshape(B, N, P * L).permute(0, 2, 1)
                    mask_1 = mask_1[:, :P, ...].permute(0, 2, 1, 3).reshape(B, N, P * L).permute(0, 2, 1)
                    
                    batch_x_new = batch_x * mask_1 if 'imputed' not in args.dataset else batch_x
                    
                    preds = model(x_enc=batch_x_new, x_mark_enc=None, x_dec=batch_x_new, x_mark_dec=None)
                    
                    if args.dataset in ['AQI_ori', 'AQI_imputed']:
                        preds=preds*mask_1
                    
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


def run_validating_and_print(args, model, val_loader, test_loader, criterion, optimizer):
    model.eval()
    with torch.no_grad(): 
        val_loss, val_mae, val_mse, val_mape = validating(args, model, val_loader, criterion, optimizer, val_type="val")
        test_loss, test_mae, test_mse, test_mape = validating(args, model, test_loader, criterion, optimizer, val_type="test")

        val_mae = (val_mae[0].item(), val_mae[1].item())
        val_mse = (val_mse[0].item(), val_mse[1].item())
        val_mape = (val_mape[0].item(), val_mape[1].item())

        test_mae = (test_mae[0].item(), test_mae[1].item())
        test_mse = (test_mse[0].item(), test_mse[1].item())
        test_mape = (test_mape[0].item(), test_mape[1].item())

    
    print(
        f'val_loss: {val_loss:.4f}, test_loss: {test_loss:.4f}, '
        f'test_mae: {test_mae[0]:.4f}+{test_mae[1]:.4f}, '
        f'test_mse: {test_mse[0]:.4f}+{test_mse[1]:.4f}, '
        f'test_mape: {test_mape[0]:.4f}+{test_mape[1]:.4f}'
    )
    model.train()
    return val_loss, val_mae, val_mse, val_mape, test_loss, test_mae, test_mse, test_mape


def common_training_loop(args, model, train_loader, val_loader, test_loader, optimizer, amp_scaler):
    '''
    Training loop for CRIB and model built based on TSL framework
    '''
    criterion_mae=nn.L1Loss()
    criterion_mse=nn.MSELoss()
    
    # val_loss, val_mae, val_mse, val_mape, test_loss, test_mae, test_mse, test_mape = run_validation_and_print(args, model, val_loader, test_loader, criterion_mae, optimizer)
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

            if args.model != "NeuralCDE":
                B, P, N, L = batch_x.shape
            # args.var_num = N
            
            with autocast_context:
                if args.model == "CRIB":
                    batch_x = batch_x * mask_1[:, :P, ...]
                    enc_out_1, enc_attns_1, enc_out_2, enc_attns_2, preds, py_z, kl = model(batch_x, x_mark=None, test_flag=False) # [batch_size, patch_num, var_num, patch_len]
                elif args.model == "NeuralCDE":
                    coeffs = batch_x  # For NeuralCDE, batch_x already contains the CDE coefficients
                    preds = model(coeffs)
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
                    loss += args.IB_weight * tra_metric
                if args.model == "CRIB":
                    if '2' in args.loss_type:
                        behavior_consistency = criterion_mse(enc_out_1, enc_out_2)
                        loss += args.Consis_weight * behavior_consistency
                    if '3' in args.loss_type:
                        loss += args.KL_weight * kl
                        
                train_loss_list.append(loss.item())

            if args.use_amp:
                amp_scaler.scale(loss).backward()
                amp_scaler.step(optimizer)
                amp_scaler.update()
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
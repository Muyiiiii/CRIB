import argparse
import torch

def parse_args(arguments=None):
    parser = argparse.ArgumentParser(description='Hyperparameter settings for MyVar.')

    parser.add_argument('--model', type=str, default='TimesNet', choices=["CRIB", "STJTransformer", "DLinear", "PAttn", "SegRNN", "Transformer", "iTransformer", "PatchTST", "TSMixer", "WPMixer", "CSDI", "NeuralCDE", "ImputeFormer", "TimesNet"], help='Model name')

    parser.add_argument('--data_path', type=str, default='./data', help='Data path')
    parser.add_argument('--dataset', type=str, default='ETTh1', choices=['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Elec', 'PEMS', 'Metr', 'BeijingAir', 'Traffic', 'Weather', 'Illness', 'Exchange', 'PEMS08', 'Elec_imputed', 'ETTh1_imputed', 'PEMS_imputed', 'Metr_imputed', 'AQI_ori', 'AQI_imp'], help='Dataset name')
    
    parser.add_argument('--missing_rate', type=float, default=0.2, help='Missing data rate')
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
    parser.add_argument('--top_k', type=int, default=3, help='top k for TimesNet model')
    parser.add_argument("--num_kernels", type=int, default=3, help="number of kernels of TimesNet",)

    # arguments for PyPOTS models
    parser.add_argument('--n_channels', type=int, default=64, help='Number of channels for PyPOTS models--CSDI')
    parser.add_argument('--d_time_embedding', type=int, default=32, help='Dimension of time embedding for PyPOTS models--CSDI')
    parser.add_argument('--n_diffusion_steps', type=int, default=50, help='Number of diffusion steps for PyPOTS models--CSDI')
    parser.add_argument('--target_strategy', type=str, default='random', choices=['mix', 'random'], help='Target strategy for PyPOTS models--CSDI')
    parser.add_argument('--is_unconditional', type=bool, default=False, help='Unconditional flag for PyPOTS models--CSDI')
    parser.add_argument('--schedule', type=str, default='quad', choices=['quad', 'linear'], help='Noise schedule for PyPOTS models--CSDI')
    parser.add_argument('--beta_start', type=float, default=0.0001, help='Minimum noise level for PyPOTS models--CSDI')
    parser.add_argument('--beta_end', type=float, default=0.5, help='Maximum noise level for PyPOTS models--CSDI')

    parser.add_argument('--input_dim', type=int, default=1, help='Input dimension for PyPOTS models--ImputeFormer')
    parser.add_argument('--output_dim', type=int, default=1, help='Output dimension for PyPOTS models--ImputeFormer')
    parser.add_argument('--ORT_weight', type=float, default=1.0, help='ORT loss weight for PyPOTS models--ImputeFormer')
    parser.add_argument('--MIT_weight', type=float, default=1.0, help='MIT loss weight for PyPOTS models--ImputeFormer')

    # arguments for training PyPOTS models
    parser.add_argument('--patience', type=int, default=None, help='Patience for early stopping for PyPOTS models')
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['adam', 'adamw'], help='Optimizer for PyPOTS models')
    parser.add_argument('--device', type=str, default=None, help='Device for PyPOTS models')
    parser.add_argument('--saving_path', type=str, default=None, help='Model saving path for PyPOTS models')
    parser.add_argument('--model_saving_strategy', type=str, default='best', choices=['best', 'last'], help='Model saving strategy for PyPOTS models')
    parser.add_argument('--verbose', type=bool, default=True, help='Verbose flag for PyPOTS models')
    parser.add_argument('--training_loss', type=str, default='MAE', choices=['MAE', 'MSE'], help='Training loss for PyPOTS models')
    parser.add_argument('--validation_metric', type=str, default='MSE', choices=['MAE', 'MSE'], help='Validation metric for PyPOTS models')

    
    if arguments is None:
        args=parser.parse_args()
    else:
        args=parser.parse_args(args=arguments)

    args=modify_args(args)
    
    return args


def modify_args(args):
    if args.seq_len % args.patch_len != 0:
        raise ValueError(f"seq_len {args.seq_len} must be divisible by patch_len {args.patch_len}")

    args.patch_num = args.seq_len // args.patch_len

    args.n_heads = args.heads_num
    args.d_model = args.model_dim
    args.e_layers = args.enc_num
    args.d_layers = args.dec_num
    
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
    elif 'ETT' in args.dataset: # ETTh1, ETTh2, ETTm1, ETTm2
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
    elif 'Exchange' in args.dataset:
        node_number=8
        args.var_num=8
        args.enc_in = 8
        args.dec_in = 8
        args.c_out = 8
    elif 'Illness' in args.dataset:
        node_number=7
        args.var_num=7
        args.enc_in = 7
        args.dec_in = 7
        args.c_out = 7
    elif 'Traffic' in args.dataset:
        node_number=862
        args.var_num=862
        args.enc_in = 862
        args.dec_in = 862
        args.c_out = 862
    elif 'Weather' in args.dataset:
        node_number=21
        args.var_num=21
        args.enc_in = 21
        args.dec_in = 21
        args.c_out = 21
    elif 'AQI' in args.dataset:
        node_number=36
        args.var_num=36
        args.enc_in = 36
        args.dec_in = 36
        args.c_out = 36

    return args
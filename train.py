import torch
import torch.nn as nn
import pandas as pd
import torch.optim as optim
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader

from utils import (
    parse_args,
    load_dataset,
    criterion_mape,
    set_seed,
    select_model,
    run_validating_and_print,
    common_training_loop,
    convert_dataset_to_pypots_format,
)
from NeuralCDE import CDE_x_augmentation
from PyPOTS_models import CSDI, ImputeFormer


COMMON_MODEL = [
    "CRIB",
    "DLinear", "SegRNN", "Transformer", "iTransformer",
    "PatchTST", "TSMixer", "WPMixer", "PAttn",
    "KANAD", "MultiPatchFormer", "FreTS",
    "TimesNet", "TimeXer",
    "NeuralCDE",
]
PYPOTS_MODEL = ["CSDI", "ImputeFormer"]


args = parse_args()

criterion_mae = nn.L1Loss()
criterion_mse = nn.MSELoss()

if args.seed != -1:
    set_seed(args.seed)

start_time = datetime.now()

train_dataset, val_dataset, test_dataset, _ = load_dataset(args=args, scaler=None)

if args.model == "NeuralCDE":
    train_dataset, val_dataset, test_dataset = CDE_x_augmentation(
        train_dataset, val_dataset, test_dataset
    )

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers, drop_last=False)
val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers, drop_last=False)
test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers, drop_last=False)


if args.model in COMMON_MODEL:
    model = select_model(args)
    print(f"Model-{args.model} is loaded")

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    amp_scaler = torch.cuda.amp.GradScaler() if args.use_amp else None

    print(
        f'{args.exp_type}-{args.dataset}-{args.model}, seed:{args.seed}, '
        f'missing_pattern:{args.missing_pattern}, missing_rate: {args.missing_rate}, '
        f'loss_type: {args.loss_type}, seq: {args.seq_len}, pred: {args.pred_len}'
    )

    common_training_loop(args, model, train_loader, val_loader, test_loader, optimizer, amp_scaler)

    print("Final Validation Results:")
    val_loss, val_mae, val_mse, val_mape, test_loss, test_mae, test_mse, test_mape = run_validating_and_print(
        args, model, val_loader, test_loader, criterion_mae, optimizer
    )
    print(f"test MAE: {test_loss:.4f}")
    print(f"test MSE: {test_mse[0]:.4f}")

    total_params = sum(p.numel() for p in model.parameters())

elif args.model in PYPOTS_MODEL:
    if args.model == "CSDI":
        model = CSDI(args=args)
    else:  # ImputeFormer
        model = ImputeFormer(args=args)

    train_set, val_set, test_set = convert_dataset_to_pypots_format(
        train_dataset, val_dataset, test_dataset
    )

    model.fit(train_set, val_set)

    def _eval(dset, set_pypots):
        preds = torch.tensor(model.impute(set_pypots).squeeze())
        mae  = criterion_mae(preds, dset.pred)
        mse  = criterion_mse(preds, dset.pred)
        mape = criterion_mape(preds, dset.pred)
        return (
            np.average(mae),
            (np.average(mae),  None),
            (np.average(mse),  None),
            (np.average(mape), None),
        )

    val_loss,  val_mae,  val_mse,  val_mape  = _eval(val_dataset,  val_set)
    test_loss, test_mae, test_mse, test_mape = _eval(test_dataset, test_set)

    total_params = sum(p.numel() for p in model.model.parameters())

else:
    raise ValueError(
        f"Model-{args.model} is not registered. Add it to COMMON_MODEL or PYPOTS_MODEL."
    )

print(f"Total trainable parameters: {total_params}")


if torch.cuda.is_available():
    try:
        peak_gpu_mem_bytes = torch.cuda.max_memory_allocated(args.device)
    except Exception:
        peak_gpu_mem_bytes = torch.cuda.max_memory_allocated()
else:
    peak_gpu_mem_bytes = 0
peak_gpu_mem_mb = round(peak_gpu_mem_bytes / (1024 ** 2), 2)

end_time = datetime.now()
elapsed_seconds = round((end_time - start_time).total_seconds(), 2)


res = {
    'Setting': [
        f'{args.exp_type}-{args.dataset}-{args.model}-{args.missing_pattern}'
        f'-missing{args.missing_rate}-loss{args.loss_type}'
        f'-model_dim{args.model_dim}-seq{args.seq_len}-pred{args.pred_len}'
        f'-seed{args.seed}-iter{args.iter}'
    ],
    'Exp_type':        [args.exp_type],
    'Dataset':         [args.dataset],
    'Model':           [args.model],
    'Missing_pattern': [args.missing_pattern],
    'Missing_rate':    [args.missing_rate],
    'Loss_type':       [args.loss_type],
    'Model_dim':       [args.model_dim],
    'IB_weight':       [args.IB_weight],
    'KL_weight':       [args.KL_weight],
    'Consis_weight':   [args.Consis_weight],
    'Seed':            [args.seed],
    'Seq_len':         [args.seq_len],
    'Pred_len':        [args.pred_len],
    'Test_MAE':          [test_mae[0]],   'Test_MAE_var':  [test_mae[1]],
    'Test_MSE':          [test_mse[0]],   'Test_MSE_var':  [test_mse[1]],
    'Test_MAPE':         [test_mape[0]],  'Test_MAPE_var': [test_mape[1]],
    'Max_GPU_Memory_MB': [peak_gpu_mem_mb],
    'Start_time':        [start_time.strftime('%Y-%m-%d %H:%M:%S')],
    'End_time':          [end_time.strftime('%Y-%m-%d %H:%M:%S')],
    'Elapsed_seconds':   [elapsed_seconds],
}

df = pd.DataFrame(res)
csv_file = args.csv_path

try:
    df_existing = pd.read_csv(csv_file)
    df.to_csv(csv_file, mode='a', index=False, header=False)
    print(f"Appended results to existing file '{csv_file}' (previous rows: {len(df_existing)}).")
except FileNotFoundError:
    df.to_csv(csv_file, mode='w', index=False)
    print(f"Created results file '{csv_file}' and wrote header + first row.")

print('\n')
print("Run start:",        start_time.strftime('%Y-%m-%d %H:%M:%S'))
print("Run end:",          end_time.strftime('%Y-%m-%d %H:%M:%S'))
print("Elapsed seconds:",  elapsed_seconds)
print("GPU Peak Memory MB:", peak_gpu_mem_mb)

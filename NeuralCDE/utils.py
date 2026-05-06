import torch
import torchcde
from torch.utils.data import Dataset

def CDE_x_augmentation(train_dataset, val_dataset, test_dataset):
    Xs=[train_dataset.data, val_dataset.data, test_dataset.data]
    Masks=[train_dataset.mask_1, val_dataset.mask_1, test_dataset.mask_1]
    Xs_augmented=[]
    for xs, masks in zip(Xs, Masks):
        B, P, N, L = xs.shape  # Batch, Patch_Num, Var_Num, Patch_Len
        # Change order: [B, P, N, L] -> [B, P, L, N] -> [B, P*N, L]
        xs = xs.permute(0, 1, 3, 2).reshape(B, P * L, N)
        masks = masks[:, :P, ...].permute(0, 1, 3, 2).reshape(B, P * L, N)
        
        xs = xs * masks
        
        T_total = P * L
        t=torch.linspace(0, 1, T_total).unsqueeze(0).unsqueeze(-1).expand(B, T_total, 1).to(xs.device)
        xs = torch.cat([t, xs], dim=2)  # [B, T, N + 1]
        
        print(f"Calculating CDE coefficients for training dataset (Shape: {xs.shape})...")
        xs = torchcde.hermite_cubic_coefficients_with_backward_differences(xs)
        print("Training Coefficient calculation completed.")
        Xs_augmented.append(xs)
        
    train_dataset.data = Xs_augmented[0]
    val_dataset.data = Xs_augmented[1]
    test_dataset.data = Xs_augmented[2]
    
    return train_dataset, val_dataset, test_dataset


class CDE_Adapter_Dataset(Dataset):
    def __init__(self, x, y, mask=None, use_mask_as_channel=True):
        """
        Args:
        x: shape [Batch, Patch_Num, Var_Num, Patch_Len]
        y: shape [Batch, Pred_Len, Var_Num]
        mask: shape [Batch, Patch_Num, Var_Num, Patch_Len] (optional)
        use_mask_as_channel: whether to concatenate mask as a feature channel to x (recommended True)
        """
        
        B, P, V, L = x.shape
        # Change order: [B, P, V, L] -> [B, P, L, V] -> [B, P*L, V]
        x_flat = x.permute(0, 1, 3, 2).reshape(B, P * L, V)
    
            
        # Add absolute time channel (Time Channel)
        # Neural CDE requires time as part of the input
        T_total = P * L
        t = torch.linspace(0, 1, T_total).unsqueeze(0).unsqueeze(-1).expand(B, T_total, 1).to(x_flat.device)
        
        # Final concatenation: [B, T, Feature_Dim + 1]
        x_augmented = torch.cat([t, x_flat], dim=2)
        
        self.input_channels = x_augmented.shape[-1] # Record the input channel number, needed when defining the model
        
        # 4. Calculate interpolation coefficients (Coefficients)
        print(f"Calculating CDE coefficients for dataset (Shape: {x_augmented.shape})...")
        # This is a time-consuming operation. If you encounter out-of-memory errors on GPU, please move .to(device) to getitem.
        self.coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(x_augmented)
        print("Coefficient calculation completed.")

    def __len__(self):
        return len(self.coeffs)

    def __getitem__(self, idx):
        # Return: (interpolation coefficients, label)
        return self.coeffs[idx], self.y[idx]
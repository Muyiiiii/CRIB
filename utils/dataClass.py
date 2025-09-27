from torch.utils.data import Dataset, DataLoader

class MyStandardScaler:
    """
    Standard the input
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean
    
class MyDataset(Dataset):
    def __init__(self, args, data, pred, mask_1):
        super().__init__()
        self.args=args
        self.data=data
        self.pred=pred
        self.mask_1=mask_1
    
    def __getitem__(self, idx):
        x=self.data[idx]
        y=self.pred[idx]
        mask_1=self.mask_1[idx]

        return x, y, mask_1
    
    def __len__(self):
        return self.data.shape[0]
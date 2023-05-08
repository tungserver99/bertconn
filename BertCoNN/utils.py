import time
import pandas as pd
import torch
from torch.utils.data import Dataset


def date(f='%Y-%m-%d %H:%M:%S'):
    return time.strftime(f, time.localtime())

def predict_mse(model, dataloader, device):
    mse, sample_count = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            user_reviews, item_reviews, ratings = map(lambda x: x.to(device), batch)
            predict = model(user_reviews, item_reviews)
            mse += torch.nn.functional.mse_loss(predict, ratings, reduction='sum').item()
            sample_count += len(ratings)
    return mse / sample_count  # dataloader上的均方误差


class DeepCoNNDataset(Dataset):
    def __init__(self, data_df_path):
        self.user = self._get_info(data_df_path, "user")
        self.item = self._get_info(data_df_path, "item")
        self.rating = self._get_info(data_df_path, "rating")

    def __getitem__(self, idx):
        return self.user[idx], self.item[idx], self.rating[idx]

    def __len__(self):
        return len(self.rating)
    
    def _get_info(self, data_df_path, field):
        df = pd.read_csv(data_df_path)
        ID_list = list(df[field])
        return ID_list


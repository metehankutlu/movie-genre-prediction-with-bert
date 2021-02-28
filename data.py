import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import pytorch_lightning as pl

class MovieDataset(Dataset):
    def __init__(self, dataframe, tokenizer, genre_names=None, max_length=200):
        super(MovieDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.df = dataframe
        self.genre_names = genre_names
        
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        synopsis = row.synopsis
        if self.genre_names is not None:
            genres = row[self.genre_names]
        encodings = self.tokenizer.encode_plus(
            synopsis,
            add_special_tokens=True,
            return_token_type_ids=False,
            max_length=self.max_length,
            padding='max_length', 
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        res = dict(
            synopsis=synopsis,
            input_ids=encodings['input_ids'].flatten(),
            attention_mask=encodings['attention_mask'].flatten(),
        )

        if self.genre_names is not None:
            res['genres'] = torch.FloatTensor(genres)
        
        return res

class MovieGenreDataModule(pl.LightningDataModule):
    def __init__(self, train_df, test_df, tokenizer, genre_names, batch_size=8, max_token_len=200):
        super().__init__()

        self.train_df = train_df
        self.test_df = test_df
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_token_len = max_token_len
        self.genre_names = genre_names

    def setup(self):
        self.train_dataset = MovieDataset(
            self.train_df, 
            self.tokenizer,
            self.genre_names,
            self.max_token_len
        )

        self.test_dataset = MovieDataset(
            self.test_df,
            self.tokenizer,
            self.genre_names,
            self.max_token_len
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8
        )

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, num_workers=4)
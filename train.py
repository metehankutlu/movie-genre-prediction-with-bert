import os
import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
import pytorch_lightning as pl
from tqdm import tqdm
from data import MovieGenreDataModule, MovieDataset
from model import MovieGenreModel

def main(config):
    BERT_MODEL_NAME = config['BERT_MODEL_NAME']
    N_EPOCHS = config['N_EPOCHS']
    BATCH_SIZE = config['BATCH_SIZE']
    LR = config['LR']
    CHECKPOINT_DIR = config['CHECKPOINT_DIR']
    CHECKPOINT = os.path.join(CHECKPOINT_DIR, 'model_lr_' + str(LR) + '.ckpt')

    df = pd.read_csv('train.csv')

    vectorizer = CountVectorizer(tokenizer = lambda x: x.split(' '))
    genres_dtm = vectorizer.fit_transform(df['genres'])
    genre_names = np.array(vectorizer.get_feature_names())
    genres_df = pd.DataFrame(genres_dtm.toarray(), columns=genre_names)
    df = pd.concat([df, genres_df], axis=1)
    train_df, test_df = train_test_split(df, test_size=0.2)

    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

    data_module = MovieGenreDataModule(train_df, test_df, tokenizer, genre_names)
    data_module.setup()

    model = MovieGenreModel(
        BERT_MODEL_NAME, 
        genre_names, 
        lr=LR,
        steps_per_epoch=len(train_df)//BATCH_SIZE,
        n_epochs=N_EPOCHS
    )

    trainer = pl.Trainer(max_epochs=N_EPOCHS, progress_bar_refresh_rate=30)
    trainer.fit(model, data_module)

    if not os.path.exists(CHECKPOINT_DIR):
        os.mkdir(CHECKPOINT_DIR)

    trainer.save_checkpoint(CHECKPOINT)

if __name__ == '__main__':
    with open('config.json') as json_file:
        config = json.load(json_file)
    main(config)
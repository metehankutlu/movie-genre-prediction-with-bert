import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from data import MovieDataset
from model import MovieGenreModel

def main(config):
    BERT_MODEL_NAME = config['BERT_MODEL_NAME']
    N_EPOCHS = config['N_EPOCHS']
    BATCH_SIZE = config['BATCH_SIZE']
    LR = config['LR']
    CHECKPOINT_DIR = config['CHECKPOINT_DIR']
    CHECKPOINT = os.path.join(CHECKPOINT_DIR, 'model_lr_' + str(LR) + '.ckpt')

    df = pd.read_csv('test.csv')
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

    dataset = MovieDataset(df, tokenizer)

    test_data = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=8
    )

    trained_model = MovieGenreModel.load_from_checkpoint(CHECKPOINT)
    trained_model.freeze()

    all_pred_genres = np.empty((0, 1),str)
    ids = np.expand_dims(df['movie_id'].values, axis=1)

    genres = np.array(trained_model.label_cols)

    for batch in tqdm(test_data, leave=False):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        _, prediction = trained_model(input_ids, attention_mask)
        pred_list = [list(zip(range(19), pred)) for pred in prediction]
        pred_list = [sorted(pred, key=lambda x: x[1], reverse=True)[:5] for pred in pred_list]
        pred_indexes = [np.array([p[0] for p in pred], dtype=int) for pred in pred_list]
        pred_genres = [genres[index] for index in pred_indexes]
        pred_genres = [np.array([p.capitalize() for p in pred]) for pred in pred_genres]
        pred_genres = [np.array([p.replace('-f', '-F') for p in pred]) for pred in pred_genres]
        pred_genres = np.array([" ".join(pred) for pred in pred_genres])
        all_pred_genres = np.append(all_pred_genres, pred_genres)

    all_pred_genres = np.expand_dims(all_pred_genres, axis=1)
    prediction_df = pd.DataFrame(
        np.concatenate([ids, all_pred_genres], axis=1), 
        columns=['movie_id', 'predicted_genres']
    )

    prediction_df.to_csv('prediction.csv', index=False)

if __name__ == '__main__':
    with open('config.json') as json_file:
        config = json.load(json_file)
    main(config)
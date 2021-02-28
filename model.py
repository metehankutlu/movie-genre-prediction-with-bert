import torch
import pytorch_lightning as pl
from transformers import BertModel, AdamW, get_linear_schedule_with_warmup
from pytorch_lightning.metrics.functional.classification import auroc  

class MovieGenreModel(pl.LightningModule):
    def __init__(self, bert_model_name, label_cols, lr, steps_per_epoch=None, n_epochs=None):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name, return_dict=True)
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, len(label_cols))
        self.dropout = torch.nn.Dropout(0.2)
        self.out = torch.nn.Sigmoid()
        self.steps_per_epoch = steps_per_epoch
        self.n_epochs = n_epochs
        self.criterion = torch.nn.BCELoss()
        self.label_cols = label_cols
        self.lr = lr
        self.save_hyperparameters()

    def forward(self, input_ids, attention_mask, genres=None):
        output = self.bert(input_ids, attention_mask)
        output = self.classifier(output.pooler_output)
        output = self.dropout(output)
        output = self.out(output)
        
        loss = 0
        if genres is not None:
            loss = self.criterion(output, genres)

        return loss, output

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        genres = batch['genres']

        loss, outputs = self(input_ids, attention_mask, genres)
        self.log('train_loss', loss, prog_bar=True, logger=True)

        return {
            'loss': loss,
            'predictions': outputs,
            'genres': genres
        }

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        genres = batch['genres']

        loss, outputs = self(input_ids, attention_mask, genres)
        self.log('val_loss', loss, prog_bar=True, logger=True)

        return loss
    
    def test_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        genres = batch['genres']

        loss, outputs = self(input_ids, attention_mask, genres)
        self.log('test_loss', loss, prog_bar=True, logger=True)

        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr)

        warmup_steps = self.steps_per_epoch // 3
        total_steps = self.steps_per_epoch * self.n_epochs - warmup_steps

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            warmup_steps,
            total_steps
        )

        return [optimizer], [scheduler]
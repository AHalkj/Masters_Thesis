import torch

from torch.nn import Linear, Module
from torch.utils.data import Dataset, RandomSampler ,SequentialSampler
from transformers import DistilBertModel
import os
from torch.utils.data import DataLoader
from Utils import get_ids_mask_labels_from_batch,AverageMeter
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report



# Define a few-shot model using BERT

class BertFewShotModel(Module):
    def __init__(self, num_classes):
        super(BertFewShotModel, self).__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.classifier = Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        return logits

# Define a few-shot trainer class
class FewShotTrainer:
    def __init__(self, model, train_dataset, val_dataset, device):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = device


    def train(self, num_epochs,device,model, batch_size, learning_rate, num_support, num_query):
        # Define loss function and optimizer
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

        # Create data loaders for training and validation datasets
        train_loader = DataLoader(
            self.train_dataset,
            sampler=RandomSampler(self.train_dataset),
            batch_size=int(batch_size),
            num_workers=6
        )

        val_loader = DataLoader(
            self.val_dataset,
            sampler=SequentialSampler(self.val_dataset),
            batch_size=int(batch_size),
            num_workers=6
        )


        for epoch in range(num_epochs):
            self.model.train()
            losses = AverageMeter()
            total_loss = 0.0
            correct = 0
            total = 0

            # Training loop
            tk0 = tqdm(train_loader, total = len(train_loader))
            for i , batch in enumerate(tk0):
                # Extract support_set and query_set from the batch

                support_sample_ids, query_sample_ids = torch.tensor_split(batch['ids'],2 )
                support_sample_mask, query_sample_mask = torch.tensor_split(batch['attention_mask'],2 )
                support_sample_labels, query_sample_labels = torch.tensor_split(batch['label'].to(device, dtype=torch.float),2  ) 

                optimizer.zero_grad()

                support_outputs = model(input_ids=support_sample_ids, attention_mask=support_sample_mask)
                support_outputs = support_outputs.logits
                support_loss = criterion(support_outputs, support_sample_labels)
                support_loss.backward()
                losses.update(support_loss.item(), support_sample_ids.size(0))

                    
                query_outputs = model(input_ids=query_sample_ids, attention_mask=query_sample_mask)
                query_outputs = query_outputs.logits
                query_loss = criterion(query_outputs, query_sample_labels)
                query_loss.backward()
                losses.update(query_loss.item(), query_sample_ids.size(0))

                optimizer.step()

            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {losses.avg:.4f}')


            evaluate_step(model, val_loader, device)

def evaluate_step(model, dataloader, device):
    model.eval()


    validation_set_predictions = []
    validation_set_labels = []

    with torch.no_grad():
        tk = tqdm(dataloader, total = len(dataloader))
        for _, batch in enumerate(tk):
            ids, mask, labels = get_ids_mask_labels_from_batch(batch, device)

            eval_output = model(ids, attention_mask=mask)
            logits = eval_output.logits.detach().cpu().numpy()
            preds = np.argmax(logits, axis=1).tolist()
            labels = labels.cpu().numpy().tolist()

            validation_set_predictions.extend(preds)
            for i in labels:
                if i ==  [0.0 , 1.0]:
                    validation_set_labels.append(1)
                else:
                    validation_set_labels.append(0)
    # Compute evaluation metrics
    print(classification_report(validation_set_labels, validation_set_predictions, zero_division=1))







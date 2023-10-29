
import argparse
import os
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification

from Config import EPOCHS
from Utils import AverageMeter, save_model, get_device
import Engine
from Fewshot import FewShotTrainer
from Dataset import FewShotDataSet
import mlflow
import torch
import torch.nn as nn

from Engine import train_and_evaluate
def select_first_file(path):
    """Selects first file in folder, use under assumption there is only one file in folder
    Args:
        path (str): path to directory or file to choose
    Returns:
        str: full path of selected file
    """
    files = os.listdir(path)
    return os.path.join(path, files[0])

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, support_embeddings, query_embeddings):
        # Calculate pairwise Euclidean distances between support and query embeddings
        distances = torch.norm(support_embeddings - query_embeddings, p=2, dim=1)

        # Calculate contrastive loss
        loss = 0.5 * ((1 - distances)**2) + 0.5 * ((torch.clamp(self.margin - distances, min=0))**2)

        return loss.mean()

def main():
    os.environ['CURL_CA_BUNDLE'] = ''

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, help="path to train data")
    parser.add_argument("--model_dir", type=str, help="path to model output")
    parser.add_argument("--learning_rate", type=float, help="learning rate for training")
    parser.add_argument("--undersampling_rate", type=float, help="undersampling rate for majority class")
    parser.add_argument("--batch_size", type=int, help="batch size for training")
    args = parser.parse_args()

    # Load and preprocess data
    df = pd.read_csv(args.train_path)
    df.text = df.text.fillna('')
    df['label'] = df.label.replace({'False': 0, 'True': 1, '0.0': 0, '1.0': 1, '1': 1, '0': 0})

    random_state = random.randint(0, 200)
    mlflow.log_metric("num_samples", len(df))
    mlflow.log_metric("random_state_split", random_state)
    print(df.label.value_counts())
    mlflow.log_metric("Negative Count", df.label.value_counts().to_list()[0])
    mlflow.log_metric("Positive Count", df.label.value_counts().to_list()[1])

    df_train, df_val = train_test_split(df, test_size=0.3, random_state=random_state)

    # Update dataset creation
    train_set = FewShotDataSet(df_train, num_support=10, num_query=10)
    val_set = FewShotDataSet(df_val, num_support=10, num_query=10)

    train_dataloader = DataLoader(
        train_set,
        sampler=RandomSampler(train_set),
        batch_size=int(args.batch_size),
        num_workers=6
    )

    validation_dataloader = DataLoader(
        val_set,
        sampler=SequentialSampler(val_set),
        batch_size=int(args.batch_size),
        num_workers=6
    )

    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=2,
    )

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=float(args.learning_rate),
                                  eps=1e-08
                                  )

    device = get_device()
    model.to(device)

    # Instantiate the contrastive loss function
    loss_fn = ContrastiveLoss(margin=1.0)

    losses = AverageMeter()
    recalls = AverageMeter()
    precisions = AverageMeter()
    accuracies = AverageMeter()

    few_shot_trainer = FewShotTrainer(model, train_dataloader, validation_dataloader, device)

    few_shot_trainer.train(num_epochs=EPOCHS, batch_size=int(args.batch_size), learning_rate=float(args.learning_rate),
                           num_support=10, num_query=10)

    for epoch in range(EPOCHS):
        losses.reset()

        print(f'STARTING EPOCH {epoch}...................')

        model.train()
        for batch in train_dataloader:
            support_set = batch["support_set"]
            query_set = batch["query_set"]

            # Forward pass for support and query sets
            support_embeddings = model(input_ids=support_set['ids'].to(device),
                                       attention_mask=support_set['attention_mask'].to(device))[0]

            query_embeddings = model(input_ids=query_set['ids'].to(device),
                                     attention_mask=query_set['attention_mask'].to(device))[0]

            # Compute loss using contrastive loss
            loss = loss_fn(support_embeddings, query_embeddings)

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())

        mlflow.log_metric('Train Loss', losses.avg, epoch)
        print('Train Loss: ', losses.avg)

        print(f'STARTING EVALUATION {epoch}...................')
        model.eval()

        accuracies, recalls, precisions = Engine.evaluate(model, validation_dataloader, device, optimizer, accuracies,
                                                          recalls, precisions)

    save_model(args.model_dir, model)


if __name__ == "__main__":
    main()

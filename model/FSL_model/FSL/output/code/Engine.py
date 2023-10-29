import logging

import torch
from tqdm import tqdm
import numpy as np
import mlflow

from Utils import b_metrics, evaluation_metrics, get_ids_mask_labels_from_batch, AverageMeter
from Config import criterion


def train(model, dataloader, device, optimizer, losses, scheduler):
    model.train()
    tk0 = tqdm(dataloader, total=len(dataloader))
    for _, batch in enumerate(tk0):
        support_set, query_set = batch  # Modify this line to unpack support_set and query_set
        support_set = support_set.to(device)
        query_set = query_set.to(device)

        model.zero_grad()

        train_output = model(support_set, query_set)
        train_output.loss.backward()
        optimizer.step()
        scheduler.step()

        losses.update(train_output.loss.item(), query_set.size(0))

    return losses



def train_and_evaluate(model, train_dataloader, validation_dataloader, device, optimizer, num_support, num_query, num_epochs, learning_rate):
    model.train()
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        losses = AverageMeter()

        for i, batch in enumerate(train_dataloader):
            support_set, query_set = batch

            support_set = [sample.to(device) for sample in support_set]
            query_set = [sample.to(device) for sample in query_set]

            optimizer.zero_grad()

            for support_sample in support_set:
                support_sample_ids, support_sample_mask, support_sample_labels = get_ids_mask_labels_from_batch(
                    support_sample, device)
                support_outputs = model(input_ids=support_sample_ids, attention_mask=support_sample_mask)
                support_loss = criterion(support_outputs, support_sample_labels)
                support_loss.backward()
                losses.update(support_loss.item(), support_sample_ids.size(0))

            for query_sample in query_set:
                query_sample_ids, query_sample_mask, query_sample_labels = get_ids_mask_labels_from_batch(
                    query_sample, device)
                query_outputs = model(input_ids=query_sample_ids, attention_mask=query_sample_mask)
                query_loss = criterion(query_outputs, query_sample_labels)
                query_loss.backward()
                losses.update(query_loss.item(), query_sample_ids.size(0))

            optimizer.step()

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {losses.avg:.4f}')

        # Evaluate at the end of each epoch
        evaluate_step(model, validation_dataloader, device)

    print("Training complete!")

def evaluate_step(model, dataloader, device):
    model.eval()

    validation_set_predictions = []
    validation_set_labels = []

    with torch.no_grad():
        for _, batch in enumerate(dataloader):
            ids, mask, labels = get_ids_mask_labels_from_batch(batch, device)

            eval_output = model(ids, attention_mask=mask)
            logits = eval_output.logits.detach().cpu().numpy()
            preds = np.argmax(logits, axis=1).tolist()
            labels = labels.cpu().numpy().tolist()

            validation_set_predictions.extend(preds)
            validation_set_labels.extend(labels)

    # Compute evaluation metrics
    b_accuracy, b_precision, b_recall = evaluation_metrics(validation_set_labels, validation_set_predictions)
    print("Validation Accuracy:", b_accuracy)
    print("Validation Precision:", b_precision)
    print("Validation Recall:", b_recall)



def batch_metrics(logits, labels):
    preds = np.argmax(logits, axis=1)  # Convert logits to predicted class indices
    tp = np.sum((preds == 1) & (labels == 1))
    tn = np.sum((preds == 0) & (labels == 0))
    fp = np.sum((preds == 1) & (labels == 0))
    fn = np.sum((preds == 0) & (labels == 1))

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 'nan'
    recall = tp / (tp + fn) if (tp + fn) > 0 else 'nan'

    return accuracy, precision, recall


def evaluate(model, dataloader, device, optimizer, accuracies, recalls, precisions):
    model.eval()
    tk0 = tqdm(dataloader, total=len(dataloader))
    for _, batch in enumerate(tk0):
        ids = batch["ids"]
        mask = batch["attention_mask"]
        labels = batch["label"]
        ids = ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        labels = labels.to(device, dtype=torch.float)

        with torch.no_grad():
            eval_output = model(
                ids,
                attention_mask=mask,
                labels=labels)

        # logits = eval_output.logits.detach().cpu()
        # logits_np= logits.numpy()
        label_ids_np = labels.cpu().numpy()

        logits = eval_output.logits.detach().cpu().numpy()

        # Calculate validation metrics
        b_accuracy, b_precision, b_recall = batch_metrics(logits, label_ids_np)
        print(b_accuracy, b_precision, b_recall)

        # Update precision, recall, accuracy only when (tp + fp) !=0; ignore nan
        # if b_precision != 'nan': precisions.update(b_precision, ids.size(0)) #val_precision.append(b_precision)
        # if b_recall != 'nan': recalls.update(b_recall, ids.size(0))
        # if b_accuracy != 'nan': accuracies.update(b_accuracy, ids.size(0))

        mlflow.log_metric('Validation Precision', precisions.avg)
        mlflow.log_metric('Validation Recall', recalls.avg)
        mlflow.log_metric('Validation Accuracy', accuracies.avg)



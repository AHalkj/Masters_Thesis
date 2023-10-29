from torch import nn
import mlflow
from pathlib import Path
import shutil
import numpy as np
import torch
from sklearn.metrics import accuracy_score, recall_score, precision_score

def get_device():
    '''
    Checks if gpu is available and if yes returns that as device
    '''
    if torch.cuda.is_available():

        # Tell PyTorch to use the GPU.
        device = torch.device("cuda:0")

        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))

  # If not...
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    return device

def b_tp(preds, labels):
    '''Returns True Positives (TP): count of correct predictions of actual class 1'''
    return sum([preds == labels and preds == 1 for preds, labels in zip(preds, labels)])

def b_fp(preds, labels):
    '''Returns False Positives (FP): count of wrong predictions of actual class 1'''
    return sum([preds != labels and preds == 1 for preds, labels in zip(preds, labels)])

def b_tn(preds, labels):
    '''Returns True Negatives (TN): count of correct predictions of actual class 0'''
    return sum([preds == labels and preds == 0 for preds, labels in zip(preds, labels)])

def b_fn(preds, labels):
    '''Returns False Negatives (FN): count of wrong predictions of actual class 0'''
    return sum([preds != labels and preds == 0 for preds, labels in zip(preds, labels)])

def b_metrics(logits, labels):
    '''
    Formulas for metrics:
      - accuracy    = (TP + TN) / N
      - precision   = TP / (TP + FP)
      - recall      = TP / (TP + FN)
      - specificity = TN / (TN + FP)
    '''
    preds = np.argmax(logits, axis=1).flatten()
    labels = labels.flatten()
    tp = b_tp(preds, labels)
    tn = b_tn(preds, labels)
    fp = b_fp(preds, labels)
    fn = b_fn(preds, labels)
    b_accuracy = (tp + tn) / len(labels)
    b_precision = tp / (tp + fp) if (tp + fp) > 0 else 'nan'
    b_recall = tp / (tp + fn) if (tp + fn) > 0 else 'nan'
    b_specificity = tn / (tn + fp) if (tn + fp) > 0 else 'nan'
    return b_accuracy, b_precision, b_recall, b_specificity

def evaluation_metrics(labels, preds):
    b_accuracy = accuracy_score(labels, preds)
    b_precision = precision_score(labels, preds, zero_division=np.nan)
    b_recall = recall_score(labels, preds, zero_division=np.nan)
    return b_accuracy, b_precision, b_recall

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_model(model_dir: str, model: nn.Module) -> None:
    """
    Saves the trained model.
    """

    code_paths = ["Train.py", "Config.py", "Engine.py"]
    full_code_paths = [
        Path(Path(__file__).parent, code_path) for code_path in code_paths
    ]

    shutil.rmtree(model_dir, ignore_errors=True)
    print(f"Saving model to {model_dir}")
    mlflow.pytorch.save_model(pytorch_model=model,
                              path=model_dir,
                              code_paths=full_code_paths,
                              # signature=signature
                              )

def preprocessing(input_text, tokenizer):

    return tokenizer.encode_plus(
        input_text,
        add_special_tokens=True,
        padding='max_length',
        truncation=True,
        max_length=128,
        return_attention_mask=True,
        return_tensors='pt'
    )

def get_ids_mask_labels_from_batch(batch, device):
    ids = batch["ids"]
    mask = batch["attention_mask"]
    labels = batch["label"]
    ids = ids.to(device, dtype=torch.long)
    mask = mask.to(device, dtype=torch.long)
    labels = labels.to(device, dtype=torch.float)
    return ids, mask, labels
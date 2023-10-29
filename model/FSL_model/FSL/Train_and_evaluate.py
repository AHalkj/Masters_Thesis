import argparse
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import os
import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from transformers import BertForSequenceClassification, get_linear_schedule_with_warmup
import torch
import wandb
import random
import pandas as pd
from Config import EPOCHS
from Utils import AverageMeter, save_model, get_device
from Dataset import LinesDataSet
from Fewshot import FewShotTrainer
from Engine import evaluate_step, get_ids_mask_labels_from_batch, train_and_evaluate
#import pdb



def main():
    os.environ['CURL_CA_BUNDLE'] = ''
    """Main function of the script."""
    parser = argparse.ArgumentParser()

    # Add arguments to the parser
    parser.add_argument("--train_path", type=str, help="path to train data")
    parser.add_argument("--model_dir", type=str, help="path to model output")
    parser.add_argument("--learning_rate", type=float, help="learning rate")
    parser.add_argument("--undersampling_rate", type=float, help="undersampling rate")
    parser.add_argument("--batch_size", type=int, help="batch size")
    parser.add_argument("--num_support", type=int, default=5)
    parser.add_argument("--num_query", type=int, default=5)

    # Parse the command-line arguments
    args = parser.parse_args()

    # paths are mounted as folder, therefore, we are selecting the file from folder
    df = pd.read_csv(args.train_path)
    print(args.train_path)
    df.text = df.text.fillna('')
    df['label']=df['label'].astype(str).apply(lambda x: x.strip())
    df['label'] = df.label.replace({'False': 0, 'True': 1, '0.0': 0, '1.0': 1, '1': 1, '0': 0, '2.0' : 1})
    random_state = random.randint(0, 200)
    mlflow.log_metric("num_samples", len(df))
    mlflow.log_metric("random_state_split", random_state)
    print(df.label.value_counts())
    mlflow.log_metric("Negative Count", df.label.value_counts().to_list()[0])
    mlflow.log_metric("Positive Count", df.label.value_counts().to_list()[1])

    df_train, df_val = train_test_split(df, test_size=0.2, random_state=random_state)
    df_val=df_val[:100]
    # Process the new CSV data into support and query sets
    support_query_df = pd.read_csv('C:/repos/Thesis/AHAKLWE/Model/FSL_model/250811_training_data.csv')
    support_query_df['label'] = support_query_df.label.replace({'False': 0, 'True': 1, '0.0': 0, '1.0': 1, '1': 1, '0': 0, '2' : 1})
    # Combine the existing training set and the support/query set
    combined_df_train = pd.concat([df_train, support_query_df], ignore_index=True)
    combined_df_train=combined_df_train[:250]
    combined_train_set = LinesDataSet(combined_df_train)

    # Create FewShotDataSet
    val_set = LinesDataSet(df_val)

    print(support_query_df.label.value_counts())

#    pdb.set_trace()
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=2,
    )

    # Recommended learning rates (Adam): 5e-5, 3e-5, 2e-5. See: https://arxiv.org/pdf/1810.04805.pdf

    # mode to CUDA only when possible
    device = get_device()
    model.to(device)
    print(f'Start training on {device} ------------')

    # Initialize the FewShotTrainer
    few_shot_trainer = FewShotTrainer(model, combined_train_set, val_set, device)

    # Train the few-shot model
    few_shot_trainer.train(num_epochs=EPOCHS,device=device,model=model, batch_size=args.batch_size, learning_rate=args.learning_rate,
                           num_support=args.num_support, num_query=args.num_query)

    save_model(args.model_dir, model)


if __name__ == "__main__":

    main()

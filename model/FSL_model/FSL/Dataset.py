import pandas as pd
import torch
from Config import TOKENIZER, MAX_LEN
from torch.utils.data import DataLoader, SequentialSampler
from transformers import BertTokenizer
import pandas as pd
import torch
from transformers import BertTokenizer

class LinesDataSet:
    def __init__(self, df: pd.DataFrame):
        assert set(['label', 'text']).intersection(set(df.columns)) == set(
            ['label', 'text']), 'Dataframe is missing columns'

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.max_len = MAX_LEN
        self.df = self.split_text(df)
        self.l=[]
        self.l2=[]

    def __len__(self):
        return len(self.df)

    def split_text(self, df):
        rows = []
        for i, row in df.iterrows():
            if len(str(row.text).split()) > self.max_len - 2:
                for j in range(0, len(str(row.text).split()), self.max_len - 2):
                    row = pd.Series({'text': ' '.join(str(row.text).split()[j:j+MAX_LEN-2]), 'label': row.label})
                    rows.append(row)
            else:
                rows.append(row)
        df = pd.DataFrame(rows)
        return df

    def __getitem__(self, item):
        text = str(self.df.iloc[item].text)
        encoding_dict = self.tokenizer(text, truncation=True, max_length=self.max_len, padding='max_length')

        ids = encoding_dict['input_ids']
        attention_mask = encoding_dict['attention_mask']
        label = int(self.df.iloc[item].label)
        if label == 1:
            label = [0, 1]
        else:
            label = [1, 0]
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long)
        }

class FewShotDataSet():
    def __init__(self, df: pd.DataFrame, num_support=5, num_query=5):
        assert set(['label', 'text']).intersection(set(df.columns)) == set(
            ['label', 'text']), 'Dataframe is missing columns'

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.max_len = MAX_LEN
        self.num_support = num_support
        self.num_query = num_query
        self.tasks = self.generate_tasks(df)

    def __len__(self):
        return len(self.tasks)

    def generate_tasks(self, df):
        tasks = []
        unique_labels = df['label'].unique()
        for label in unique_labels:
            support_samples = df[df['label'] == label].sample(self.num_support, replace=True)
            query_samples = df[df['label'] == label].sample(self.num_query, replace=True)
            task = {
                "support_set": support_samples.to_dict('records'),
                "query_set": query_samples.to_dict('records')
            }
            tasks.append(task)
        return tasks


    def tokenize_and_pad(self, samples):
        encoded_samples = []
        for sample in samples:
            text = sample['text']
            encoding_dict = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_len,
                padding='max_length'
            )
            encoded_sample = {
                'ids': encoding_dict['input_ids'],
                'attention_mask': encoding_dict['attention_mask'],
                'label': int(sample['label'])
            }
            encoded_samples.append(encoded_sample)
        return encoded_samples
    def __getitem__(self, idx):
        task = self.tasks[idx]
        support_set = self.tokenize_and_pad(task['support_set'])
        query_set = self.tokenize_and_pad(task['query_set'])
        return {
            "support_set": support_set,
            "query_set": query_set
        }


def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)

    # Fill NaN values in the 'text' column
    df.text = df.text.fillna('')

    # Convert labels to integers
    df['label'] = df.label.replace({'False': 0, 'True': 1, '0.0': 0, '1.0': 1})

    return df


def main():
    file_path = "C:\\repos\\Thesis\\AHAKLWE\\FSL_model\\250811_training_data.csv"

    # Load and preprocess the data
    df = load_and_preprocess_data(file_path)

    # Create dataset instances
    lines_dataset = LinesDataSet(df)
    fewshot_dataset = FewShotDataSet(df)

    # Create a DataLoader instance
    dataloader = DataLoader(
        lines_dataset,
        sampler=SequentialSampler(lines_dataset),
        batch_size=16
    )


if __name__ == "__main__":
    main()

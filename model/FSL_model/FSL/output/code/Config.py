from transformers import BertTokenizer, DistilBertTokenizer
from torch import nn

TOKENIZER = DistilBertTokenizer.from_pretrained(
        'distilbert-base-uncased',
        do_lower_case = True
        )

MAX_LEN = 128

BATCH_SIZE = 64

EPOCHS = 50

LearningRate= 2e-5

criterion = nn.CrossEntropyLoss()
import argparse
import os
import random

from sklearn.model_selection import train_test_split
import pandas as pd
import mlflow
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score, balanced_accuracy_score, make_scorer
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold

from utils import create_vocabulary, stemming_tokenizer, save_model



def select_first_file(path):
    """Selects first file in folder, use under assumption there is only one file in folder
    Args:
        path (str): path to directory or file to choose
    Returns:
        str: full path of selected file
    """
    files = os.listdir(path)
    return os.path.join(path, files[0])



def main():
    mlflow.sklearn.autolog()
    os.environ['CURL_CA_BUNDLE'] = ''
    """Main function of the script."""

    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, help="path to train data")
    parser.add_argument("--model_dir", type=str, help="path to model output")
    args = parser.parse_args()

    print(args.train_path)

    # paths are mounted as folder, therefore, we are selecting the file from folder
    df = pd.read_csv(args.train_path)

    df.text = df.text.fillna('')
    df['label'] = df.label.replace({'False': 0, 'True': 1, '0.0': 0, '1.0': 1})

    # undersampling majority class to be at max n
    n = int(0.7 * len(df))
    msk = df.groupby('label')['label'].transform('size') >= n
    df = pd.concat((df[msk].groupby('label').sample(n=n), df[~msk]), ignore_index=True)
    print(df.label.value_counts())

    random_state = random.randint(0, 200)
    mlflow.log_metric("num_samples", len(df))
    mlflow.log_metric("random_state_split", random_state)
    mlflow.log_metric("Negative Count", df.label.value_counts()[0])
    mlflow.log_metric("Positive Count", df.label.value_counts()[1])

    df = df.assign(
        stemmed_text = df.text.apply(lambda x: stemming_tokenizer(x, stopword_removal= True))
    )

    df_train, df_test = train_test_split(df, test_size = 0.3, random_state= random_state)
    # # Extracting the label column

    X_train, y_train = df_train.stemmed_text.to_numpy(), df_train.label.to_numpy()
    X_test, y_test = df_test.stemmed_text.to_numpy(), df_test.label.to_numpy()


    vocabulary = create_vocabulary(pd.Series(X_train)) 



    model = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('SVC', SVC()),
    ])

    hyperparam_dict = {
        'SVC__C' : [0.1, 1, 10, 100, 1000],
        'SVC__gamma': [1, 0.1, 0.01, 0.001, 0.0001],
        'SVC__kernel': ['rbf', 'linear'],
        'SVC__probability': [True],


        'tfidf__max_df': (0.25, 0.5, 0.75),
        'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
        'tfidf__max_features': (None, 5000, 10000, 50000),
        'tfidf__use_idf': (True, False),
        'tfidf__norm': ('l1', 'l2', None),
    }

    kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    clf_random =  RandomizedSearchCV(estimator = model, param_distributions = hyperparam_dict, n_iter = 20, verbose=2, cv = kfold, random_state=42, n_jobs = -1, scoring = 'average_precision')
    clf_random.fit(X_train, y_train) 
    
    best_model = clf_random.best_estimator_
    best_score = clf_random.best_score_
    print(best_score)

    ## evaluation: 
    probas = best_model.predict_proba(X_test)[:,1]
    preds = 1*(probas > 0.3)
    print(preds)

    rec = recall_score(y_test, preds)
    prec = precision_score(y_test, preds)
    acc = accuracy_score(y_test, preds)
    print(rec, prec, acc)

    mlflow.log_metric("Recall", rec)
    mlflow.log_metric("Precision", prec)
    mlflow.log_metric("Accuracy", acc)

       # Registering the model to the workspace
    # print("Registering the model via MLFlow")
    # mlflow.sklearn.log_model( sk_model= best_model, artifact_path= "SVC_train", registered_model_name= "SVC_train") 

    # Saving the model to a file
    mlflow.sklearn.save_model(
        sk_model=best_model,
        path=args.model_dir
    )
    
    #save_model(args.model_dir, best_model)


if __name__ == "__main__":
    # Start Logging
    # mlflow.set_experiment('bert_slightly_advancesd')
    # mlflow.start_run()
    main()
    # mlflow.end_run()
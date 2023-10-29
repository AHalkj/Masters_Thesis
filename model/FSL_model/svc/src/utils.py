from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import nltk
from pathlib import Path
import shutil
import mlflow
from nltk.corpus import stopwords
import pandas as pd

nltk.download('punkt')
nltk.download('stopwords')

def stem_words(x: list):
    '''
    Takes a list and returns a list of stemmed words
        Input: x: List of words in sentence
        Output: list of strings, where strings are stemmed words
    '''
    stemmer = nltk.stem.snowball.SnowballStemmer('english')
    if isinstance(x, float):
        return [str(x)]
    else:
        return([stemmer.stem(word) for word in x])


def create_vocabulary(X_train):
    '''
    Takes a list of sentences (training set) and creates a list of unique words in that list
    Input: X_train: list or pd.Series
    Output: list of unique values
    '''
    vocabulary = X_train.apply(
        lambda x: nltk.tokenize.word_tokenize(x) if type(x) == str else x)
    vocabulary = vocabulary.apply(lambda x: stem_words(x)).explode().unique()
    possible_nans = ['na', 'n/a', 'nan']
    vocabulary = pd.DataFrame(
        [word for word in vocabulary if not word in possible_nans])
    return vocabulary.iloc[:, 0].values.tolist()

def stemming_tokenizer(input_str, stopword_removal = False):
    '''
    Takes a sentence and returns a sentence where all the words are stemmed
    '''
    stop_words = set(stopwords.words('english'))
    split_text = nltk.tokenize.word_tokenize(input_str)
    if stopword_removal:
        
        words = [w for w in split_text if not w.lower() in stop_words]

    words = stem_words(split_text)
    
    return ' '.join(words)


svm_best_parameters = {'C': 2, 'break_ties': False, 'cache_size': 200, 'class_weight': 'balanced', 'coef0': 0.0, 'decision_function_shape': 'ovr', 'degree': 3,\
    'gamma': 'scale', 'kernel': 'rbf', 'max_iter': -1, 'probability': True, 'random_state': None, 'shrinking': False, 'tol': 0.01, 'verbose': False}


def save_model(model_dir: str, model) -> None:
    """
    Saves the trained model.
    """
    # input_schema = Schema(
    #     [ColSpec(type=DataType.double, name=f"col_{i}") for i in range(784)],\
    #      ColSpec(type=DataType.double, name=f"col_{i}") for i in range(784)],\
    #      ColSpec(type=DataType.double, name=f"col_{i}") for i in range(784)]])
    # output_schema = Schema([TensorSpec(np.dtype(np.int64), (-1, 512))])
    # signature = mlflow.models.infer_signature()#ModelSignature(inputs=input_schema, outputs=output_schema)
    # print(signature)
    # code_paths = ["train.py", "utils.py"]
    # full_code_paths = [
    #     Path(Path(__file__).parent, code_path) for code_path in code_paths
    # ]

    # shutil.rmtree(model_dir, ignore_errors=True)
    print(f"Saving model to {model_dir}")
    mlflow.sklearn.save_model(
                              model =model,
                              path=model_dir,
                              # code_paths=full_code_paths,
                              # signature=signature
                              )

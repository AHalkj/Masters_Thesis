import nltk
import pandas as pd
import numpy as np
# from googletrans import Translator
# from imblearn.over_sampling import RandomOverSampler
# import fitz
# from fitz import Rect
from ast import literal_eval

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


def stemming_tokenizer(input_str):
    '''
    Takes a sentence and returns a sentence where all the words are stemmed
    '''
    split_text = nltk.tokenize.word_tokenize(input_str)
    words = stem_words(split_text)
    
    return ' '.join(words)

def stem_input(text_vector):
    '''
    Takes list of sentences and stems all of them
    '''
    stemmed_text = text_vector.apply(lambda x: stemming_tokenizer(str(x)))
    return stemmed_text

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

def augment_translate(positives: list, lang: str):
    '''
    Takes list of sentences and backtranslates them to a language. Lanugage must be in the list otherwise error
        Input: positives: list of sentences (usually only positive class so that we can augment that)
               lang:  string (two chars) representing language in google lang shortening
    '''
    if lang not in ['es', 'fr', 'de', 'la', 'ko', 'no', 'la']:
        raise NameError('Language not Available')
    translator = Translator()
    Translation = translator.translate(positives, dest=lang)
    german_text = [t.text for t in Translation]
    BackTranslation = translator.translate(german_text, lang_tgt='en')
    english_text = [t.text for t in BackTranslation]
    return english_text

def augment_df(df, langs: list):
    '''
    Takes dataframe and returns dataframe where positives have been augmented (once for every language in the list)
    Input: df:dataframe. Must contain the columns text and label
           langs: list of two-char-string (google lang abbreviation for languages)
    Ouptut: df with columns text, label, and if it was in df before: report
    '''
    pos = df[df.label == 1].text.tolist()
    if 'report' in df.columns:
        reppos = df[df.label == 1].report.tolist()
        rep = df.report.tolist()
    X = df.text.tolist()
    y = df.label.tolist()
    
    for lang in langs:
        augm = augment_translate(pos, lang)
        X.extend(augm)
        y.extend([1 for t in augm])
        if 'report' in df.columns:
            rep.extend(reppos)
    if 'report' in df.columns:
        df_ret = (pd.DataFrame({'text': X, 'label': y, 'report': rep}))
    else: 
        df_ret = (pd.DataFrame({'text': X, 'label': y})) 

    return df_ret


def oversample_df(df, strat, random_state = 42):
    '''
    Takes a dataframe and upsamples the minority class to the desired strategy (aka. proportion)
    Input: df: dataframe. Must contain columns text and label
           strat: Number > 0, proportion of minority class after upsampling
    Output: Oversampled dataframe
    '''
    oversample = RandomOverSampler(sampling_strategy=strat, random_state=random_state)
    texts_over, label_over = oversample.fit_resample(
        np.array(df.text).reshape(-1, 1), np.array(df.label))
    if 'report' in df.columns:
        report_over, label_over_l = oversample.fit_resample(
            np.array(df.report).reshape(-1, 1), np.array(df.label))

        df = pd.DataFrame({'text': texts_over.squeeze(),
                      'label': label_over.squeeze(),
                       'report': report_over.squeeze()})
    else: 
        df = pd.DataFrame({'text': texts_over.squeeze(),
                      'label': label_over.squeeze()})
    df = df.assign(
        stemmed_text = df.text.apply(lambda x: stemming_tokenizer(x))
    )            
        
    return df

def annotate(pdf_path: str, df: pd.DataFrame) -> str:
    '''
    takes filepath of original pdf and saves annotated pdf with _annotated added to the path
    '''
    file = pdf_path.split('\\')[-1].replace('.pdf', '')
    output_file = file.replace('.pdf', '')
    file_loc = f'./{output_file}_annotated.pdf'
    filepath = pdf_path
    pdfDoc = fitz.open(filepath)
    print('File is being extracted')
    for i, row in df.iterrows():
        if row.label == 1:
            quad = literal_eval(row.quads)
            # if inported dataframe: quad = literal_eval(row.quads) because it's going to be a string
            page_no = row.page_no
            rec = Rect(quad)
            pdfDoc[page_no].add_highlight_annot(rec)

    pdfDoc.save(file_loc)
    pdfDoc.close()
    return file_loc

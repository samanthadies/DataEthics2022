########################################################################################################
# textCleaning.py
# This program cleans the text training data for the scoring criteria by removing
# punctuation, numbers, and extra white spaces, lemmatizing, lowercasing, and converting the score into
# binary.
########################################################################################################


from nltk.stem import WordNetLemmatizer
import pandas as pd
import numpy as np
from tqdm import tqdm
import regex as re
import warnings
stemmer = WordNetLemmatizer()
tqdm.pandas()


OUTPUT_FP = '../data/cleaned/'


def remove_punctuation(row):
    """
    removes all non-alhpabetic, non-space characters from the text input
    :param row: row of df to clean
    :return: cleaned row of df
    """

    # identify characters to keep
    regex = re.compile('[^a-zA-Z ]')
    row = regex.sub('', row)

    # return the row
    return row


def remove_extra_spaces(row):
    """
    removes all extra white spaces beyond the word delimiters
    :param row: row of df to clean
    :return: cleaned row of df
    """

    # identify the whitespace pattern and remove extra white spaces
    whitespace_pattern = r'\s+'
    row = re.sub(whitespace_pattern, ' ', row)
    row = row.strip()

    return row


def lemmatize_text(row):
    """
    lemmatizes the input text
    :param row: row of df to clean
    :return: cleaned row of df
    """

    # lemmatize each word in the row
    row = row.split()
    row = ' '.join([stemmer.lemmatize(word) for word in row])

    return row


def clean_text(row):
    """
    calls other cleaning methods on the inputted row
    :param row: row of df to clean
    :return: cleaned row of df
    """

    # remove punctuation
    row = remove_punctuation(row)

    # lowercase all text
    row = row.lower()

    # remove extra white spaces
    row = remove_extra_spaces(row)

    # lemmatize the text
    row = lemmatize_text(row)

    return row


def convert_to_binary(df):
    """
    converts the scores into binary, where 0 means absent and 1 means present
    :param df: df to convert to binary
    :return: binary df
    """

    criteria = pd.read_csv('../data/raw/cleaning_criteria.csv')
    criteria = criteria.columns

    # convert scores to binary for ml dfs
    for item in criteria:
        df[item] = df[item].astype('int')
        df.loc[df[item] != 0, item] = 1

    return df


def preprocess_text(df):
    """
    cleans each row of data from the datafram and returns new column for the cleaned text
    :param df: dataframe
    :return: column of dataframe with cleaned text
    """

    # clean the text
    df['text'] = df.progress_apply(lambda x: clean_text(x['text']), axis=1, result_type='expand')

    # convert the scores to binary
    df = convert_to_binary(df)

    return df['text']


def preprocess(BASE_FP):
    """
    reads in files and preprocesses data based on df type
    :param BASE_FP: starting file path
    :return: void
    """

    warnings.filterwarnings('ignore')

    ml = ['training', 'training_biased', 'testing']

    # iterate through ml csvs
    for item in ml:
        print('\n' + item + '\n')

        # read each csv
        ML_FP_READ = BASE_FP + item + '_raw.csv'
        df = pd.read_csv(f'{ML_FP_READ}')
        df = df.replace(np.nan, 0, regex=True)

        # clean df
        preprocess_text(df)

        # save dfs as csv
        ML_FP_WRITE = OUTPUT_FP + item + '.csv'
        df.to_csv(f'{ML_FP_WRITE}', index=False)

        ML_FP_WRITE_TEXT = OUTPUT_FP + item + '_text.csv'
        df['text'].to_csv(f'{ML_FP_WRITE_TEXT}', index=False)


def main():
    """

    :return: void
    """

    print("running textCleaning.py main")


if __name__ == '__main__':
    main()

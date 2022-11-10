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

ml = ['training', 'training_biased', 'testing']

criteria = ['data source', 'class collection', 'ground truth size', 'ground truth discussion', 'random sample',
            'dem dist', 'informed consent', 'data public', 'irb', 'limitations', 'anonymity', 'data quality',
            'missing values', 'ethics section', 'people', 'thesis', 'contributions', 'implications', 'motivation',
            'generalizability', 'target pop', 'lit - task', 'lit - context', 'hypotheses', 'data availability',
            'preprocessing', 'var selection', 'var construction', 'var reconstruction', 'list models', 'model steps',
            'model choice', 'define measures', 'common uses', 'cv', 'hyperparameters', 'attempted models',
            'code availability', 'infrastructure/packages', 'biases/fairness', 'eval metrics', 'model comparison',
            'experiment cases', 'fig - explain', 'fig - faithful', 'fig - interpretable', 'takeaways', 'results context',
            'results implications', 'results use', 'results misuse']


########################################################################################################
# remove_punctuation
# Inputs: row
# Return: row
# Description: removes all non-alhpabetic, non-space characters from the text input
########################################################################################################
def remove_punctuation(row):

    # identify characters to keep
    regex = re.compile('[^a-zA-Z ]')
    row = regex.sub('', row)

    # return the row
    return row


########################################################################################################
# remove_extra_spaces
# Inputs: row
# Return: row
# Description: removes all extra white spaces beyond the word delimiters
########################################################################################################
def remove_extra_spaces(row):

    # identify the whitespace pattern and remove extra white spaces
    whitespace_pattern = r'\s+'
    row = re.sub(whitespace_pattern, ' ', row)
    row = row.strip()

    return row


########################################################################################################
# lemmatize_text
# Inputs: row
# Return: row
# Description: lemmatizes the input text
########################################################################################################
def lemmatize_text(row):

    # lemmatize each word in the row
    row = row.split()
    row = ' '.join([stemmer.lemmatize(word) for word in row])

    return row


########################################################################################################
# clean_text
# Inputs: row
# Return: row
# Description: calls other cleaning methods on the inputted row
########################################################################################################
def clean_text(row):

    # remove punctuation
    row = remove_punctuation(row)

    # lowercase all text
    row = row.lower()

    # remove extra white spaces
    row = remove_extra_spaces(row)

    # lemmatize the text
    row = lemmatize_text(row)

    return row


########################################################################################################
# convert_to_binary
# Inputs: df
# Return: df
# Description: converts the scores into binary, where 0 means absent and 1 means present
########################################################################################################
def convert_to_binary(df):

    # convert scores to binary for ml dfs
    for item in criteria:
        df[item] = df[item].astype('int')
        df.loc[df[item] != 0, item] = 1

    return df


########################################################################################################
# preprocess_text
# Inputs: df
# Return: df['Cleaned']
# Description: cleans each row of data from the datafram and returns new column for the cleaned text
########################################################################################################
def preprocess_text(df):

    # clean the text
    df['text'] = df.progress_apply(lambda x: clean_text(x['text']), axis=1, result_type='expand')

    # convert the scores to binary
    df = convert_to_binary(df)

    return df['text']


########################################################################################################
# preprocess
# Inputs: BASE_FP - starting file path, type - 'criteria' or 'classification' to denote df type
# Return: N/A
# Description: reads in files and preprocesses data based on df type
########################################################################################################
def preprocess(BASE_FP):

    warnings.filterwarnings('ignore')

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


########################################################################################################
# main
# Inputs: N/A
# Return: N/A
# Description: N/A
########################################################################################################
def main():

    print("running textCleaning.py main")


if __name__ == '__main__':
    main()

import gensim
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
import pandas as pd
from pprint import pprint
import gensim.corpora as corpora
import warnings
import os


BASE_FP_OUTPUT = '../output/topics/'
BASE_FP = '../data/cleaned/criteria/'

criteria = ['people', 'thesis', 'contributions', 'implications', 'motivation', 'target_pop', 'lit_task', 'lit_context',
            'hypothesis', 'data_avail', 'preprocess', 'var_select', 'var_construct', 'var_reconstruct', 'list_models',
            'model_steps', 'model_choice', 'define_measures', 'common_uses', 'cv', 'hyperparameters',
            'attempted_models', 'code_avail', 'infrastructure', 'fairness', 'metrics', 'comparison', 'cases',
            'fig_explain', 'fig_faith', 'fig_interpret', 'takeaways', 'scope', 'results_context',
            'results_implications', 'results_use', 'results_misuse']


def values():
    idealized_columns = ['data_source_i', 'class_collection_i', 'random_sample_i', 'dem_dist_i', 'informed_consent_i',
                         'data_public_i', 'irb_i', 'ground_truth_size_i', 'ground_truth_discussion_i', 'limitations_i',
                         'preprocess_anonymity_i', 'preprocess_drop_i', 'preprocess_missing_values_i',
                         'preprocess_noise_i',
                         'preprocess_text_i', 'ethics_section_i', 'people_i', 'thesis_i', 'contributions_i',
                         'implications_i', 'motivation_i', 'target_pop_i', 'lit_task_i', 'lit_context_i',
                         'hypothesis_i',
                         'data_avail_i', 'preprocess_i', 'var_select_i', 'var_construct_i', 'var_reconstruct_i',
                         'list_models_i', 'model_steps_i', 'model_choice_i', 'define_measures_i', 'common_uses_i',
                         'cv_i',
                         'hyperparameters_i', 'attempted_models_i', 'code_avail_i', 'infrastructure_i', 'fairness_i',
                         'metrics_i', 'comparison_i', 'cases_i', 'fig_explain_i', 'fig_faith_i', 'fig_interpret_i',
                         'takeaways_i', 'scope_i', 'results_context_i', 'results_implications_i', 'results_use_i',
                         'results_misuse_i', '%_of_idealized']

    ml = ['people', 'thesis', 'contributions', 'implications', 'motivation', 'target pop', 'lit - task', 'lit - context',
          'hypotheses',	'data availability', 'preprocessing', 'var selection', 'var construction', 'var reconstruction',
          'list models', 'model steps', 'model choice', 'define measures', 'common uses', 'cv', 'hyperparameters',
          'attempted models', 'code availability', 'infrastructure/packages', 'biases/fairness', 'eval metrics',
          'model comparison', 'experiment cases', 'fig - explain', 'fig - faithful', 'fig - interpretable', 'takeaways',
          'scope', 'results context', 'results implications', 'results use', 'results misuse']

    df = pd.read_csv('../data/raw/dslc_full.csv')

    df2 = pd.read_csv('../data/raw/training_papers.csv')

    df3 = pd.read_csv('../data/raw/training_sections.csv')
    df3 = df3.fillna(0)


    for item in idealized_columns:
        print(df[item].value_counts())

    print()
    print()

    for item in idealized_columns:
        print(df2[item].value_counts())

    print()
    print()

    for item in ml:
        print(df3[item].value_counts())


def topics():
    WRITE_FP = os.path.join(BASE_FP_OUTPUT, 'summary_output.txt')
    summary_file = open(WRITE_FP, 'w')

    for item in criteria:
        print(item)

        CRITERIA_FP = BASE_FP + item + '/' + item + '_CLEANED.csv'
        df = pd.read_csv(CRITERIA_FP)
        print(df.head())
        print(df.info())

        to_use = df[df['Score'] == 1]

        data = to_use['text'].values.tolist()
        data_words = list(sent_to_words(data))
        # remove stop words
        data_words = remove_stopwords(data_words)
        # print(data_words[:1][0][:30])

        # Create Dictionary
        id2word = corpora.Dictionary(data_words)
        # Create Corpus
        texts = data_words
        # Term Document Frequency
        corpus = [id2word.doc2bow(text) for text in texts]
        # View
        # print(corpus[:1][0][:30])

        # number of topics
        num_topics = 10
        # Build LDA model
        lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                               id2word=id2word,
                                               num_topics=num_topics)
        # Print the Keyword in the 10 topics
        pprint(lda_model.print_topics())
        doc_lda = lda_model[corpus]

        summary_file.write('\n\n' + str(item) + '\n\n')
        summary_file.write(str(lda_model.print_topics()))

    summary_file.close()


def sent_to_words(sentences):
    for sentence in sentences:
        # deacc=True removes punctuations
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))


def remove_stopwords(texts):
    stop_words = stopwords.words('english')
    stop_words.extend(['from', 'subject', 're', 'edu', 'use'])
    return [[word for word in simple_preprocess(str(doc))
             if word not in stop_words] for doc in texts]


def main():

    warnings.filterwarnings('ignore')

    # values()
    # topics()

    a_dict = {'color': 'blue', 'fruit': 'apple', 'pet': 'dog'}
    for key, value in a_dict.items():
        print(key, '->', value)

if __name__ == '__main__':

    main()
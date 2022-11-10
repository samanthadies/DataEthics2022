########################################################################################################
# eda.py
# This program computes basic statistics of a number of attributes from 'thesis_raw.csv'
# and outputs the summary statistics. The program also creats appropriate visualizations of the data.
########################################################################################################


import pandas as pd
import matplotlib.pyplot as plt
import os
import plotly.graph_objects as go
import warnings


BASE_FP = '../data/'
BASE_FP_OUTPUT_I = '../output/eda/idealized/'
BASE_FP_OUTPUT_CS = '../output/eda/cs/'
BASE_FP_OUTPUT_PH = '../output/eda/ph/'
BASE_FP_OUTPUT_RESEARCH = '../output/eda/researchers/'
BASE_FP_OUTPUT_VISUALIZATIONS = '../output/visualizations/'
BASE_FP_OUTPUT_PREDICITONS = '../output/classification/predictions/'


idealized_columns = ['data_source_i', 'class_collection_i', 'random_sample_i', 'dem_dist_i', 'informed_consent_i',
                     'data_public_i', 'irb_i', 'ground_truth_size_i', 'ground_truth_discussion_i', 'limitations_i',
                     'preprocess_anonymity_i', 'preprocess_drop_i', 'preprocess_missing_values_i', 'preprocess_noise_i',
                     'ethics_section_i', 'people_i', 'thesis_i', 'contributions_i', 'implications_i', 'motivation_i',
                     'target_pop_i', 'lit_task_i', 'lit_context_i', 'hypothesis_i', 'data_avail_i', 'preprocess_i',
                     'var_select_i', 'var_construct_i', 'var_reconstruct_i', 'list_models_i', 'model_steps_i',
                     'model_choice_i', 'define_measures_i', 'common_uses_i', 'cv_i', 'hyperparameters_i',
                     'attempted_models_i', 'code_avail_i', 'infrastructure_i', 'fairness_i', 'metrics_i', 'comparison_i',
                     'cases_i', 'fig_explain_i', 'fig_faith_i', 'fig_interpret_i', 'takeaways_i', 'generalizability_i',
                     'results_context_i', 'results_implications_i', 'results_use_i', 'results_misuse_i',
                     '%_of_idealized']

cs_columns = ['data_source_cs', 'class_collection_cs', 'random_sample_cs', 'dem_dist_cs', 'irb_cs',
              'ground_truth_size_cs', 'ground_truth_discussion_cs', 'preprocess_drop_cs', 'limitations_cs', 'people_cs',
              'thesis_cs', 'contributions_cs', 'implications_cs', 'motivation_cs', 'target_pop_cs', 'lit_task_cs',
              'lit_context_cs', 'preprocess_cs', 'var_select_cs', 'var_construct_cs', 'var_reconstruct_cs',
              'list_models_cs', 'model_steps_cs', 'model_choice_cs', 'define_measures_cs', 'cv_cs', 'hyperparameters_cs',
              'attempted_models_i', 'infrastructure_cs', 'metrics_cs', 'comparison_cs', 'cases_cs', 'fig_explain_cs',
              'fig_faith_cs', 'fig_interpret_cs', 'takeaways_cs', 'results_context_cs', 'results_implications_cs',
              'results_use_cs', '%_of_cs']

ph_columns = ['data_source_ph', 'class_collection_ph', 'random_sample_ph', 'dem_dist_ph', 'informed_consent_ph', 'irb_ph',
              'ground_truth_size_ph', 'ground_truth_discussion_ph', 'limitations_ps', 'people_ph', 'thesis_ph',
              'contributions_ph', 'implications_ph', 'motivation_ph', 'target_pop_ph', 'lit_task_ph', 'lit_context_ph',
              'hypothesis_ph', 'var_construct_ph', 'list_models_ph', 'model_steps_ph', 'model_choice_ph',
              'define_measures_ph', 'common_uses_ph', 'infrastructure_ph', 'metrics_ph', 'fig_explain_ph', 'fig_faith_ph',
              'fig_interpret_ph', 'takeaways_ph', 'generalizability_ph', 'results_context_ph', 'results_implications_ph',
              'results_use_ph', '%_of_ph']

researcher_columns = ['Num_Researchers', 'Choudhury', 'Drezde', 'Coppersmith', 'data_source_i', 'class_collection_i',
                      'random_sample_i', 'dem_dist_i', 'informed_consent_i', 'data_public_i', 'irb_i',
                      'ground_truth_size_i', 'ground_truth_discussion_i', 'limitations_i', 'preprocess_anonymity_i',
                      'preprocess_drop_i', 'preprocess_missing_values_i', 'preprocess_noise_i', 'ethics_section_i',
                      'people_i', 'thesis_i', 'contributions_i', 'implications_i', 'motivation_i', 'target_pop_i',
                      'lit_task_i', 'lit_context_i', 'hypothesis_i', 'data_avail_i', 'preprocess_i', 'var_select_i',
                      'var_construct_i', 'var_reconstruct_i', 'list_models_i', 'model_steps_i', 'model_choice_i',
                      'define_measures_i', 'common_uses_i', 'cv_i', 'hyperparameters_i', 'attempted_models_i',
                      'code_avail_i', 'infrastructure_i', 'fairness_i', 'metrics_i', 'comparison_i', 'cases_i',
                      'fig_explain_i', 'fig_faith_i', 'fig_interpret_i', 'takeaways_i', 'generalizability_i',
                      'results_context_i', 'results_implications_i', 'results_use_i', 'results_misuse_i',
                      '%_of_idealized']


########################################################################################################
# read
# Inputs: filename
# Return: idealized - dataframe 1, cs - dataframe 2, ph - dataframe 3, researchers - dataframe 4
# Description: reads in provided file, sets maximum score to 100, splits dataframe into four sub-dataframes
# containing scores for idealized, cs, ph and researcher norms, and returns the sub-dataframes
########################################################################################################
def read(filename):

    df = pd.read_csv(filename)
    df = df.dropna(subset=idealized_columns)

    for index, value in df['%_of_idealized'].items():
        if value > 100:
            df['%_of_idealized'][index] = 100

    # create idealized df
    idealized = df[idealized_columns].copy()

    # create cs df
    cs = df[cs_columns].copy()

    # create ph df
    ph = df[ph_columns].copy()

    # create reserachers df
    researchers = df[researcher_columns].copy()

    return idealized, cs, ph, researchers


########################################################################################################
# generate_summary_stats
# Inputs: idealized - dataframe 1, cs - dataframe 2, ph - dataframe 3, researchers - dataframe 4
# Return: N/A
# Description: calculates summary statistics, and writes statistics to output csv's
########################################################################################################
def generate_summary_stats(idealized, cs, ph, researchers):

    # generate summary stats
    idealized_stats = summary_stats_by_df(idealized_columns, idealized)
    cs_stats = summary_stats_by_df(cs_columns, cs)
    ph_stats = summary_stats_by_df(ph_columns, ph)
    researcher_stats = summary_stats_by_df(researcher_columns, researchers)

    # output summary stats to csv's
    idealized_stats.to_csv(f'{BASE_FP_OUTPUT_I}idealized_summary_stats.csv', index=False)
    cs_stats.to_csv(f'{BASE_FP_OUTPUT_CS}cs_summary_stats.csv', index=False)
    ph_stats.to_csv(f'{BASE_FP_OUTPUT_PH}ph_summary_stats.csv', index=False)
    researcher_stats.to_csv(f'{BASE_FP_OUTPUT_RESEARCH}researcher_summary_stats.csv', index=False)


########################################################################################################
# summary_stats_by_df
# Inputs: df - dataframe, columns - columns to get summary stats from
# Return: dataframe with summary statistics
# Description: calculates summary statistics about df's columns using pd.describe()
########################################################################################################
def summary_stats_by_df(columns, df):

    # create and format dataframe with first column
    first_attribute = columns[0]
    stats_df_overall = pd.DataFrame({first_attribute: df[first_attribute].describe()})
    stats_df_overall.reset_index(level=0, inplace=True)
    stats_df_overall.rename(columns={'index': 'summary_stat_type', first_attribute: first_attribute}, inplace=True)

    # generage summary stats of all other columns and merge into one dataframe
    for attribute in columns:
        if attribute != first_attribute:
            stats_df = pd.DataFrame({attribute: df[attribute].describe()})
            stats_df.reset_index(level=0, inplace=True)
            stats_df.rename(columns={'index': 'summary_stat_type', attribute: attribute}, inplace=True)
            stats_df_overall = stats_df_overall.merge(stats_df, on='summary_stat_type')

    return stats_df_overall


########################################################################################################
# make_visualizations
# Inputs: idealized - dataframe 1, cs - dataframe 2, ph - dataframe 3, researchers - dataframe 4
# Return: N/A
# Description: generates visualizations to aid with exploratory data analysis
########################################################################################################
def make_visualizations(idealized, cs, ph, researchers):

    # plot appropriate histograms
    plot_histograms_overall(idealized, cs, ph, researchers)

    # plot appropriate boxplots
    plot_boxplots(idealized, cs, ph)

    # find researcher stats + plots
    researcher_plots(researchers)

    # make heatmap
    create_heatmap(idealized, '%_of_idealized', f'{BASE_FP_OUTPUT_VISUALIZATIONS}')


########################################################################################################
# plot_histograms_overall
# Inputs: idealized - dataframe 1, cs - dataframe 2, ph - dataframe 3, researchers - dataframe 4
# Return: N/A
# Description: creates histograms of each of the columns of interest, plus overall hist plots for each
# df
########################################################################################################
def plot_histograms_overall(idealized, cs, ph, researchers):

    # create idealized histograms
    idealized.hist()
    file_name = 'histograms_i.png'
    plt.savefig(os.path.join(BASE_FP_OUTPUT_I, file_name))
    plt.clf()

    for column in idealized_columns:
        plot_histograms(idealized, column, BASE_FP_OUTPUT_I, "ideal")

    # create cs histograms
    cs.hist()
    file_name = 'histograms_cs.png'
    plt.savefig(os.path.join(BASE_FP_OUTPUT_CS, file_name))
    plt.clf()

    for column in cs_columns:
        plot_histograms(cs, column, BASE_FP_OUTPUT_CS, "cs")

    # create ph histograms
    ph.hist()
    file_name = 'histograms_ph.png'
    plt.savefig(os.path.join(BASE_FP_OUTPUT_PH, file_name))
    plt.clf()

    for column in ph_columns:
        plot_histograms(ph, column, BASE_FP_OUTPUT_PH, "ph")

    # create researchers histograms
    ph.hist()
    file_name = 'histograms_researchers.png'
    plt.savefig(os.path.join(BASE_FP_OUTPUT_RESEARCH, file_name))
    plt.clf()

    for column in researcher_columns:
        plot_histograms(researchers, column, BASE_FP_OUTPUT_RESEARCH, "research")


########################################################################################################
# plot_histograms
# Inputs: df - dataframe, column - column to plot distribution of with histogram
# Return: N/A
# Description: plots histogram of specified dataframe column and saves plot
########################################################################################################
def plot_histograms(df, column, FILE_PATH, type):

    # change the color of the histograms
    if type == "cs":
        color = '#7EA6E0'
    elif type == "ph":
        color = '#EA6B66'
    else:
        color = '#B266FF'

    # create and label the histograms
    plt.hist(df[column], color=color, ec=color)
    title_label = "Distribution of  " + str(column)
    plt.title(title_label)
    plt.xlabel(column)
    plt.ylabel("Frequency")
    if (column == "%_of_cs") or (column == "%_of_ph") or (column == "%_of_idealized") or (column == "predicted_score"):
        plt.xlim([30,100])

    # save figure
    file_name = str(column) + '_histogram.png'
    plt.savefig(os.path.join(FILE_PATH, file_name))

    # clear plot
    plt.clf()


########################################################################################################
# plot_boxplots
# Inputs: idealized - dataframe 1, cs - dataframe 2, ph - dataframe 3
# Return: N/A
# Description: plots boxplots of overall scores and saves plots
########################################################################################################
def plot_boxplots(idealized, cs, ph):

    # create idealized boxplots
    idealized.boxplot(column=['%_of_idealized'])
    file_name = 'boxplot_i.png'
    plt.savefig(os.path.join(BASE_FP_OUTPUT_I, file_name))
    plt.clf()

    # create cs boxplots
    cs.boxplot(column=['%_of_cs'])
    file_name = 'boxplot_cs.png'
    plt.savefig(os.path.join(BASE_FP_OUTPUT_CS, file_name))
    plt.clf()

    # create ph boxplots
    ph.boxplot(column=['%_of_ph'])
    file_name = 'boxplot_ph.png'
    plt.savefig(os.path.join(BASE_FP_OUTPUT_PH, file_name))
    plt.clf()


########################################################################################################
# researcher_plots
# Inputs: researchers - dataframe
# Return: N/A
# Description: plots figures and descriptive statistics for the three main researchers
########################################################################################################
def researcher_plots(researchers):

    # calculate summary stats for De Choudhury, Drezde, Coppersmith, and the remaining authors
    researchers_dict = {'Choudhury': [], 'Drezde': [], 'Coppersmith': [], 'Other': []}

    # make De Choudhury df
    df_choudhury = researchers[researchers['Choudhury'] == 1]
    for item in df_choudhury['%_of_idealized']:
        researchers_dict['Choudhury'].append(item)

    # make Drezde df
    df_drezde = researchers[researchers['Drezde'] == 1]
    for item in df_drezde['%_of_idealized']:
        researchers_dict['Drezde'].append(item)

    # make Coppersmith df
    df_coppersmith = researchers[researchers['Coppersmith'] == 1]
    for item in df_coppersmith['%_of_idealized']:
        researchers_dict['Coppersmith'].append(item)

    # make df of remaining authors
    df_other_dc = researchers[researchers['Choudhury'] == 0]
    df_other_ddc = df_other_dc[df_other_dc['Drezde'] == 0]
    df_other = df_other_ddc[df_other_ddc['Coppersmith'] == 0]
    for item in df_other['%_of_idealized']:
        researchers_dict['Other'].append(item)

    # calculate and save summary stats about top researchers
    fp = BASE_FP_OUTPUT_RESEARCH + 'top_researchers.csv'
    to_write = open(fp, 'w')

    to_write.write('De Choudhury stats:\n\n')
    to_write.write(str(df_choudhury['%_of_idealized'].describe()) + '\n\n\n')
    to_write.write('Drezde stats:\n\n')
    to_write.write(str(df_drezde['%_of_idealized'].describe()) + '\n\n\n')
    to_write.write('Coppersmith stats:\n\n')
    to_write.write(str(df_coppersmith['%_of_idealized'].describe()) + '\n\n\n')
    to_write.write('Other author stats:\n\n')
    to_write.write(str(df_other['%_of_idealized'].describe()) + '\n\n\n')

    # create comparative boxplots
    fig, ax = plt.subplots()
    ax.boxplot(researchers_dict.values())
    ax.set_xticklabels(researchers_dict.keys())
    ax.set_ylabel('Idealized Score (%)')
    ax.set_xlabel('Researcher')

    plt.savefig(os.path.join(BASE_FP_OUTPUT_RESEARCH, 'comp_boxplots_researchers.png'))
    plt.clf()

    # make overlapping histograms for De Choudhury, Drezde, and Coppersmith
    plt.hist(df_choudhury['%_of_idealized'], label="Choudhury")
    plt.hist(df_drezde['%_of_idealized'], label="Drezde")
    plt.hist(df_coppersmith['%_of_idealized'], label="Coppersmith")

    plt.xlabel("Scores")
    plt.ylabel("Frequency")
    plt.title("Distribution of Scores")
    plt.legend(loc='upper right')

    plt.savefig(os.path.join(BASE_FP_OUTPUT_RESEARCH, 'comp_hists_researchers.png'))
    plt.clf()


########################################################################################################
# create_heatmap
# Inputs: idealized
# Return: N/A
# Description: Create heatmap to display the paper score breakdown, where the papers are ordered by
# paper score and the columns are sorted by average score
########################################################################################################
def create_heatmap(idealized, column, fp):

    temp_df = idealized.sort_values(by=column)
    df = temp_df.drop([column], axis=1)
    df = df.reindex(df.mean().sort_values(ascending=False).index, axis=1)
    df_list = df.values.tolist()

    fig = go.Figure(data=go.Heatmap(z=df_list, x=df.columns, colorscale='purples'))
    fig.update_layout(
        margin=dict(l=20, r=100, t=20, b=20),
        xaxis_nticks=len(df.columns)
    )

    fig.write_image(os.path.join(fp, 'heatmap.png'))


########################################################################################################
# perform_eda
# Inputs: filename
# Return: N/A
# Description: reads in file, generates summary statistics, and creates figures for exploratory data
# analysis
########################################################################################################
def perform_eda(filename):

    warnings.filterwarnings('ignore')

    # read data and create four sub-dataframes for the different definitions/analyses
    idealized, cs, ph, researchers = read(filename)

    # generate summary statistics and basic plots
    generate_summary_stats(idealized, cs, ph, researchers)

    # generates visualizations to aid with exploratory data analysis
    make_visualizations(idealized, cs, ph, researchers)


def analyze_predictions(filename):

    warnings.filterwarnings('ignore')

    predictions = pd.read_csv(filename)

    stats = summary_stats_by_df(predictions.columns, predictions)
    stats.to_csv(f'{BASE_FP_OUTPUT_PREDICITONS}summary_stats.csv', index=False)

    create_heatmap(predictions, 'predicted_score', f'{BASE_FP_OUTPUT_PREDICITONS}')

    plot_histograms(predictions, 'predicted_score', f'{BASE_FP_OUTPUT_PREDICITONS}', type)


########################################################################################################
# main
# Inputs: N/A
# Return: N/A
# Description:
########################################################################################################
def main():

    print('running eda.py main')


if __name__ == '__main__':

    main()

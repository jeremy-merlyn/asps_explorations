import pandas as pd
import models
import sklearn
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support as score
import seaborn as sns
import numpy as np
import time
import os
from agents_and_prompts import *

mistral_7b = models.MistralModel("mistral_7b")

model = mistral_7b

INPUT_CSV = 'data/asps_router_eval.csv'
# INPUT_CSV = 'data/demo_examples.csv'
# INPUT_CSV = 'data/scratch_input.csv'


# OUTPUT_CSV = 'data/scratch_output.csv'



df = pd.read_csv(INPUT_CSV)

print(len(df), 'rows in input csv')

full_categories_dict = {
    '1': 'wiki',
    '2': 'bing',
    '3': 'gemini',
    '4': 'none'
}  



# filter out any exx that are missing a label, or use a label that is not in the categories_dict
df = df[df['route'].isin(full_categories_dict.values())]



def construct_categories_dict(agents):
    categories = {}
    for i,agent in enumerate(agents, start=1):
        categories[str(i)] = agent
    return categories

def construct_prompt(agents):
    prompt = PROMPT_INTRO
    num_agents = str(len(agents) + 1)
    DEFAULT_DESCRIPTION = '''\n\n%s. Use this category only if the user query does not fit in any of the categories above.''' % num_agents
    for i,agent in enumerate(agents, start=1):
        prompt += "\n\n" + str(i) + ". " + AGENT_DESRCIPTIONS_DICT[agent]
    prompt += DEFAULT_DESCRIPTION
    return prompt





def run_case(df, case_number):

    SELECTED_AGENTS = AGENT_COMBO_CASES[case_number]

    selected_categories_dict = construct_categories_dict(SELECTED_AGENTS)
    selected_categories_dict[str(len(selected_categories_dict)+1)] = 'none'

    print("Categories and their numbers:", selected_categories_dict)

    CONSTRUCTED_PROMPT = construct_prompt(SELECTED_AGENTS)

    print(CONSTRUCTED_PROMPT)

    OUTPUT_CSV = 'data/combinations/case_%s.csv' % case_number

    start = time.time()

    for i, row in df.iterrows():
        print(row['query'])
        model_response = model.get_response_tgi(CONSTRUCTED_PROMPT, row['query'])
        print(model_response)
        df.at[i, model.name+'_response'] = model_response
        predicted_category = selected_categories_dict.get(model_response, "none")
        df.at[i, model.name+'_category'] = predicted_category

    end = time.time()

    time_per_query = (end-start)/len(df)
    print("Mean time per query: {:.4f}".format(time_per_query))



    # add column for correctness
    df['correctness'] = df[model.name+'_category'] == df['route']

    # add defaulted category, assumed gemini
    df['defaulted_category'] = df[model.name+'_category']
    df.loc[df['defaulted_category'] == 'none', 'defaulted_category'] = 'gemini'
    df['defaulted_correctness'] = df['defaulted_category'] == df['route']

    accuracy = df['correctness'].mean()
    accuracy_after_default = df['defaulted_correctness'].mean()

    #format accuracy to 4 decimal places
    accuracy = "{:.4f}".format(accuracy)
    accuracy_after_default = "{:.4f}".format(accuracy_after_default)

    print('accuracy: {}'.format(accuracy))
    print('accuracy_after_default: {}'.format(accuracy_after_default))



    df.to_csv(OUTPUT_CSV,index=False)
    print("output printed to %s" % OUTPUT_CSV)


    actual, predicted, predicted_after_default = df['route'], df[model.name+'_category'], df['defaulted_category']

    precision, recall, fscore, support = score(actual, predicted)

    precision_w_default, recall_w_default, fscore_w_default, support_w_default = score(actual, predicted_after_default)


    # print('precision: {}'.format(precision))
    # print('recall: {}'.format(recall))
    # print('fscore: {}'.format(fscore))
    # print('support: {}'.format(support))

    # df = pd.read_csv('may_27_mistral_7b_evalset_responses.csv')
    # actual, predicted = df['route'], df[model.name+'_category']


    # sort by correctness, then by "route", then by prediction
    df = df.sort_values(by=['correctness', 'route', model.name+'_category'])

    OUTPUT_CSV_WITH_STATS = 'data/combinations/stats_case_%s.csv' % case_number

    df.to_csv(OUTPUT_CSV_WITH_STATS,index=False)

    # add prompt line comment to csv
    with open(OUTPUT_CSV_WITH_STATS, 'r') as f:
        lines = f.readlines()
        lines.insert(0, '# Combination of agents: %s\n\n' % SELECTED_AGENTS)
        lines.insert(1, '# Prompt: %s\n\n' % CONSTRUCTED_PROMPT)
    # add accuracy, precision,recall,fscore,support to csv, format them to 4 decimal places
        lines.insert(2, '# accuracy: {}\n'.format(accuracy))
        lines.insert(3, '# precision: {}\n'.format(np.round(precision, 4)))
        lines.insert(4, '# recall: {}\n'.format(np.round(recall, 4)))
        lines.insert(5, '# fscore: {}\n'.format(np.round(fscore, 4)))
        lines.insert(6, '# support: {}\n\n'.format(support))
        lines.insert(7, '# accuracy_after_default: {}\n'.format(accuracy_after_default))
        lines.insert(8, '# precision_after_default: {}\n'.format(np.round(precision_w_default, 4)))
        lines.insert(9, '# recall_after_default: {}\n'.format(np.round(recall_w_default, 4)))
        lines.insert(10, '# fscore_after_default: {}\n'.format(np.round(fscore_w_default, 4)))
        lines.insert(11, '# support_after_default: {}\n\n'.format(support_w_default))
    #add time per query to csv
        lines.insert(12, '# time per query: {:.4f}\n\n'.format(time_per_query))    


    with open(OUTPUT_CSV_WITH_STATS, 'w') as f:
        f.writelines(lines)

    print("output with stats printed to %s" % OUTPUT_CSV_WITH_STATS)



    print("output with stats printed to %s" % OUTPUT_CSV_WITH_STATS)



    try:
        os.makedirs('data/confusion_matrices')
    except FileExistsError:
        pass

    # confusion matrix before defaults    
    confusionmatrix = metrics.confusion_matrix(actual, predicted, labels=list(full_categories_dict.values()))
    disp = metrics.ConfusionMatrixDisplay(confusionmatrix, display_labels=list(full_categories_dict.values()))
    disp.plot()
    plt.savefig('data/confusion_matrices/case_%s.png' % case_number)

    # confusion matrix after defaults
    confusionmatrix_after_default = metrics.confusion_matrix(actual, predicted_after_default, labels=list(full_categories_dict.values()))
    disp_after_default = metrics.ConfusionMatrixDisplay(confusionmatrix_after_default, display_labels=list(full_categories_dict.values()))
    disp_after_default.plot()
    plt.savefig('data/confusion_matrices/case_%s_after_default.png' % case_number)


run_case(df, 1)
run_case(df, 2)

# for n in range(1,13):
#     run_case(df, n)

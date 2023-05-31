import pandas as pd
import csv
import numpy as np
import random
import yaml
import os
from data import preprocess
import string

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Random seed
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def set_seed(seed: int) -> None:
    """Sets various random seeds. """
    random.seed(seed)
    np.random.seed(seed)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# File writing
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def make_folders(filename: str) -> None:
    """If the containing directory does not exist, create it."""
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        print("created directory: ", os.path.dirname(filename))

def write_output_to_csv(
    experiment_name: str,
    model_name: str,
    data_split: str,    
    examples: list,
    predictions: list, 
    task:str
    ) -> None:
    """Write examples and their model predictions to output file.

    Given an experiment_name 'D2', a model_name such as 'deepmoji_linear', and data_split 'dev_output.csv', create the file:
        outputs/D2/deepmoji_linear/dev_output.csv

    Args:
        experiment_name: specified by the main setup yml file

        model_name: specified by the model's config yml file

        data_split: a string representing either train, dev or test

        examples: a list of the examples of same size as predictions

        predictions: a list of model predictions

        task: whether this is primary or adaptation task
    """
    
    filename = "outputs/{0}/{1}/{2}/{3}_output.csv".format(
        experiment_name, task, data_split, model_name, 
    )
    make_folders(filename)

    if(data_split == "adaptation_test"):
        first_sentences, second_sentences = zip(*examples)
        df = pd.DataFrame(
            data=zip(first_sentences, second_sentences, predictions),
            columns=["Sentence_1", "Sentence_2", "Prediction"]
        )
    else:
        df = pd.DataFrame(data=zip(examples, predictions), columns=['Example', 'Prediction'])
    df.to_csv(filename)


'''
def write_adaptation_output_to_csv(
    experiment_name: str,
    model_name: str,
    examples: list,
    predictions: list,
) -> None:
    """Write examples and their model predictions to output file.

    Given an experiment_name 'D2' and a model_name such as 'deepmoji_linear' create the file:
        outputs/D2/deepmoji_linear/adaptation_output.csv

    Args:
        experiment_name: specified by the main setup yml file

        model_name: specified by the model's config yml file

        examples: a list of the examples of same size as predictions. For the adaptation task, each example is a pair of sentences.

        predictions: a list of model predictions
    """
    filename = "outputs/{0}/adaptation/{1}_output.csv".format(
        experiment_name, model_name)
    make_folders(filename)

    # Data was shuffled so first not the tweet in general
    first_sentences, second_sentences = zip(*examples)
    df = pd.DataFrame(
        data=zip(first_sentences, second_sentences, predictions),
        columns=["Sentence_1", "Sentence_2", "Prediction"]
    )
    df.to_csv(filename)
'''
def write_results_to_out(results: list, filename: str) -> None:
    """
    Write F1 score of each model to results file.
    # TODO: make scores.out just a csv file, and output confusion matrices elsewhere
  
    Args:
        results: list of results data for each model
        filename: name of output file
    """ 
    make_folders(filename)
    output = '\n\n'.join(results) + '\n'

    with open(filename, 'w') as f:
        f.write(output)

def write_error_analysis_to_files(
    experiment_name: str,
    model_name: str,
    data_split: str,    
    examples: list,
    gold_labels: list,
    predictions: list,
    task
    ) -> None:

    match_filename = "outputs/{0}/{1}/{2}/error_analysis/{3}_correct_pred.csv".format(
        experiment_name, task, data_split, model_name
    )
    mismatch_filename = "outputs/{0}/{1}/{2}/error_analysis/{3}_incorrect_pred.csv".format(
        experiment_name, task, data_split, model_name
    )
    make_folders(match_filename)
    make_folders(mismatch_filename)

    if data_split == 'adaptation_test':
        # create matching prediction and gold labels df
        match_df = pd.DataFrame(columns = ['Sarcastic Example', 'Non-sarcastic rephrase'])
        # create mis-matching prediction and gold labels df
        mismatch_df = pd.DataFrame(columns = ['Sarcastic Example', 'Non-sarcastic rephrase'])

    else:
        # create matching prediction and gold labels df
        match_df = pd.DataFrame(columns = ['Example', 'Gold Label', 'Prediction'])
        # create mis-matching prediction and gold labels df
        mismatch_df = pd.DataFrame(columns = ['Example', 'Gold Label', 'Prediction'])
    
    for example, gold_label, prediction in zip(examples, gold_labels, predictions):
        if data_split == 'adaptation_test':
            example, gold_label, prediction = shuffled_to_original(example, gold_label, prediction)
            
            if gold_label == prediction:
                match_df = match_df.append({'Sarcastic Example': example[0], 'Non-sarcastic rephrase': example[1]}, ignore_index = True)

            else:
                mismatch_df = mismatch_df.append({'Sarcastic Example': example[0], 'Non-sarcastic rephrase': example[1]}, ignore_index = True)

        else:
            # write matching predictions and gold labels to df
            if gold_label == prediction:
                match_df = match_df.append({'Example': example, 'Gold Label': gold_label, 'Prediction': prediction}, ignore_index = True)
            # write mismatching predictions and gold labels to df
            else:
                mismatch_df = mismatch_df.append({'Example': example, 'Gold Label': gold_label, 'Prediction': prediction}, ignore_index = True)

    match_df.to_csv(match_filename)
    mismatch_df.to_csv(mismatch_filename)

    if data_split == 'adaptation_test':
        correct_avg_sarc_len = round((match_df['Sarcastic Example'].apply(len).sum()) / len(match_df['Sarcastic Example']), 2)
        correct_avg_rephrase_len = round((match_df['Non-sarcastic rephrase'].apply(len).sum()) / len(match_df['Non-sarcastic rephrase']), 2)

        incorrect_avg_sarc_len = round((mismatch_df['Sarcastic Example'].apply(len).sum()) / len(mismatch_df['Sarcastic Example']), 2)
        incorrect_avg_rephrase_len = round((mismatch_df['Non-sarcastic rephrase'].apply(len).sum()) / len(mismatch_df['Non-sarcastic rephrase']), 2)

        return [model_name, correct_avg_sarc_len, correct_avg_rephrase_len, incorrect_avg_sarc_len, incorrect_avg_rephrase_len]

    else:
        return


def shuffled_to_original(example: tuple, gold_label: int, prediction: int)->tuple:
    if gold_label:
        gold_label = 0
        prediction = int(not prediction)
        example = (example[1], example[0])
    
    return example, gold_label, prediction

def write_avg_len_to_out(results: list, experiment_name: str, task: str)-> None:
    output_filename = "outputs/{0}/{1}/adaptation_test/error_analysis/0_avg_str_len_info.txt".format(
        experiment_name, task)

    with open(output_filename, 'w') as f:
        for result in results:
            model_name, correct_avg_sarc_len, correct_avg_rephrase_len, incorrect_avg_sarc_len, incorrect_avg_rephrase_len = result[0], result[1], result[2], result[3], result[4]
            f.write("{0}\n\tCorrect Predictions\n\t\tAvg str length of sarc twts: {1}\n\t\tAvg str length of nonsarc rephrases: {2}".format(
                model_name, correct_avg_sarc_len, correct_avg_rephrase_len))

            f.write("\n\tIncorrect Predictions\n\t\tAvg str length of sarc twts: {0}\n\t\tAvg str length of nonsarc rephrases: {1}".format(
                incorrect_avg_sarc_len, incorrect_avg_rephrase_len))

            f.write('\n***************************************************************************\n')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# File reading
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def load_data(filename, task="primary") -> list:
    """
    Load the train and test files for model consumption.

    If primary task, return a list of str, int pairs: 
        - [ (tweet, label), ... ]

    If adaptation task, return a list of tuples of form ((str, str), int)
        - [ ((tweet, rephrase), index), ... ]

    Args:
        filename: a string representing the csv file to load the data from.
        task: a string, either 'primary' or 'secondary'.

    Returns:
        A list of example, int pairs where each pair is a single instance and its label.
    """
    data = []
    task_example = None
    if task == "primary":
        task_example = primary_task_example
    elif task == "adaptation":
        task_example = secondary_task_example
    else:
        raise ValueError("The task argument must be 'primary' or 'adaptation', but received '{}'".format(task))

    with open(filename, encoding='utf-8') as csvfile:
        file_reader = csv.DictReader(csvfile, quotechar='\"')
        data = [task_example(row) for row in file_reader]

    return data

def load_yaml(filename: str)->dict:
    """
    Load file containing hyperparameters.

    Args:
        filename: name of yaml file
    
    Returns: 
        a dictionary containing the data
    """
    with open(filename, 'r') as f:
        data = yaml.safe_load(f)
    return data

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Utility functions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def primary_task_example(row: pd.Series) -> tuple:
    """Take a row of the training or test dataframe, preprocess as appropriate, and return the tuple representing an example for primary task, of the form 
        - (tweet, label)

    Args: 
        row: a pd.Series representing a single row of the training or test data loaded from a CSV.
    Returns:
        example: a tuple representing a single, labeled example for training or prediction. 
    """
    if "tweet" in row:
        return (preprocess(row["tweet"], 'primary'), int(row["sarcastic"]))
    elif "text" in row:
        return (((preprocess(row["text"], 'primary')), int(row["sarcastic"])))

def secondary_task_example(row: pd.Series) -> tuple:
    """Take a row of the training dataframe, preprocess as appropriate, and return the tuple representing an example for adaptation task, of the form 
        - ((tweet, rephrase), label)

    NB: There are only 867 examples total for the adaptation task, because there are only rephrases for the 867 positives in the training dataset. The adaptation dataset is a csv file with only two columns.

    Args:
        row: a row of the pd.DataFrame for the adaptation data. 
    Returns:
        an example of shape ((tweet, rephrase), index)
    """
    example = ((preprocess(row["tweet"]), preprocess(row["rephrase"])), 0)
    return shuffle_tweet_rephrase(example)


def shuffle_tweet_rephrase(example: tuple) -> tuple:
    """Swap the order of sarcastic tweet and its rephrase with probability 0.5.

    Args: 
        example: a tuple of form (tweet, rephrase, index) representing a single adaptation task example.
        
    Returns: 
        a tuple of the same shape, shuffled in order with index updated.
    """
    index = np.random.choice([0,1], p=[0.5, 0.5])
    if index:
        pair, label = example
        pair_ = pair[1], pair[0]
        example = (pair_, index)
    return example

def shuffle_split_data(train: list, test: list = []) -> tuple:
    """Shuffles and splits the data into train, dev, and test."""
    random.shuffle(test)
    train, dev = train_dev_split(train)
    return train, dev, test

def train_dev_split(train: list, dev_fraction: float=0.1) -> tuple:
    # indices 0 - 866 are positives
    # 867 - 3467 are negatives
    all_positives = train[:866]
    all_negatives = train[866:]
    percent_pos = len(all_positives) / len(train) # ~ 0.25
    
    dev_size = int(len(train) * dev_fraction)
    train_size = len(train) - dev_size

    # sample train
    num_train_positives = int(train_size * percent_pos)
    num_train_negatives = train_size - num_train_positives
    train_positives = random.sample(all_positives, num_train_positives)
    train_negatives = random.sample(all_negatives, num_train_negatives)
    train = train_positives + train_negatives    
    random.shuffle(train)    

    # sample dev
    num_dev_positives = int(dev_size * percent_pos)
    num_dev_negatives = dev_size - num_dev_positives
    dev_positives = random.sample(all_positives, num_dev_positives)
    dev_negatives = random.sample(all_negatives, num_dev_negatives)
    dev = dev_positives + dev_negatives
    random.shuffle(dev)

    return train, dev

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# String formatting
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def format_confusion_matrix(cm, task="primary") -> str:
    """
    If Primary Task:

    Columns = Predicted
    Rows = True Labels
                        Not Sarcastic   Sarcastic

        Not Sarcastic                   
        Sarcastic    

    If Adaptation Task:
    
    # TODO: This doesn't seem like the best way to view the results.

    Columns = Predicted
    Rows = True Labels
                        Sentence 1   Sentence 2

        Sentence 1                   
        Sentence 2    
    
    """     
    print("task: {}".format(task), "confusion matrix: ", cm)
    cm_string = """
    Columns = Predicted
    Rows = True Labels

                Not Sarcastic   Sarcastic
    
    Not Sarcastic       {0}     {1}
    Sarcastic           {2}      {3}
    """.format(*cm.ravel())
    if task == "primary":
        return cm_string    
    elif task == "adaptation":
        return cm_string.replace("Not Sarcastic", "Sentence 1\t\t").replace("Sarcastic", "Sentence 2")
    else:
        raise ValueError("The argument `task' must be 'primary' or 'adaptation' ")


def format_model_result(
    model_name: str, 
    train_results: tuple, 
    dev_results: tuple, 
    test_results: tuple,
    task="primary",
) -> str:
    """Formats the model results data as a str."""
    if train_results:
        train_score, train_cm = train_results
        train_cm_string = format_confusion_matrix(train_cm)

    if dev_results:
        dev_score, dev_cm = dev_results
        dev_cm_string = format_confusion_matrix(dev_cm)

    if test_results:
        test_score, test_cm = test_results
        test_cm_string = format_confusion_matrix(test_cm, task=task)

    results_string = model_name + "\n\t" + task.upper() + "\n\t"

    score_string = "F1"

    if(task=="adaptation"):
        score_string = "Accuracy"

    if(train_results):
        results_string += """TRAIN
    **************************
    {0} Score: {1}
    Confusion Matrix: 
    {2}

    """.format(
        score_string, train_score, train_cm_string, 
        )

    #If we're creating the dev results
    if(dev_results):
        results_string +=     """DEV
    **************************
    {0} Score: {1}
    Confusion Matrix: 
    {2}
        
    """.format(score_string, dev_score, dev_cm_string)

    if(test_results):
        results_string += """TEST
    **************************    
    {0} Score: {1}
    Confusion Matrix: 
    {2}
    
    """.format(score_string, test_score, test_cm_string)

    results_string += """ 

    **************************
    **************************

    """

    return results_string
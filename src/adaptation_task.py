import sys
import yaml
from util import *
from model import *
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from primary_task import evaluation
import time


def accuracy_evaluation(predictions: list, labels: list) -> tuple:
    """Get f1 score and confusion matrix for predictions.
    
    Args:
        predictions: a list of size |examples| of ints in [0,1] reprsenting the predictions of a single model

        labels: a list of size |examples| of ints in [0,1] representing the true labels for each example

    Returns:
        a tuple (f1, cm) representing the f1 score and confusion matrix for the predictions.
    """
    f1 = accuracy_score(predictions, labels)
    cm = confusion_matrix(labels, predictions)
    return f1, cm

def main():
    '''
    Main function that loads train and test data and model configs, creates a model that trains and classifies
    examples, and writes output and results to files.
    '''
    setup_file = sys.argv[1]

    # load setup
    setup = load_yaml(setup_file)

    experiment_name = setup['name']
    adaptation_data_file = setup['adaptation_data']
    train_file = setup['train_data']
    model_configs = setup['model_configs']

    # set random seed
    set_seed(42)

    # load data 
    # TODO: Should we split the adaptation data, or treat as a test set?
    adaptation_data = load_data(adaptation_data_file, task="adaptation")
    adaptation_examples, adaptation_labels = list(zip(*adaptation_data))

    train = load_data(train_file)
    train, dev, _ = shuffle_split_data(train)

    train_examples, train_labels = list(zip(*train))
    dev_examples, dev_labels = list(zip(*dev))

    # specify the class of models used in experiment
    models = {
        'PredictRandom': PredictRandom,
        'PredictFalse': PredictFalse,
        'DeepMojiPolarityWithVoting': DeepMojiPolarityWithVoting,
        'DeepMojiWithSVM': DeepMojiWithSVM,
        'DeepMojiPolarityWithKNN': DeepMojiPolarityWithKNN,
        'DeepMojiPolarityWithAdaBoost': DeepMojiPolarityWithAdaBoost,
    }
    
    dev_model_results = []
    eval_model_results = []
    avg_len_results = []

    for filename in model_configs:

        # begin timer for performance tracking purposes
        starttime = time.perf_counter()
        
        model_config = load_yaml(filename)
        if not model_config['enabled']:
            continue

        # load model
        model_name = model_config['name']
        print(model_name)
        model_type = models[model_config['model']]
        model = model_type(model_config['hyperparameters'])

        print("training...")
        model.fit(train_examples, train_labels)
        print("train predictions...")
        train_predictions = model.predict(train_examples, split='train')
        print("dev predictions...")
        dev_predictions = model.predict(dev_examples, split='dev')        
        print("adaptation task predictions...")        
        # test_predictions = model.predict(test_examples, split='test')
        adaptation_predictions = model.predict(adaptation_examples, task="adaptation")

        # output time taken to train model
        traintime = time.perf_counter()
        print(model_name + " trained model in " + str(round(traintime - starttime, 4)) + " seconds")

        write_output_to_csv(
            experiment_name, model_name, 'train', 
            train_examples, train_predictions, 'adaptation'
        )
        write_output_to_csv(
            experiment_name, model_name, 'dev', 
            dev_examples, dev_predictions, 'adaptation'
        )
        write_output_to_csv(
            experiment_name, model_name, 'adaptation_test',
            adaptation_examples, adaptation_predictions, 'adaptation'
        )

        # write correct and incorrect predictions to respective outputs and store info on avg len of strings

        write_error_analysis_to_files(
            experiment_name, model_name, 'train',
            train_examples, train_labels, train_predictions, 'adaptation'
        )


        write_error_analysis_to_files(
            experiment_name, model_name, 'dev',
            dev_examples, dev_labels, dev_predictions, 'adaptation' 
        )

        avg_len_results.append(
            write_error_analysis_to_files(
                experiment_name, model_name, 'adaptation_test',
                adaptation_examples, adaptation_labels, adaptation_predictions, 'adaptation' 
            ))

        # write to results/scores.out
        train_results = accuracy_evaluation(train_predictions, train_labels)
        dev_results = accuracy_evaluation(dev_predictions, dev_labels)
        adaptation_results = accuracy_evaluation(adaptation_predictions, adaptation_labels)

        dev_model_results.append(
            format_model_result(
                model_name=model_name, 
                train_results=train_results, 
                dev_results=dev_results,
                test_results=None,
                task="adaptation",
            )
        )
        eval_model_results.append(
            format_model_result(
                model_name=model_name, 
                train_results=None, 
                dev_results=None,
                test_results=adaptation_results,
                task="adaptation",
            )
        )

        #end timer for performance tracking purposes
        endtime = time.perf_counter()
        print(model_name + " ran in " + str(round(endtime - starttime, 4)) + " seconds")

    write_results_to_out(dev_model_results, setup['adaptation_dev_folder']+setup['results_filename'])
    write_results_to_out(eval_model_results, setup['adaptation_eval_folder']+setup['results_filename'])
    write_avg_len_to_out(avg_len_results, experiment_name, 'adaptation')



if __name__ == "__main__":
    main()
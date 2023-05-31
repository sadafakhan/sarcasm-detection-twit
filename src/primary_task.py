import sys
import yaml
from util import *
from model import *
from sklearn.metrics import f1_score, confusion_matrix
import time

def evaluation(predictions: list, labels: list) -> tuple:
    """Get f1 score and confusion matrix for predictions.
    
    Args:
        predictions: a list of size |examples| of ints in [0,1] reprsenting the predictions of a single model

        labels: a list of size |examples| of ints in [0,1] representing the true labels for each example

    Returns:
        a tuple (f1, cm) representing the f1 score and confusion matrix for the predictions.
    """
    f1 = f1_score(predictions, labels, average='macro')
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
    train_file = setup['train_data']
    test_file = setup['test_data']
    model_configs = setup['model_configs']

    # set random seed
    set_seed(42)

    # load data
    train = load_data(train_file)
    test = load_data(test_file)
    train, dev, test = shuffle_split_data(train, test)

    train_examples, train_labels = list(zip(*train))
    dev_examples, dev_labels = list(zip(*dev))
    test_examples, test_labels = list(zip(*test))

    # specify the class of models used in experiment
    models = {
        'PredictRandom': PredictRandom,
        'PredictFalse': PredictFalse,
        'DeepMojiWithSVM': DeepMojiWithSVM,
        'DeepMojiWithRandomForest': DeepMojiWithRandomForest,
        'DeepMojiWithAdaBoost': DeepMojiWithAdaBoost,
        'DeepMojiWithMLP': DeepMojiWithMLP,
        'DeepMojiWithKNN': DeepMojiWithKNN,
        'DeepMojiWithNaiveBayes': DeepMojiWithNaiveBayes,
        'DeepMojiWithLogisticRegression': DeepMojiWithLogisticRegression,
        'DeepMojiWithVoting': DeepMojiWithVoting,
        'DeepMojiPolarityWithKNN': DeepMojiPolarityWithKNN,
        'DeepMojiPolarityWithAdaBoost': DeepMojiPolarityWithAdaBoost,
        'DeepMojiPolarityWithVoting': DeepMojiPolarityWithVoting,
        'DeepMojiPolarityWithMLP': DeepMojiPolarityWithMLP,
    }
    
    dev_model_results = []
    eval_model_results = []

    for filename in model_configs:

        # begin timer for performance tracking purposes
        starttime = time.perf_counter()

        
        model_config = load_yaml(filename)

        if not model_config['enabled']:
            continue # Immediately go to next iteration of the loop

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
        print("test predictions...")        
        test_predictions = model.predict(test_examples, split='test')

        # output time taken to train model
        traintime = time.perf_counter()
        print(model_name + " trained model in " + str(round(traintime - starttime, 4)) + " seconds")

        write_output_to_csv(
            experiment_name, model_name, 'train', 
            train_examples, train_predictions, 'primary'
        )
        write_output_to_csv(
            experiment_name, model_name, 'dev', 
            dev_examples, dev_predictions, 'primary'
        )
        write_output_to_csv(
            experiment_name, model_name, 'test', 
            test_examples, test_predictions, 'primary'
        )

        # write correct and incorrect predictions to respective outputs
        write_error_analysis_to_files(
            experiment_name, model_name, 'train',
            train_examples, train_labels, train_predictions, 'primary'
        )

        write_error_analysis_to_files(
            experiment_name, model_name, 'dev',
            dev_examples, dev_labels, dev_predictions , 'primary'
        )

        write_error_analysis_to_files(
            experiment_name, model_name, 'test',
            test_examples, test_labels, test_predictions , 'primary'
        )

        # write to results/scores.out
        train_results = evaluation(train_predictions, train_labels)
        dev_results = evaluation(dev_predictions, dev_labels)
        test_results = evaluation(test_predictions, test_labels)

        dev_model_results.append(
            format_model_result(
                model_name=model_name, 
                train_results=None, 
                dev_results=dev_results,
                test_results=None
            )
        )
        eval_model_results.append(
            format_model_result(
                model_name=model_name, 
                train_results=None, 
                test_results=test_results,
                dev_results=None
            )
        )

        #end timer for performance tracking purposes
        endtime = time.perf_counter()
        print(model_name + " ran in " + str(round(endtime - starttime, 4)) + " seconds")

    write_results_to_out(dev_model_results, setup['primary_dev_folder']+setup['results_filename'])
    write_results_to_out(eval_model_results, setup['primary_eval_folder']+setup['results_filename'])


if __name__ == "__main__":
    main()

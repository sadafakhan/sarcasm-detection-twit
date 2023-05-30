# 573-affect-research-group

Connie Chen, Mickey Shi, Nathaniel Imel, Sadaf Khan

## Task Descriptions

This project is based on the iSarcasmEval shared-task (Task 6 at SemEval 2022).

**primary task**: Given a text, determine whether it is sarcastic or non-sarcastic;

**adaptation task**: Given a sarcastic text and its non-sarcastic rephrase, i.e. two texts that convey the same meaning, determine which is the sarcastic one.

## Structure of this repository

The _data_ folder consists of the train and test datasets that are used to train and evaluate models for the task.  
The _doc_ folder contains reports, powerpoints, additional documentation.  
The _outputs_ folder contains output from the system that are read during evalution.  
The _environment.yml_ file contains the packages necessary for this project.  
The _results_ folder contains scores from running the primary task's evaluation system on the outputs of the affect recognition system.
The _setup_ folder contains config files file paths relevant to each deliverable.  
The _src_ folder contains source code, including models, configs, evaluation tools, and driver scripts.  

The code consists of the following parts:

Model building:

- intitialize a statistical model defined in the models/ subfolder and by its config file

Training and Evaluation:

- Fit the model to training data
- Evaluate the model on dev set for hyperparameter selection
- Evaluate the best model on test set

Results:

- Report the results of the model's performance

## Requirements

The required virtual environment is specified by the YAML file in the root of the repository. To create the environment named `sarcasm' and activate it, run  

``conda env create --file environment.yml``  

``conda activate sarcasm``

NB: To update the conda virtual environment, run

``conda env export --from-history --no-build > environment.yml``

and remove the prefix afterwards.

If for some reason you need to remove the conda environment (e.g. to build one from a new YAML file), use the command:

``conda env remove --name sarcasm``

### TorchMoji installation
After setting up the virtual environment, follow the installation steps to install torchmoji via the installation process in the [TorchMoji GitHub repository](https://github.com/huggingface/torchMoji) _with the virtual environment activated_.

This should include cloning the repository into a local directory, running

``pip install -e .`` 

to install TorchMoji, running 

``python scripts/download_weights.py``

to download the pretrained model weights.

To test whether TorchMoji was successfully installed, run

``python examples/encode_texts.py``

**Unfortunately, the torchMoji software contained in the current huggingface repository contains a bug.** You will get the following error (if it was installed correctly):

  File "/torchMoji/torchmoji/lstm.py", line 78, in forward
      input, batch_sizes = input
      ValueError: too many values to unpack (expected 2)

You need to change this line to read:
    
      input, batch_sizes, _, _ = input

If you run the script again, now it should run to completion, generating text embeddings for the provided 5 sentences. This indicates that our project is ready to be tested.


## Replicating main results

The main experimental results can be reproduced by running the script src/primary_task.sh. On patas (dryas), run

```condor_submit D4.cmd.```

The resulting file contains the output of running the script, including the time it took to train each model and
the total amount of time it took for each model to run.

``D4.out`` 

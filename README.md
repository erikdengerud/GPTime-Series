# GPTime-Series
Code used in my masters thesis. We show that a simple MLP can be trained jointly across all time series in the M4 dataset to a score that could have placed second in the M4 forecasting competition. Using a pre-trained model trained jointly on all the time series in the FRED dataset, we can forecast M4 with a score that could have placed 14th. Both of these are far better than the benchmark MLP in the M4 competition. The benchmark used one MLP for each time series and placed 57th of 59.

## Usage
The GPTime module can be used to download data, preprocess data, train models, evaluate models, and finetune models. The tasks are provided as arguments together with the path of a config file with the arguments for the task. In addition, the scripts in the `scripts`-folder are used to create ensembles and the slurm scripts used to train ensembles using external resources with a slurm worlkload scheduler.

### GPTime Module
The module can be used by running e.g. `python3 -m GPTime --task train --cfg_path cfg_file.yml` to train a model using arguments in the specified config file.

Running `python3 -m GPTime --help` gives the options:
```
Usage: __main__.py [OPTIONS]                                                                                         
Options:                                                                                                             
  --cfg_path TEXT                 [required]                                                                         
  --task [source|preprocess|train|evaluate|finetune]                                                                 
                                  Name of task to execute  [required]                                                
  --help                          Show this message and exit. 
```

**Source**
Sourcing the M4 and FRED datasets. The FRED dataset requires an API key from https://fred.stlouisfed.org/docs/api/api_key.html. Downloading the FRED dataset takes several days due to the request limit of the API. An example of settings are in `configs/config_source.yml`.

**Preprocess**
Preprocessing the downloaded datasets into a common format for all datasets. Again, an example of settings are in `configs/config_preprocess.yml`.

**Train**
Training a model on a dataset. The model is an MLP and can be jointly trained on all time series or for each frequency. Example settings are in `configs/config_train.yml`.

**Evaluate**
Evaluating a saved model on the test part of the M4 dataset. This gives the MASE, SMAPE, and OWA scores for each frequency and in total. Example settings are in `configs/config_evaluate.yml`.

**Finetune**
Finetuning a pre-trained model. We use this to pretrain a finetuned model on the M4 dataset. This code is even less pretty than the rest, but might be redone in the future. Example settings are in `configs/config_finetune.yml`.

### Scripts
The scripts folder has scripts that are used to create the config files and slurm jobs to train an ensemble. It also has files used to evaluate a trained ensemble. Most of the scripts use config files.

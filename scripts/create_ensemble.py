import os
import yaml
from box import Box
import click
import logging
import glob
from itertools import product
import pandas as pd

def create_ensemble_slurm(slurm_jobs_folder, ensemble_name, model_name, cfg_path, evaluate_cfg_path:str=None):

    slurm_job_name = f"{model_name}.slurm"
    slurm_job_file_path = os.path.join(*[slurm_jobs_folder, ensemble_name, slurm_job_name])
    with open(slurm_job_file_path, "w") as f:
        f.write("#!/bin/sh\n")
        f.write(f"#SBATCH --partition=GPUQ\n")
        f.write(f"#SBATCH --account=share-ie-imf\n")
        f.write(f"#SBATCH --time=6:00:00\n") # the box config parses to minutes
        f.write(f"#SBATCH --nodes=1\n")
        f.write(f"#SBATCH --mem=16000\n")
        f.write(f"#SBATCH --job-name={ensemble_name}\n")
        f.write(f"#SBATCH --output={ensemble_name}-{model_name}.out\n")
        f.write(f"#SBATCH --mail-user=eriko1306@gmail.com\n")
        f.write(f"#SBATCH --mail-type=ALL\n")
        f.write(f"#SBATCH --gres=gpu:1\n")

        f.write("module load GCC/8.3.0 CUDA/10.1.243 OpenMPI/3.1.4 PyTorch/1.3.1-Python-3.7.4\n")
        f.write("source venv/bin/activate\n")

        f.write(f"python3 -m GPTime --task train --cfg_path {cfg_path}\n")
        if evaluate_cfg_path is not None:
            f.write(f"python3 -m GPTime --task evaluate --cfg_path {evaluate_cfg_path}")
        f.close()

def create_run_slurm_jobs(slurm_jobs_folder, ensemble_name):
    run_slurm_fname = f"{slurm_jobs_folder}/run_{ensemble_name}.sh"
    with open(run_slurm_fname, "w") as f:
        f.write("#!/bin/sh\n")
        echo_string = ""
        slurm_fnames = glob.glob(os.path.join(slurm_jobs_folder, ensemble_name) + "/*")
        slurm_fnames = [fname for fname in slurm_fnames if "evaluate" not in fname]
        for i, slurm_fname in enumerate(slurm_fnames):
            f.write(f"chmod u+x {slurm_fname}\n")
            f.write(f"RES{i}=$(sbatch --parsable {slurm_fname})\n")
            f.write(f"echo \"$RES{i}, {slurm_fname}\" >> submitted_jobs_names.log\n")
            if (i+1)%5 == 0:
                f.write("sleep 5\n")
        f.close()


@click.command()
@click.option("--cfg_path", required=True)
def create_ensemble(cfg_path):
    """
    TODO: make model save path
          make evaluate ensemble!
    """
    horizons = {
        "Y": 6,
        "Q": 8,
        "M": 18,
        "W": 13,
        "D": 14,
        "H": 48,
        }
    with open(cfg_path, "r") as ymlfile:
        ensemble_cfg = Box(yaml.safe_load(ymlfile))
    
    ensemble_folders = glob.glob(os.path.join(ensemble_cfg.cfg_files_path, "*"))
    ensemble_folder = os.path.join(ensemble_cfg.cfg_files_path, ensemble_cfg.ensemble_name)
    if ensemble_folder in ensemble_folders:
        print("An ensemble with that name already exists!")
        #raise Warning
    else:
        os.makedirs(os.path.join(ensemble_cfg.cfg_files_path, ensemble_cfg.ensemble_name))
        os.makedirs(os.path.join(ensemble_cfg.slurm_jobs_folder, ensemble_cfg.ensemble_name))
        os.makedirs(os.path.join(ensemble_cfg.storage_folder, ensemble_cfg.ensemble_name))

    # create ensemble member cfgs
    with open(ensemble_cfg.train_cfg_path, "r") as ymlfile:
        train_cfg = Box(yaml.safe_load(ymlfile))
    with open(ensemble_cfg.evaluate_cfg_path, "r") as ymlfile:
        evaluate_cfg = Box(yaml.safe_load(ymlfile))
    
    if ensemble_cfg.global_model:
        for member in product(ensemble_cfg.loss_functions, ensemble_cfg.forecast_inits, ensemble_cfg.input_window_lengths):
            # create model name and paths
            model_name = "global"
            ensemble_member_name = f"{member[0]}-{member[1]}-{member[2]}"
            #model_name = f"global-{member[0]}-{member[1]}-{member[2]}"
            ensemble_member_yml = ensemble_member_name + ".yml"
            ensemble_member_cfg_path = os.path.join(ensemble_folder, ensemble_member_yml)
            # set the ensemble variation
            train_cfg.name = model_name
            train_cfg.model_save_path = os.path.join(*[ensemble_cfg.storage_folder, ensemble_cfg.ensemble_name, ensemble_member_name])
            train_cfg.criterion_name = member[0]
            train_cfg.seasonal_init = True if member[1] == "seasonal_init" else False
            train_cfg.model_params_mlp.in_features = member[2]
            train_cfg.val = ensemble_cfg.val_set
            # set all frequencies to True
            for f in train_cfg.dataset_params.frequencies:
                train_cfg.dataset_params.frequencies[f] = True
            train_cfg.dataset_params.frequencies["O"] = False
            # save the cfg file
            train_cfg.to_yaml(ensemble_member_cfg_path)

            # evaluate_cfg
            evaluate_cfg.name = model_name
            evaluate_cfg.model_save_path = train_cfg.model_save_path
            evaluate_cfg.result_path = train_cfg.model_save_path
            evaluate_cfg.predictions_path = train_cfg.model_save_path
            evaluate_cfg.scale = train_cfg.scale
            evaluate_cfg.seasonal_init = train_cfg.seasonal_init
            evaluate_cfg.global_model = True
            evaluate_cfg.model_params_mlp = train_cfg.model_params_mlp
            evaluate_cfg.val_set = ensemble_cfg.val_set
            ensemble_member_cfg_path_evaluate = os.path.join(ensemble_folder, ensemble_member_name + "-evaluate.yml")
            evaluate_cfg.to_yaml(ensemble_member_cfg_path_evaluate)
            
            create_ensemble_slurm(
                ensemble_cfg.slurm_jobs_folder,
                ensemble_cfg.ensemble_name,
                ensemble_member_name,
                ensemble_member_cfg_path,
                ensemble_member_cfg_path_evaluate
                )
            os.makedirs(
                os.path.join(*[
                    ensemble_cfg.storage_folder,
                    ensemble_cfg.ensemble_name,
                    ensemble_member_name
                    ]
                ))
    else:
        epochs = {"Y":1000, "Q":1000, "M":600, "W":20000, "D":4000, "H":20000}
        early_stop = {"Y":150, "Q":150, "M":90, "W": 3000, "D":600, "H":3000}
        patience = {"Y": 50, "Q": 50, "M": 30, "W": 1000, "D": 200, "H": 1000}
        log_freq = {"Y":30, "Q":20, "M":20, "W":1000, "D":1000, "H":1000}
        evaluate_configs_list = []
        for member in product(ensemble_cfg.loss_functions, ensemble_cfg.forecast_inits, ensemble_cfg.lookbacks):
            for freq in train_cfg.dataset_params.frequencies:
                # create model name and paths
                model_name = freq
                ensemble_member_name = f"{member[0]}-{member[1]}-{member[2]}"
                ensemble_member_yml = ensemble_member_name + "-"+ freq + ".yml"
                ensemble_member_cfg_path = os.path.join(ensemble_folder, ensemble_member_yml)
                # set the ensemble variation
                train_cfg.name = model_name
                train_cfg.model_save_path = os.path.join(*[ensemble_cfg.storage_folder, ensemble_cfg.ensemble_name, ensemble_member_name])
                train_cfg.criterion_name = member[0]
                train_cfg.seasonal_init = True if member[1] == "seasonal_init" else False
                train_cfg.model_params_mlp.in_features = member[2] * horizons[freq]
                train_cfg.max_epochs = epochs[freq]
                train_cfg.early_stop_tenacity = early_stop[freq]
                train_cfg.log_freq = log_freq[freq]
                train_cfg.patience = patience[freq]
                train_cfg.val = ensemble_cfg.val_set
                # set all frequencies to True
                for f in train_cfg.dataset_params.frequencies:
                    if f == freq:
                        train_cfg.dataset_params.frequencies[f] = True
                    else:
                        train_cfg.dataset_params.frequencies[f] = False
                # save the cfg file
                train_cfg.to_yaml(ensemble_member_cfg_path)
                create_ensemble_slurm(
                    ensemble_cfg.slurm_jobs_folder,
                    ensemble_cfg.ensemble_name,
                    ensemble_member_name + "-" + freq,
                    ensemble_member_cfg_path,
                    )
            # evaluate_cfg
            evaluate_cfg.name = ensemble_member_name
            evaluate_cfg.model_save_path = train_cfg.model_save_path
            evaluate_cfg.result_path = train_cfg.model_save_path
            evaluate_cfg.predictions_path = train_cfg.model_save_path
            evaluate_cfg.scale = train_cfg.scale
            evaluate_cfg.seasonal_init = train_cfg.seasonal_init
            evaluate_cfg.global_model = False
            evaluate_cfg.model_params_mlp = train_cfg.model_params_mlp
            evaluate_cfg.lookback = member[2]
            evaluate_cfg.val_set = ensemble_cfg.val_set
            ensemble_member_cfg_path_evaluate = os.path.join(ensemble_folder, ensemble_member_name + "-evaluate.yml")
            evaluate_cfg.to_yaml(ensemble_member_cfg_path_evaluate)
            evaluate_configs_list.append(ensemble_member_cfg_path_evaluate)

            os.makedirs(
                os.path.join(*[
                    ensemble_cfg.storage_folder,
                    ensemble_cfg.ensemble_name,
                    ensemble_member_name
                    ]
                ))

        slurm_job_name = f"evaluate-{ensemble_cfg.ensemble_name}.slurm"
        slurm_job_file_path = os.path.join(*[ensemble_cfg.slurm_jobs_folder, ensemble_cfg.ensemble_name, slurm_job_name])
        with open(slurm_job_file_path, "w") as f:
            f.write("#!/bin/sh\n")
            f.write(f"#SBATCH --partition=GPUQ\n")
            f.write(f"#SBATCH --account=share-ie-imf\n")
            f.write(f"#SBATCH --time=1:00:00\n") # the box config parses to minutes
            f.write(f"#SBATCH --nodes=1\n")
            f.write(f"#SBATCH --mem=16000\n")
            f.write(f"#SBATCH --job-name={ensemble_cfg.ensemble_name}\n")
            f.write(f"#SBATCH --output={ensemble_cfg.ensemble_name}-evaluate.out\n")
            f.write(f"#SBATCH --mail-user=eriko1306@gmail.com\n")
            f.write(f"#SBATCH --mail-type=ALL\n")
            f.write(f"#SBATCH --gres=gpu:1\n")

            f.write("module load GCC/8.3.0 CUDA/10.1.243 OpenMPI/3.1.4 PyTorch/1.3.1-Python-3.7.4\n")
            f.write("source venv/bin/activate\n")

            for member_cfg in evaluate_configs_list:
                f.write(f"python3 -m GPTime --task evaluate --cfg_path {member_cfg}\n")
            f.close()

    """
    else:
        for freq in train_cfg.dataset_params.frequencies:
            for member in product(ensemble_cfg.loss_functions, ensemble_cfg.forecast_inits, ensemble_cfg.input_window_lengths):
                # create model name and paths
                model_name = freq
                ensemble_member_name = f"{freq}-{member[0]}-{member[1]}-{member[2]}"
                ensemble_member_yml = ensemble_member_name + ".yml"
                ensemble_member_cfg_path = os.path.join(ensemble_folder, ensemble_member_yml)
                # set the ensemble variation
                train_cfg.name = model_name
                train_cfg.model_save_path = os.path.join(*[ensemble_cfg.storage_folder, ensemble_cfg.ensemble_name, ensemble_member_name])
                train_cfg.criterion_name = member[0]
                train_cfg.seasonal_init = True if member[1] == "seasonal_init" else False
                train_cfg.model_params_mlp.in_features = member[2]
                # set all frequencies to True
                for f in train_cfg.dataset_params.frequencies:
                    if f == freq:
                        train_cfg.dataset_params.frequencies[f] = True
                    else:
                        train_cfg.dataset_params.frequencies[f] = False
                # save the cfg file
                train_cfg.to_yaml(ensemble_member_cfg_path)
                # evaluate_cfg
                evaluate_cfg.name = model_name
                evaluate_cfg.model_save_path = train_cfg.model_save_path
                evaluate_cfg.result_path = train_cfg.model_save_path
                evaluate_cfg.predictions_path = train_cfg.model_save_path
                evaluate_cfg.scale = train_cfg.scale
                evaluate_cfg.seasonal_init = train_cfg.seasonal_init
                evaluate_cfg.global_model = True
                evaluate_cfg.model_params_mlp = train_cfg.model_params_mlp
                ensemble_member_cfg_path_evaluate = os.path.join(ensemble_folder, ensemble_member_name + "-evaluate.yml")
                evaluate_cfg.to_yaml(ensemble_member_cfg_path_evaluate)

                create_ensemble_slurm(
                    ensemble_cfg.slurm_jobs_folder,
                    ensemble_cfg.ensemble_name,
                    ensemble_member_name,
                    ensemble_member_cfg_path,
                    ensemble_member_cfg_path_evaluate
                    )

            os.makedirs(
                os.path.join(*[
                    ensemble_cfg.storage_folder,
                    ensemble_cfg.ensemble_name,
                    ensemble_member_name
                    ]
                ))
    """
    create_run_slurm_jobs(ensemble_cfg.slurm_jobs_folder, ensemble_cfg.ensemble_name)


if __name__=="__main__":
    create_ensemble()



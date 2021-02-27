import os
import yaml
from box import Box
import click
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

        f.write(f"python3 -m GPTime --task finetune --cfg_path {cfg_path}\n")
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
def finetune_ensemble(cfg_path):
    with open(cfg_path, "r") as ymlfile:
        finetune_cfg = Box(yaml.safe_load(ymlfile))
    
    model_cfgs = glob.glob(os.path.join(*[finetune_cfg.config_path, finetune_cfg.experiment_path.split("/")[-1], "*"]))
    model_cfgs = [f for f in model_cfgs if "evaluate" not in f]
    
    finetune_name = "finetune_" + finetune_cfg.experiment_path.split("/")[-1]
    os.makedirs(os.path.join(finetune_cfg.config_path, finetune_name), exist_ok=True)
    os.makedirs(os.path.join(finetune_cfg.slurm_jobs_folder, finetune_name), exist_ok=True)
    os.makedirs(os.path.join(finetune_cfg.storage_folder, finetune_name), exist_ok=True)
    
    if finetune_cfg.global_model:
        for model_cfg_path in model_cfgs:
            # Read the model training file
            with open(model_cfg_path, "r") as ymlfile:
                member_train_cfg = Box(yaml.safe_load(ymlfile))

            # Copy the finetuning file
            member_finetune_cfg = finetune_cfg.copy()

            # Change the model parameters and dataset frequencies
            member_finetune_cfg.model_params_mlp = member_train_cfg.model_params_mlp
            member_finetune_cfg.dataset_params.frequencies = member_train_cfg.dataset_params.frequencies
            member_finetune_cfg.model_path = os.path.join(member_train_cfg.model_save_path, "global.pt")
            member_finetune_cfg.model_save_path = os.path.join(*[finetune_cfg.storage_folder, finetune_name, member_train_cfg.model_save_path.split("/")[-1]])

            # Save the config file
            save_path = os.path.join(finetune_cfg.config_path, finetune_name, member_train_cfg.model_save_path.split("/")[-1] + ".yml")
            member_finetune_cfg.to_yaml(save_path)

            # Save the config file and make slurm job        
            create_ensemble_slurm(
                member_finetune_cfg.slurm_jobs_folder,
                finetune_name,
                member_train_cfg.model_save_path.split("/")[-1],
                save_path,
                )
            os.makedirs(member_finetune_cfg.model_save_path, exist_ok=True)
        create_run_slurm_jobs(finetune_cfg.slurm_jobs_folder, finetune_name)

    else:
        epochs_1 = {"Y":500, "Q":500, "M":300, "W":10000, "D":2000, "H":10000}
        epochs_2 = {"Y":750, "Q":750, "M":450, "W":15000, "D":3000, "H":15000}
        epochs_3 = {"Y":1000, "Q":1000, "M":600, "W":20000, "D":4000, "H":20000}
        early_stop = {"Y":150, "Q":150, "M":90, "W": 3000, "D":600, "H":3000}
        patience = {"Y": 50, "Q": 50, "M": 30, "W": 1000, "D": 200, "H": 1000}
        log_freq = {"Y":30, "Q":20, "M":20, "W":1000, "D":1000, "H":1000}
        for model_cfg_path in model_cfgs:
            print(model_cfg_path)
            with open(model_cfg_path, "r") as ymlfile:
                member_train_cfg = Box(yaml.safe_load(ymlfile))

            # Copy the finetuning file
            member_finetune_cfg = finetune_cfg.copy()

            # Change the model parameters and dataset frequencies
            member_finetune_cfg.model_params_mlp = member_train_cfg.model_params_mlp
            member_finetune_cfg.dataset_params.frequencies = member_train_cfg.dataset_params.frequencies
            member_finetune_cfg.model_path = os.path.join(member_train_cfg.model_save_path, member_train_cfg.name + ".pt")
            member_finetune_cfg.model_save_path = os.path.join(*[finetune_cfg.storage_folder, finetune_name, member_train_cfg.model_save_path.split("/")[-1]])
            member_finetune_cfg.max_epochs_1 = epochs_1[member_train_cfg.name]
            member_finetune_cfg.max_epochs_2 = epochs_2[member_train_cfg.name]
            member_finetune_cfg.max_epochs_3 = epochs_3[member_train_cfg.name]
            member_finetune_cfg.early_stop_tenacity = early_stop[member_train_cfg.name]
            member_finetune_cfg.patience = patience[member_train_cfg.name]
            member_finetune_cfg.log_freq = log_freq[member_train_cfg.name]
            member_finetune_cfg.name = member_train_cfg.name

            # Save the config file
            save_path = os.path.join(finetune_cfg.config_path, finetune_name, member_train_cfg.model_save_path.split("/")[-1] + "-" + member_train_cfg.name + ".yml")
            member_finetune_cfg.to_yaml(save_path)

            # Save the config file and make slurm job 
            create_ensemble_slurm(
                member_finetune_cfg.slurm_jobs_folder,
                finetune_name,
                member_train_cfg.model_save_path.split("/")[-1] + member_train_cfg.name,
                save_path,
                )
            os.makedirs(member_finetune_cfg.model_save_path, exist_ok=True)
        create_run_slurm_jobs(finetune_cfg.slurm_jobs_folder, finetune_name)


if  __name__ == "__main__":
    finetune_ensemble()

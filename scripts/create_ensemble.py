import os
import yaml
from box import Box
import click
import logging
import glob
from itertools import product

def create_ensemble_slurm(slurm_jobs_folder, ensemble_name, model_name, cfg_path):

    slurm_job_name = f"{model_name}.slurm"
    slurm_job_file_path = os.path.join(*[slurm_jobs_folder, ensemble_name, slurm_job_name])
    with open(slurm_job_file_path, "w") as f:
        f.write("#!/bin/sh\n")
        f.write(f"#SBATCH --partition=GPUQ\n")
        f.write(f"#SBATCH --account=share-ie-imf\n")
        f.write(f"#SBATCH --time=1:00:00\n") # the box config parses to minutes
        f.write(f"#SBATCH --nodes=1\n")
        f.write(f"#SBATCH --mem=16000\n")
        f.write(f"#SBATCH --job-name={ensemble_name}\n")
        f.write(f"#SBATCH --output={ensemble_name}-{model_name}.out\n")
        f.write(f"#SBATCH --mail-user=eriko1306@gmail.com\n")
        f.write(f"#SBATCH --mail-type=ALL\n")
        f.write(f"#SBATCH --gres=gpu:1\n")

        f.write("module load GCC/8.3.0 CUDA/10.1.243 OpenMPI/3.1.4 PyTorch/1.3.1-Python-3.7.4\n")
        f.write("source venv/bin/activate\n")

        f.write(f"python3 -m GPTime --task train --cfg_path {cfg_path}")
        f.close()

def create_run_slurm_jobs(slurm_jobs_folder, ensemble_name):
    run_slurm_fname = f"{slurm_jobs_folder}/run_{ensemble_name}.sh"
    with open(run_slurm_fname, "w") as f:
        f.write("#!/bin/sh\n")
        echo_string = ""
        slurm_fnames = glob.glob(os.path.join(slurm_jobs_folder, ensemble_name) + "/*")
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
            # set all frequencies to True
            for f in train_cfg.dataset_params.frequencies:
                train_cfg.dataset_params.frequencies[f] = True
            # save the cfg file
            train_cfg.to_yaml(ensemble_member_cfg_path)
            create_ensemble_slurm(
                ensemble_cfg.slurm_jobs_folder,
                ensemble_cfg.ensemble_name,
                ensemble_member_name,
                ensemble_member_cfg_path
                )
            os.makedirs(
                os.path.join(*[
                    ensemble_cfg.storage_folder,
                    ensemble_cfg.ensemble_name,
                    ensemble_member_name
                    ]
                ))
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
                create_ensemble_slurm(
                    ensemble_cfg.slurm_jobs_folder,
                    ensemble_cfg.ensemble_name,
                    ensemble_member_name,
                    ensemble_member_cfg_path
                    )
            os.makedirs(
                os.path.join(*[
                    ensemble_cfg.storage_folder,
                    ensemble_cfg.ensemble_name,
                    ensemble_member_name
                    ]
                ))

    create_run_slurm_jobs(ensemble_cfg.slurm_jobs_folder, ensemble_cfg.ensemble_name)


if __name__=="__main__":
    create_ensemble()



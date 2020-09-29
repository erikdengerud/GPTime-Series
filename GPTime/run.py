import click
import logging
import yaml
from box import Box
import sys
import glob

sys.path.append("")
logger = logging.getLogger(__name__)

from GPTime.config import cfg
from GPTime.source.data_sourcing import source
from GPTime.preprocess.preprocessing import preprocess
from GPTime.model.train import train
from GPTime.model.evaluate import evaluate
with open("GPTime/credentials.yml", "r") as ymlfile:
    credentials = Box(yaml.safe_load(ymlfile))

#source(credentials, small_sample=True)


def run_pipeline():

    # Check if sourced data exists
    for ds in cfg.source.path:
        # glob dir and check empty
        dirs = glob.glob(cfg.source.path[ds]["raw"] + "*")
        if len(dirs) == 0:
            logger.info("Sourcing raw data.")
            source(credentials=credentials, small_sample=True)
            break
    logger.info("Finished sourcing step.")
    # Check if preprocessed data exists
    for ds in cfg.preprocess.path:
        # glob dir and check empty
        dirs = glob.glob(cfg.preprocess.path[ds] + "*")
        if len(dirs) == 0:
            logger.info("preprocessing raw data.")
            preprocess()
            break
    logger.info("Finished preprocessing step.")

    # Check if trained model exists. Must check if all parameters are equal.
    model_names = [m[:-3] for m in glob.glob("GPTime/models/*.pt")]
    if cfg.run.name in model_names:
        logger.info("Model with that name already trained. Checking if parameters are the same.")
        # check if parameters are the same
        # if same load model
        # else train model
    else:
        # train model
        logger.info("Training model.")

    logger.info("Finished training step.")
    # Evaluate model on test data.
    evaluate()
    logger.info("Finished evaluating step.")

if __name__ == "__main__":
    run_pipeline()


"""
tasks = {
    "source": source,
    "preprocess": preprocess,
    "train": train,
    "evaluate": evaluate,
}
logger = logging.getLogger(__name__)


def main(task):
    try:
        tasks[task]()
    except:
        logger.error(f"Task {task} failed")
        raise


@click.command()
@click.option(
    "--task",
    type=click.Choice(tasks.keys()),
    required=True,
    help="Name of task to execute",
)
def main_cli(task):
    main(task)
"""
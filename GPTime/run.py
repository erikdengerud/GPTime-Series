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
from GPTime.model.train2 import train2
from GPTime.model.evaluate import evaluate
with open("GPTime/credentials.yml", "r") as ymlfile:
    credentials = Box(yaml.safe_load(ymlfile))

#source(credentials, small_sample=True)


def run_pipeline():
    tasks = {
        "source": source,
        "preprocess": preprocess,
        #"train": train,
        "train": train2,
        "evaluate": evaluate,
    }

    perform_task = {
        "source": cfg.run.source_hard,
        "preprocess": cfg.run.preprocess_hard,
        "train": cfg.run.train_hard,
        "evaluate": cfg.run.evaluate_hard,
    }   

    for task in tasks:
        if perform_task[task]:
            logger.info(f"Task: {task}")
            try:
                tasks[task]()
            except:
                logger.error(f"Task {task} failed")
                raise
    """
    logger.info("Training step.")
    train()
    logger.info("Done training model.")
    logger.info("Evaluating model.")
    m4_res = evaluate(cfg)
    logger.info("Done evaluating")
    logger.info("Results:")
    logger.info(m4_res)
    """
    """
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
    """
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
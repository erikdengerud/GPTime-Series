import yaml
from box import Box

with open("./configs/config.yml", "r") as ymlfile:
    cfg = Box(yaml.safe_load(ymlfile))

import os
import logging.config

os.makedirs(cfg.path.logs, exist_ok=True)

if os.path.exists(cfg.path.log_config):
    with open(cfg.path.log_config, "r") as ymlfile:
        log_config = yaml.safe_load(ymlfile)

    # Set up the logger configuration
    logging.config.dictConfig(log_config)
else:
    raise FileNotFoundError(f"Log yaml configuration file not found in {cfg.path.log_config}")

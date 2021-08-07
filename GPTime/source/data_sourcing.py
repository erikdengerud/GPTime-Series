import sys
import logging
sys.path.append("")

from GPTime.source.source_FRED import source_FRED
from GPTime.source.source_M4 import source_M4

from GPTime.config import cfg

logger = logging.getLogger(__name__)


def source(cfg)->None:
    """
    Sourcing raw data from data sources.
    """

    # Fred
    if cfg.source_FRED: 
        source_FRED(credentials=credentials.FRED, small_sample=small_sample)
    if cfg.source_M4:
        source_M4(
            train_download_paths=cfg.path.M4.download.train.values(),
            test_download_paths=cfg.path.M4.download.test.values(),
            meta_download_path=cfg.path.M4.download.meta,
            train_store_path=cfg.path.M4.store.raw_train,
            test_store_path=cfg.path.M4.store.raw_test,
            meta_store_path=cfg.path.M4.store.meta,
            )

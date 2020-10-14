import json
import glob
import logging
import matplotlib.pyplot as plt
import os
import numpy as np
import sys

sys.path.append("")
from GPTime.config import cfg

logger = logging.getLogger(__name__)
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

def calculate_statistics():
    """
    Calculating statistics of the FRED dataset.
    """
    lengths = []
    freqs = []
    dirs = glob.glob(cfg.preprocess.path.FRED + "/*")
    for d in dirs:
        fnames = glob.glob(d + "/*")
        for fname in fnames:
            with open(fname, "r") as fp:
                json_list = json.load(fp)
                fp.close()
            for ts in json_list:
                lengths.append(len(ts["observations"]))
                freqs.append(ts["frequency"])
    logger.info("Frequency distribution:")
    for f in list(set(freqs)):
        logger.info(f"{f} : {len([t for t in freqs if t == f]) / len(freqs) * 100:.2f}%")
    logger.info("Plotting")
    lengths = np.array(lengths)
    freqs = np.array(freqs)

    plt.figure()
    plt_array = lengths[lengths < (3 * np.median(lengths))]
    plt.hist(plt_array)
    plt.savefig(os.path.join(cfg.figures.path, "histogram_lengths_all.pdf"))
    all_length_freq = [lengths]
    labels = ["all"]
    for f in list(set(freqs)):
        logger.info(f"Plotting freq {f}...")
        plt.figure()
        lengths_freq = []
        for freq, length in zip(freqs, lengths):
            if freq == f:
                lengths_freq.append(length)
        plt_array = lengths[(lengths < (3 * np.median(lengths))) & (freqs == f)]
        plt.hist(plt_array)
        plt.savefig(os.path.join(cfg.figures.path, f"histogram_lengths_{f}.pdf"))

        all_length_freq.append(lengths_freq)
        labels.append(f)
    logger.info("Boxplot...")
    plt.figure()
    plt.boxplot(all_length_freq)
    plt.yscale("log")
    locs, _ = plt.xticks()  # Get the current locations and labels.
    plt.xticks(locs, labels)
    plt.savefig(os.path.join(cfg.figures.path, f"boxplot_lengths_freqs.pdf"))

    logger.info("Done")

if __name__ == "__main__":
    calculate_statistics()
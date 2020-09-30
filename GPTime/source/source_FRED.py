import fred
from typing import Dict, List, Tuple
import numpy as np
import json
import logging
import os
import sys
import time
sys.path.append("")

from GPTime.config import cfg

logger = logging.getLogger(__name__)

def crawl_fred(api_key:str, nodes_to_visit:List[int]=[0], sleep_time:int=60, rate_limit:int=100) -> None:
    fred.key(api_key)
    bottom_nodes = []
    ts_ids_freqs = {}
    num_visited = 0
    num_requests = 0
    while nodes_to_visit:
        curr_node = nodes_to_visit.pop()
        #logger.info(f"Current node: {curr_node:>4}")
        children = fred.children(curr_node)
        num_requests += 1
        try:
            if children["categories"]:
                for child in children["categories"]:
                    nodes_to_visit.append(child["id"])
            else:
                try:
                    bottom_nodes.append(curr_node)
                    seriess = fred.category_series(curr_node)["seriess"]
                    num_requests += 1
                    for ts in seriess:
                        ts_ids_freqs[ts["id"]] = ts["frequency_short"]
                except Exception as e:
                    logger.debug(e)
            num_visited += 1
            if num_visited % 100 == 0:
                logger.info(f"Visited {num_visited:>5} nodes and currently have {len(ts_ids_freqs):>6} time series ids saved")
                fname = time.ctime().replace(" ", "-").replace(":","-")+".json"
                with open(os.path.join(cfg.source.path.FRED.meta, fname), "w") as fp:
                    json.dump(ts_ids_freqs, fp, sort_keys=True, indent=4, separators=(',', ': '))
                    fp.close()
                fname = time.ctime().replace(" ", "-").replace(":","-")+"-nodes-to-visit.txt"
                with open(os.path.join(cfg.source.path.FRED.meta, fname), "w") as f:
                    f.write("\n".join([str(node) for node in nodes_to_visit]))
                    f.close()
            if num_requests > rate_limit:
                time.sleep(sleep_time)
                num_requests=0
        except Exception as e:
            logger.debug(e)
            logger.debug(f"Current node {curr_node}")
            logger.debug(f"{num_requests:>3} requests last minute")
    with open(os.path.join(cfg.source.path.FRED.meta, "ids_freq_list.json"), "w") as fp:
        json.dump(ts_ids_freqs, fp, sort_keys=True, indent=4, separators=(',', ': '))
        fp.close()


def source_FRED(credentials, small_sample:bool=False, id_freq_list_path:str="") -> None:
    """
    Source the full FRED dataset and save to files. https://fred.stlouisfed.org/
    """
    # Setup directories f they do not exist
    # TODO

    # Create fred connection using api-key
    fred.key(credentials.API_KEY_FED.key)

    if small_sample:
        try:
            if id_freq_list_path == "":
                filename="dummy_id_freq_list.json"
                with open(cfg.source.path.FRED.meta + filename, "r") as fp:
                    ids_freqs = json.load(fp)
            else:
                try:
                    with open(id_freq_list_path, "r") as fp:
                        ids_freqs = json.load(fp)
                except Exception as e:
                    logger.warning(e)
                    logger.warning(f"Not able to read provided file in path {id_freq_list_path}.")
            logger.info("Using precomputed list for retrieval from FRED.")
        except Exception as e:
            logger.info(e)
            logger.info("Not able to find predefined list of ids. Crawling FRED instead.") 
            # Crawl to get a full list of available time series.
            ids_freqs = {}
            for s in fred.category_series(33936)["seriess"]:
                ids_freqs[s["id"]] = s["frequency_short"]
            
            filename="dummy_id_freq_list.json"
            #path = os.path.join(cfg.source.path.FRED.meta, filename)
            with open(cfg.source.path.FRED.meta + filename, "w") as fp:
                json.dump(ids_freqs, fp, sort_keys=True, indent=4, separators=(',', ': '))
        
        # Download and save all time series. saving each sample as a JSON
        for id in ids_freqs.keys():
            observations = fred.observations(id)
            json_out = {
                "source" : "FRED",
                "id" : id,
                "frequency" : ids_freqs[id],
                "values" : [float(obs["value"]) for obs in observations["observations"]]
            }
            filename=f"{id}.json"
            #path = os.path.join(cfg.source.path.FRED.raw, filename)
            with open(cfg.source.path.FRED.raw + filename, "w") as fp:
                json.dump(json_out, fp)
            
        # Statistics of sourcing

        # Random dummy data for preprocessing
        num_preprocessed = 0
        for i in range(10000):
            if num_preprocessed % 1000 == 0:
                curr_dir = f"dir{num_preprocessed // 1000 :03d}/"
                os.makedirs(cfg.source.path.FRED.raw + curr_dir, exist_ok=True)
            out = {
                "source" : "FRED",
                "id" : f"{i:04d}",
                "frequency" : np.random.choice(["Y", "Q", "M", "W", "D", "H"]),
                "values" : list(np.random.rand(100)),
            }
            filename = f"{i:04d}.json"
            with open(cfg.source.path.FRED.raw + curr_dir + filename, "w") as fp:
                json.dump(out, fp)
            num_preprocessed += 1

    else:
        # Crawl to get a full list of available time series.
        # save every n minutes to avoid having to go redo...
        if not os.path.isfile(os.path.join(cfg.source.path.FRED.meta, "ids_freq_list.json")):
            logger.info("Crawling FRED.")
            crawl_fred(api_key=credentials.API_KEY_FED.key, nodes_to_visit=[0], sleep_time=cfg.source.api.FRED.sleep, rate_limit=cfg.source.api.FRED.limit)
        
        path = os.path.join(cfg.source.path.FRED.meta, "ids_freq_list.json")
        logger.info(f"Loading {path}.")

        # Download and save all time series. Start new files as the size gets too large 
        # more than 1000 files (https://softwareengineering.stackexchange.com/questions/254551/is-creating-and-writing-to-one-large-file-faster-than-creating-and-writing-to-ma). 

    #logger.info(len(time_series))
    #for ts in time_series[:5]:
    #    plt.plot(ts)
    #    plt.show()

if __name__ == "__main__":
    import yaml
    import sys
    from box import Box
    #import matplotlib.pyplot as plt  

    with open("GPTime/credentials.yml", "r") as ymlfile:
        credentials = Box(yaml.safe_load(ymlfile))
    #sys.path.append("")
    #sys.path.append("../..")
    # load yml file to dictionary
    #import os
    #print(os.getcwd())
    #credentials = yaml.load(open("GPTime/credentials.yml"))

    # access values from dictionary
    source_FRED(credentials.FRED, small_sample=False)
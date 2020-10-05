import fred
from typing import Dict, List, Tuple
import numpy as np
import json
import logging
import os
import sys
import time
import glob
sys.path.append("")

from GPTime.config import cfg

logger = logging.getLogger(__name__)

def crawl_fred(api_key:str, nodes_to_visit:List[int]=[0], sleep_time:int=60, rate_limit:int=100) -> None:
    """
    Crawling the FRED dataset. Saving all time series ids and metadata.
    """
    fred.key(api_key)
    file_number = 0
    tot_downloaded = 0
    num_nodes_visited = 0
    num_requests = 0
    list_json:List[Dict] = []
    num_files_written = 0
    curr_dir = f"dir{tot_downloaded // cfg.source.files_per_folder :04d}/"
    # initialize
    category_names = {}
    for node in nodes_to_visit:
        node_children = fred.children(node)
        for child in node_children["categories"]:
            category_names[child["id"]] = {"name": child["name"], "parent_id": child["parent_id"]}
    while nodes_to_visit:
        curr_node = nodes_to_visit.pop()
        #logger.info(f"Current node: {curr_node:>4}")
        try:
            children = fred.children(curr_node)
            num_requests += 1
            if children["categories"]:
                for child in children["categories"]:
                    nodes_to_visit.append(child["id"])
                    category_names[child["id"]] = {"name": child["name"], "parent_id": child["parent_id"]}

            seriess = fred.category_series(curr_node)["seriess"]
            num_requests += 1
            for ts in seriess:
                id_meta = ts
                id_meta["source"] = "FRED"
                id_meta["node_id"] = curr_node
                id_meta["category_name"] = category_names[curr_node]["name"]
                id_meta["parent_id"] = category_names[curr_node]["parent_id"]

                tot_downloaded += 1
                list_json.append(id_meta)
                if len(list_json) > cfg.source.samples_per_json:
                    filename = f"meta_{num_files_written:>06}.json"
                    if num_files_written % cfg.source.files_per_folder == 0:
                        curr_dir = f"dir{num_files_written // cfg.source.files_per_folder :04d}/"
                        os.makedirs(os.path.join(cfg.source.path.FRED.meta, curr_dir), exist_ok=True)
                    with open(os.path.join(*[cfg.source.path.FRED.meta, curr_dir, filename]), "w") as fp:
                        json.dump(list_json, fp, sort_keys=True, indent=4, separators=(",", ": "))
                        fp.close()
                    num_files_written += 1
                    list_json = []
                
            num_nodes_visited += 1

            if num_nodes_visited % 100 == 0:
                logger.info(f"Visited {num_nodes_visited:>5} nodes and currently have {tot_downloaded:>6} time series ids saved")
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

def download_ids(api_key:str, sleep_time:int=60, rate_limit:int=100) -> None:
    """
    Downloading all time series in the provided JSON file.
    """ 
    fred.key(api_key)
    #num_requests = 0
    tot_downloaded = 0
    num_files_written = 0
    list_json = []
    request_time_stamps = []
    #curr_dir = f"dir{tot_downloaded // cfg.source.files_per_folder :04d}/"

    dirs = glob.glob(cfg.source.path.FRED.meta + "/*")
    for d in dirs:
        file_names = glob.glob(d + "/*")
        for fname in file_names:
            with open(fname, "r") as fp:
                ids_meta = json.load(fp)
                fp.close()

            for id_meta in ids_meta:
                try:
                    observations = fred.observations(id_meta["id"])
                    request_time_stamps.append(time.time())
                    #num_requests += 1
                    ts = {
                        "id" : id_meta["id"],
                        "source" : id_meta["source"],
                        "node_id" : id_meta["node_id"],
                        "category_name" : id_meta["category_name"],
                        "parent_id" : id_meta["parent_id"],
                        "frequency" : id_meta["frequency"],
                        "observations" : [{"date" : obs["date"], "value" : obs["value"]} for obs in observations["observations"]]
                    }

                    tot_downloaded += 1
                    list_json.append(ts)
                    if len(list_json) > cfg.source.samples_per_json:
                        filename = f"raw_{num_files_written:>06}.json"
                        if num_files_written % cfg.source.files_per_folder == 0:
                            curr_dir = f"dir{num_files_written // cfg.source.files_per_folder :04d}/"
                            os.makedirs(os.path.join(cfg.source.path.FRED.raw, curr_dir), exist_ok=True)
                        with open(os.path.join(*[cfg.source.path.FRED.raw, curr_dir, filename]), "w") as fp:
                            json.dump(list_json, fp, sort_keys=True, indent=4, separators=(",", ": "))
                            fp.close()
                        
                        with open(os.path.join(cfg.source.path.FRED.meta, "ids_downloaded.txt"), "a") as fp:
                            for j in list_json:
                                fp.write(j["id"])
                                fp.write("\n")
                            fp.close()

                        num_files_written += 1
                        list_json = []

                    if tot_downloaded % 10000 == 0:
                        logger.info(f"Downloaded {tot_downloaded} time series.")

                except Exception as e:
                    logger.info(f"Failed to download id {id_meta['id']} from fname {fname}.")
                    logger.warning(e)

                if len(request_time_stamps) > rate_limit:
                    first = request_time_stamps.pop(0)
                    if time.time() - first < sleep_time:
                        #logger.info(f"Sleeping for {request_time_stamps[0]-first}.")
                        time.sleep(request_time_stamps[0]-first)
      
        logger.info(f"Written files in directory {d} and currently have {tot_downloaded:>6} time series saved")


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
                json.dump(ids_freqs, fp, sort_keys=True, indent=4, separators=(",", ": "))
        
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
        #if not os.path.isfile(os.path.join(cfg.source.path.FRED.meta, "ids_freq_list_test.json")):
        #logger.info("Crawling FRED.")
        #crawl_fred(api_key=credentials.API_KEY_FED.key, nodes_to_visit=[0], sleep_time=cfg.source.api.FRED.sleep, rate_limit=cfg.source.api.FRED.limit)
        #logger.info("Done crawling.")
        #path = os.path.join(cfg.source.path.FRED.meta, "ids_meta.json")

        logger.info(f"Downloading.")
        download_ids(api_key=credentials.API_KEY_FED.key, sleep_time=cfg.source.api.FRED.sleep, rate_limit=cfg.source.api.FRED.limit)


if __name__ == "__main__":
    import yaml
    import sys
    from box import Box

    with open("GPTime/credentials.yml", "r") as ymlfile:
        credentials = Box(yaml.safe_load(ymlfile))

    # access values from dictionary
    source_FRED(credentials.FRED, small_sample=False)

import fred
from typing import Dict, List, Tuple
import numpy as np
#import logging
import os
import sys
sys.path.append("")

from GPTime.config import cfg

#logger = logging.getLogger(__name__)

#import matplotlib.pyplot as plt  


def source_FRED(credentials, small_sample:bool=False) -> None:
    """
    Source the full FRED dataset and save to files. https://fred.stlouisfed.org/


    {'realtime_start': '2020-09-24',
    'realtime_end': '2020-09-24',
    'order_by': 'series_id',
    'sort_order': 'asc',
    'count': 12,
    'offset': 0,
    'limit': 1000,
    'seriess': [{'id': 'ATLSBUBEI',
    'realtime_start': '2020-09-24',
    'realtime_end': '2020-09-24',
    'title': 'Business Expectations Index (DISCONTINUED)',
    'observation_start': '2015-01-01',
    'observation_end': '2020-07-01',
    'frequency': 'Monthly',
    'frequency_short': 'M',
    'units': 'Index 2015-2018=100',
    'units_short': 'Index 2015-2018=100',
    'seasonal_adjustment': 'Not Seasonally Adjusted',
    'seasonal_adjustment_short': 'NSA',
    'last_updated': '2020-07-29 10:01:03-05',
    'popularity': 25,
    'group_popularity': 25,
    """
    # Setup directories f they do not exist
    # TODO

    # Create fred connection using api-key
    fred.key(credentials.API_KEY_FED.key)


    if small_sample:
        # Crawl to get a full list of available time series.
        ids_freqs:List[Tuple] = []
        for s in fred.category_series(33936)["seriess"]:
            ids_freqs.append((s["id"], s["frequency_short"]))
        np.save("GPTime/data/raw_data/meta/FRED/id_freq_list.npy", ids_freqs)
        
        #with(open("GPTime/data/raw_data/meta/FRED/id_freq_list.txt", "w")) as f:
        #    f.write("\n".join(ids))
        #    f.close()
        #print(ids)

        # Download and save all time series. Start new files as the size gets too large.
        time_series = []
        for id, freq in ids_freqs:
            observations = fred.observations(id)
            ts = []
            for obs in observations["observations"]:
                ts.append(float(obs["value"]))
            time_series.append(np.array(ts))
        np.save("GPTime/data/raw_data/small/FRED/FRED_test.npy", time_series)
        # Statistics of sourcing

        # Random dummy data for preprocessing
        X = np.random.rand(10000, 200)
        filename="dummy_test.npy"
        path = os.path.join(cfg.source.path.FRED, filename)
        np.save(path, X)
    else:
        pass
        # Crawl to get a full list of available time series.

        # Download and save all time series. Start new files as the size gets too large. 

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
    source_FRED(credentials.FRED, small_sample=True)
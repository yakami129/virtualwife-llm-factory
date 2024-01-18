import logging

import pandas as pd

logger = logging.getLogger(__name__)


class DatasetReader:
    def read(self, path: str):
        data = pd.read_excel(path, header=0)
        source_data_size = len(data)
        logger.info(
            f"=> reader dataset success # source_data_size:{source_data_size} #")
        return data

import logging
import os
from typing import Optional

from sklearn.model_selection import train_test_split
from analysis.reader import DatasetReader
from analysis.writer import DataSetWriterDriver

logger = logging.getLogger(__name__)


class DatasetComplexity:

    def calculate(self, data):
        data['target_column'] = data['question'].apply(len)
        return data


class DatasetSegmentation:

    def split(self, data, test_size: float, random_state: int, target_column: str):
        train_data = data
        train_target = data[target_column]
        x_train, x_test, y_train, y_test = train_test_split(
            train_data, train_target, test_size=test_size, random_state=random_state)
        return x_train, x_test, y_train, y_test


class DatasetProcess:
    dataset_writer_driver: DataSetWriterDriver
    dataset_reader: DatasetReader
    dataset_segmentation: DatasetSegmentation
    dataset_complexity: DatasetComplexity

    def __init__(self):
        self.dataset_writer_driver = DataSetWriterDriver()
        self.dataset_reader = DatasetReader()
        self.dataset_segmentation = DatasetSegmentation()
        self.dataset_complexity = DatasetComplexity()

    def run(self, work_dir_path: str, config: Optional[dict]) -> str:
        task_id = config["task_id"]
        finetune_config = config["finetune"]
        model_type = finetune_config["model_type"]
        dataset_config = config["dataset"]
        source_file_path = dataset_config["source_file_path"]
        test_size = float(dataset_config["ds_test_proportion"])
        random_state = int(dataset_config["random_state"])
        work_dataset_dir_path = work_dir_path + "/dataset/" + f"dataset-{task_id}"

        # 加载数据集
        data = self.dataset_reader.read(source_file_path)

        # 计算数据集复杂度
        data = self.dataset_complexity.calculate(data)

        # 分割训练集和测试集
        x_train, x_test, y_train, y_test = self.dataset_segmentation.split(data, test_size=test_size,
                                                                           random_state=random_state,
                                                                           target_column="target_column")
        logging.info(f"=> split dataset success # train_size:{len(x_train)}  test_size:{len(x_test)} #")

        # 如果目录不存在，则创建目录
        if not os.path.exists(work_dataset_dir_path):
            os.makedirs(work_dataset_dir_path, exist_ok=True)
            logging.info(f"=> create dir {work_dataset_dir_path} success")

        work_dataset_train_path = self.dataset_writer_driver.writer(type=model_type, data=x_train,
                                                                    filename="train_dataset",
                                                                    output_path=work_dataset_dir_path)
        work_dataset_test_path = self.dataset_writer_driver.writer(type=model_type, data=x_test,
                                                                   filename="test_dataset",
                                                                   output_path=work_dataset_dir_path)
        logging.info(f"=> writer dataset success # work_dataset_dir_path:{work_dataset_dir_path} #")
        return work_dataset_train_path, work_dataset_test_path


if __name__ == "__main__":
    work_dir_path = "/Users/zhangyajun/Documents/CodeWorkSpace/skyjun/virtualwife-llm-factory/output"
    source_file_path = "/Users/zhangyajun/Documents/CodeWorkSpace/skyjun/virtualwife-llm-factory/output/新增数据集_2024-01-09-23-16-10.xlsx"
    config = {
        "task_id": "1",
        "finetune": {
            "model_type": "llama"
        },
        "dataset": {
            "source_file_path": source_file_path,
            "ds_test_proportion": "0.1",
            "random_state": "16"
        }
    }
    process = DatasetProcess()
    process.run(work_dir_path, config)

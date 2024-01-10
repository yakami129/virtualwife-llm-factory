import json
import logging
from abc import abstractmethod, ABC

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class DatasetReader:
    def read(self, path: str):
        data = pd.read_excel(path, header=0)
        source_data_size = len(data)
        logger.info(
            f"=> reader dataset success # source_data_size:{source_data_size} #")
        return data


class DataSetWrite(ABC):

    @abstractmethod
    def writer(self, data, filename: str, output_path: str):
        pass


class RWKVDataSetWriter(DataSetWrite):

    def writer(self, data, filename: str, output_path: str):
        file_path = f"{output_path}/{filename}.jsonl"
        with open(file_path, 'w', encoding='utf-8') as file:
            for index, row in data.iterrows():
                text = f"User: {row['question']}\n\nAssistant: {row['answer']}\n\n"
                output_line = {"text": text}
                # 将每个对象写入文件的单独一行
                file.write(json.dumps(output_line, ensure_ascii=False) + "\n")

        return file_path


class DataSetWriterDriver:

    def writer(self, type: str, data, filename: str, output_path: str):
        return self.get_strategy(type).writer(data, filename, output_path)

    def get_strategy(self, type: str) -> DataSetWrite:
        if type == "rwkv":
            return RWKVDataSetWriter()
        else:
            raise ValueError("Unknown type")


class DataSetComplexity:

    def calculate(self, data):
        data['target_column'] = data['question'].apply(len)
        return data


class DataSetHandle:

    def split(self, data, test_size: float, random_state: int, target_column: str):
        train_data = data
        train_target = data[target_column]
        x_train, x_test, y_train, y_test = train_test_split(
            train_data, train_target, test_size=test_size, random_state=random_state)
        return x_train, x_test, y_train, y_test


if __name__ == "__main__":
    data_path = "/Users/zhangyajun/Documents/CodeWorkSpace/skyjun/virtualwife-llm-factory/output/新增数据集_2024-01-09-23-16-10.xlsx"
    output_path = "/Users/zhangyajun/Documents/CodeWorkSpace/skyjun/virtualwife-llm-factory/output"

    # 初始化读取工具
    reader = DatasetReader()
    data = reader.read(data_path)

    # 计算数据复杂度
    complexity = DataSetComplexity()
    data = complexity.calculate(data)

    print(data)

    # 分割数据
    handle = DataSetHandle()
    x_train, x_test, y_train, y_test = handle.split(data, test_size=0.01, random_state=16,
                                                    target_column="target_column")

    # 写入数据集
    writer_driver = DataSetWriterDriver()

    # 写入成rwkv格式
    writer_driver.writer(type="rwkv", data=x_train,
                         filename="train_dataset", output_path=output_path)
    writer_driver.writer(type="rwkv", data=x_test,
                         filename="test_dataset", output_path=output_path)

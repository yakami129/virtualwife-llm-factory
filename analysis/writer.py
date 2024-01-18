import json
from abc import abstractmethod, ABC


class DatasetWrite(ABC):

    @abstractmethod
    def writer(self, data, filename: str, output_path: str):
        pass


class RWKVDatasetWriter(DatasetWrite):
    def writer(self, data, filename: str, output_path: str):
        file_path = f"{output_path}/{filename}.jsonl"
        with open(file_path, 'w', encoding='utf-8') as file:
            for index, row in data.iterrows():
                text = f"User: {row['question']}\n\nAssistant: {row['answer']}\n\n"
                output_line = {"text": text}
                # 将每个对象写入文件的单独一行
                file.write(json.dumps(output_line, ensure_ascii=False) + "\n")
        return file_path


class LLaMADatasetWriter(DatasetWrite):

    def writer(self, data, filename: str, output_path: str):
        # 转换数据格式
        file_path = f"{output_path}/{filename}.json"
        output_data = []
        for index, row in data.iterrows():
            output_data.append({
                "instruction": row['question'],
                "input": "",
                "output": row['answer'],
                "system": ""
            })
        # 将转换后的数据保存为JSON文件
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(output_data, file, ensure_ascii=False, indent=2)
        return file_path

class DataSetWriterDriver:

    def writer(self, type: str, data, filename: str, output_path: str):
        return self.get_strategy(type).writer(data, filename, output_path)

    def get_strategy(self, type: str) -> DatasetWrite:
        if type == "rwkv":
            return RWKVDatasetWriter()
        elif type == "llama":
            return LLaMADatasetWriter()
        else:
            raise ValueError("Unknown type")

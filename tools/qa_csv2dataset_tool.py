import glob
import os
import time
from typing import Optional

import pandas as pd


class QAcsv2datasetTool:

    def __init__(self):
        return

    def run(self, args: Optional[dict], **kwargs):
        csv_dir_path = args["csv_dir_path"]
        output_path = args["output_path"]

        # 获取csv_dir_path文件夹下所有的csv文件
        csv_files = glob.glob(os.path.join(csv_dir_path, "*.csv"))

        # CSV文件转换为标准数据集
        datasets = []
        q = []
        a = []
        status = False
        for file in csv_files:
            csv = self.__read_csv(file, 0)
            self.__convert_json(a, csv, datasets, q, status)

        # 写入到标准数据集
        self.__write_excel(datasets, output_path)

    def __convert_json(self, a, csv, datasets, q, status):
        csv_len = len(csv)
        for index, row in csv.iterrows():
            role_name = row["角色名称"]
            if role_name == "Q" and status:
                dataset = {
                    "question": ",".join(q),
                    "answer": ",".join(a)
                }
                datasets.append(dataset)
                q = [row["内容"]]
                a = []
                status = False
            else:
                if role_name == "Q":
                    q.append(row["内容"])
                    status = False
                elif role_name == "A":
                    a.append(row["内容"])
                    status = True
                    if index == (csv_len - 1):
                        dataset = {
                            "question": ",".join(q),
                            "answer": ",".join(a)
                        }
                        datasets.append(dataset)

    def __read_csv(self, path: str, header: int):
        # 加载数据集
        data = pd.read_csv(path, header=header)
        source_data_size = len(data)
        print(
            f"=> reader dataset success # source_data_size:{source_data_size}#")
        return data

    def __write_excel(self, dataset, output_path: str):
        time_str = time.strftime("%Y-%m-%d-%H-%M-%S")

        # 创建一个空的DataFrame来保存结果
        output_df = pd.DataFrame(columns=['index', 'question', 'answer'])
        for index in range(len(dataset)):
            item = dataset[index]
            row = pd.DataFrame(
                [{'index': index, 'question': item['question'], 'answer': item['answer']}])
            output_df = pd.concat([output_df, row], ignore_index=True)

        # 写入excel
        print("======================== Table ======================== ")
        print(output_df)
        print(f"dataset size : {len(dataset)}")
        filename = "新增数据集_" + time_str + ".xlsx"
        output_df.to_excel(f"{output_path}/{filename}", index=False)


if __name__ == '__main__':
    args = {
        "csv_dir_path": "/Users/zhangyajun/Documents/CodeWorkSpace/skyjun/virtualwife-llm-factory/output/marked",
        "output_path": "/Users/zhangyajun/Documents/CodeWorkSpace/skyjun/virtualwife-llm-factory/output"
    }
    tool = QAcsv2datasetTool()
    tool.run(args)

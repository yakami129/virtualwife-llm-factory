# -*- coding: utf-8 -*-
import json
import logging
import os
import zipfile
from typing import Optional

import faiss
import numpy as np
import pandas as pd
from FlagEmbedding import FlagModel


class FlagModelFactory:
    embed_model: FlagModel

    def __init__(self, config: Optional[dict]):
        embed_model_path = config["embed_model_path"]
        # 向量检索模型
        self.embed_model = FlagModel(embed_model_path,
                                     query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                                     use_fp16=True)

    def get_embed_model(self):
        return self.embed_model


class GenEmbedIndex:
    flag_model_factory: FlagModelFactory

    def __init__(self, flag_model_factory: FlagModelFactory):
        self.flag_model_factory = flag_model_factory

    def generation(self, dataset_path: str, work_dir_path: str, config: Optional[dict]):

        embed_index_output_path = work_dir_path + "/embed_index.idx"
        dataset_output_path = work_dir_path + "/dataset.json"

        embed_index = faiss.IndexFlatL2(1024)  # build the index
        data = pd.read_excel(dataset_path, header=0)

        # 生成向量词表
        for index, row in data.iterrows():
            val_q = row["question"]
            embedding = np.array([self.flag_model_factory.get_embed_model().encode(val_q)])
            embed_index.add(embedding)
        faiss.write_index(embed_index, embed_index_output_path)

        # 生成json数据集
        dataset = []
        for index, row in data.iterrows():
            dataset.append({
                "question": row['question'],
                "answer": row['answer'],
            })
        with open(dataset_output_path, 'w', encoding='utf-8') as file:
            json.dump(dataset, file, ensure_ascii=False, indent=2)

        return embed_index_output_path, dataset_output_path

    def __create_work_dir(self, work_dir: str) -> str:
        if not os.path.exists(work_dir):
            os.makedirs(work_dir, exist_ok=True)
            logging.info(f"=> makedirs {work_dir} success")
        return work_dir


class GenRolePackageTool:
    flag_model_factory: FlagModelFactory
    gen_embed_index: GenEmbedIndex

    def __init__(self, args: Optional[dict]):
        self.flag_model_factory = FlagModelFactory(args)
        self.gen_embed_index = GenEmbedIndex(self.flag_model_factory)
        return

    def run(self, args: Optional[dict], **kwargs):
        role_name = args["role_name"]
        system_prompt_path = args["system_prompt_path"]
        dataset_path = args["dataset_path"]
        output_path = args["output_path"]

        # 生成对话风格embedIndex
        embed_index_output_path, dataset_output_path = self.gen_embed_index.generation(dataset_path, output_path, args)

        # 打包成zip
        # 创建的 ZIP 文件的路径
        zip_file = output_path + f"/{role_name}.zip"
        with zipfile.ZipFile(zip_file, 'w') as zipf:
            zipf.write(embed_index_output_path, arcname='embed_index.idx')
            zipf.write(dataset_output_path, arcname='dataset.json')
            zipf.write(system_prompt_path, arcname='system_prompt.txt')
        print(f"=> # gen role package # output:{zip_file}")

        # 删除临时文件
        os.remove(embed_index_output_path)
        os.remove(dataset_output_path)


if __name__ == '__main__':
    args = {
        "embed_model_path": "/Users/zhangyajun/Documents/CodeWorkSpace/skyjun/virtualwife-llm-factory/models/baai/bge-large-zh-v1.5",
        "role_name": "example",
        "system_prompt_path": "/Users/zhangyajun/Documents/CodeWorkSpace/skyjun/virtualwife-llm-factory/dataset/example_system_prompt.txt",
        "dataset_path": "/Users/zhangyajun/Documents/CodeWorkSpace/skyjun/virtualwife-llm-factory/dataset/example_dataset.xlsx",
        "output_path": "/Users/zhangyajun/Documents/CodeWorkSpace/skyjun/virtualwife-llm-factory/output",
    }

    gen_role_package_tool = GenRolePackageTool(args)
    gen_role_package_tool.run(args)

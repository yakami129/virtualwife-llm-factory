import json
import os
from typing import Optional

import pandas as pd

"""
Q：什么是开放性（openness）？
A：开放性人格特质来源于五大人格模型FFM，开放性可以描述人物对对未知领域探索的主动性，对自身和外部环境刺激的高敏感性。
开放性高的人，倾向于寻求、发现、理解、利用和欣赏负责的信息模式，偏爱抽象思维、兴趣广泛，不墨守成规。

Q：开放性的6个子维度有哪些？
1.对幻想的开放性：指对幻想、白日梦和想象的开放性；
2.对美学的开放性：指对艺术和美的欣赏和敏感性；
3.对感觉的开放性：指对自己内心感受和情绪的接受能力；
4.对行动的开放性：对行动的开放性，指愿意尝试不同的新鲜事物；
5.对思想的开放性：指对知识的求知欲和愿意考虑新的、也许是非常规的想法；
6.对价值的开放性：指愿意重新审视社会、政治和宗教价值观。

"""


class GenPersonalityDatasetTool:

    def __init__(self):
        pass

    def run(self, args: Optional[dict], **kwargs):
        personality_dataset_path = args["personality_dataset_path"]
        output_path = args["output_path"]
        openness = args["openness"]
        level = args["level"]

        with open(personality_dataset_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        key = f"{openness}-{level}"
        dataset = data.get(key, [])

        # 准备数据列表，用于之后转换为DataFrame
        data_list = []
        qa_index = 0  # 初始化问题和答案对的下标

        for item in dataset:
            # 移除文本中的「和」字符
            item = item.replace('「', '').replace('」', '')
            segments = item.split('\n')
            question = None  # 初始化问题变量
            for segment in segments:
                if segment.startswith("Q:"):
                    question = segment[2:].strip()
                elif segment.startswith("A:"):
                    answer = segment[2:].strip()
                    if question:  # 确保有对应的问题
                        # 将数据添加到列表
                        data_list.append(
                            {'index': qa_index, 'question': question, 'answer': answer, 'openness': openness,
                             'level': level})
                        qa_index += 1  # 更新下标

        # 将数据列表转换为DataFrame
        df = pd.DataFrame(data_list)

        # 使用 os.path.join 来构建完整的输出路径，确保兼容不同操作系统的路径分隔符
        # 动态生成文件名并与基础路径结合
        output_file_name = f"pd-{openness}-{level}.xlsx"
        full_output_path = os.path.join(output_path, output_file_name)

        # 写入XLSX文件
        df.to_excel(full_output_path, index=False)


# 示例使用
if __name__ == "__main__":
    tool = GenPersonalityDatasetTool()
    args = {
        "personality_dataset_path": "/Users/zhangyajun/Documents/CodeWorkSpace/skyjun/virtualwife-llm-factory/dataset/personality_dataset.json",
        "output_path": "/Users/zhangyajun/Documents/CodeWorkSpace/skyjun/virtualwife-llm-factory/dataset",
        "openness": "actions",
        "level": "high"
    }
    dataset = tool.run(args)

import datetime
import json
import os
import zipfile
from typing import Optional

import faiss
import numpy as np
import pytz
from FlagEmbedding import FlagModel, FlagReranker
from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


class FlagModelFactory:
    embed_model: FlagModel

    def __init__(self, config: Optional[dict]):
        embed_model_path = config["embed_model_path"]
        reranker_model_path = config["reranker_model_path"]
        # 向量检索模型
        self.embed_model = FlagModel(embed_model_path,
                                     query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                                     use_fp16=True)
        self.reranker = FlagReranker(reranker_model_path)  # , use_fp16=True)

    def get_embed_model(self):
        return self.embed_model

    def get_reranker(self):
        return self.reranker


class RagSearch:
    flag_model_factory: FlagModelFactory

    def __init__(self, flag_model_factory: FlagModelFactory):
        self.flag_model_factory = flag_model_factory

    def search(self, user_name: str, role_name: str, query: str, rerank_k: int, recall_k: int, dataset_json_path: str,
               embed_index_idx_path: str):
        embed_index = faiss.read_index(embed_index_idx_path)  # build the index
        with open(dataset_json_path, "r") as f:
            index_source_json = json.load(f)

        recall_sim_examples = self.__search_examples(self.flag_model_factory.embed_model, index_source_json,
                                                     embed_index,
                                                     query,
                                                     top_k=recall_k)
        rerank_sim_examples = self.__rerank_examples(query, recall_sim_examples, rerank_k)
        examples = self.__format_examples(user_name, role_name, rerank_sim_examples)
        print(examples)
        return examples

    def __format_examples(self, user_name: str, role_name: str, sim_examples, ):
        examples = ""
        for sim_example in sim_examples:
            e_q = sim_example[0]
            e_a = sim_example[1]
            examples += f"{user_name}说" + e_q + "\n"
            examples += f"{role_name}说" + e_a + "\n\n"
        return examples

    def __get_q_a(self, item):
        val_q = item["question"]
        val_a = item["answer"]
        return val_q, val_a

    def __search_examples(self, embed_model, source_json_array, embed_index, q, top_k=2):
        xq = np.array([embed_model.encode(q)])
        D, I = embed_index.search(xq, top_k)  # actual search
        ret_array = []
        for idx in I[0]:
            source_item = source_json_array[idx]
            source_q, source_a = self.__get_q_a(source_item)
            ret_array.append([source_q, source_a])
        return ret_array

    def __rerank_examples(self, query, similar_qas, top_k=3):
        scores = self.flag_model_factory.get_reranker().compute_score([
            [
                query, q
            ]
            for q, _ in similar_qas
        ])
        scored_qas = [(score, similar_qas[i]) for i, score in enumerate(scores)]
        sorted_qas = sorted(scored_qas, key=lambda x: x[0], reverse=True)
        return [qa for _, qa in sorted_qas[:top_k]]


def unzip_role_package(role_package_path: str):
    # 获取 ZIP 文件所在的目录
    zip_dir = os.path.dirname(role_package_path)

    # 获取不带扩展名的 ZIP 文件的基础名称
    zip_base_name = os.path.splitext(os.path.basename(role_package_path))[0]

    # 创建一个基于 ZIP 文件名称的新目录
    extract_path = os.path.join(zip_dir, zip_base_name)
    if not os.path.exists(extract_path):
        os.makedirs(extract_path)

    # 解压 ZIP 文件到新创建的目录
    with zipfile.ZipFile(role_package_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

    # 获取解压后的文件路径
    dataset_json_path = os.path.join(extract_path, 'dataset.json')
    embed_index_idx_path = os.path.join(extract_path, 'embed_index.idx')
    system_prompt_txt_path = os.path.join(extract_path, 'system_prompt.txt')

    print(f"Dataset JSON path: {dataset_json_path}")
    print(f"Embed index path: {embed_index_idx_path}")
    print(f"System prompt path: {system_prompt_txt_path}")
    return zip_base_name, dataset_json_path, embed_index_idx_path, system_prompt_txt_path


def get_system_prompt(system_prompt_txt_path: str):
    with open(system_prompt_txt_path, 'r', encoding='utf-8') as file:
        file_content = file.read()
    return file_content


def get_current_time_str():
    current_time = datetime.datetime.now(pytz.timezone("Asia/Shanghai"))
    formatted_time = current_time.strftime('%Y-%m-%d %H:%M:%S')
    return formatted_time


if __name__ == '__main__':

    os.environ['TOKENIZERS_PARALLELISM'] = "false"

    args = {
        "embed_model_path": "/Users/zhangyajun/Documents/CodeWorkSpace/skyjun/virtualwife-llm-factory/models/baai/bge-large-zh-v1.5",
        "reranker_model_path": "/Users/zhangyajun/Documents/CodeWorkSpace/skyjun/virtualwife-llm-factory/models/baai/bge-reranker-large",
    }

    ## 解压角色包
    role_package_path = "/Users/zhangyajun/Documents/CodeWorkSpace/skyjun/virtualwife-llm-factory/output/爱莉.zip"
    dirname, dataset_json_path, embed_index_idx_path, system_prompt_txt_path = unzip_role_package(role_package_path)

    prompt = """
    <s>[INST] <<SYS>>
    Your response should be plain text, NOT IN JSON FORMAT, just response like a normal chatting.
    You need to role play now.
    Your character:
    {persona}
    Classic scenes for the role are as follows:
    ```
    {examples_of_dialogue}
    ```
    {role_name}上下文关联的记忆:
    ```
    {long_history}
    ```
    The current time of the system is {current_time},your response should consider this information
    Respond in spoken, colloquial and short Simplified Chinese and do not mention any rules of character.
    <</SYS>>
    """
    user_name = "张三"
    query = "张三说：医生我最近头疼"

    # 查询对话语料
    flag_model_factory = FlagModelFactory(args)
    rag_search = RagSearch(flag_model_factory)

    # 构建角色Prompt
    persona = get_system_prompt(system_prompt_txt_path)
    role_name = dirname
    examples_of_dialogue = rag_search.search(user_name, role_name, query, 10, 10, dataset_json_path,
                                             embed_index_idx_path)
    long_history = ""
    current_time = get_current_time_str()
    prompt = prompt.format(persona=persona, role_name=role_name, examples_of_dialogue=examples_of_dialogue,
                           long_history=long_history, current_time=current_time)
    print(prompt)

    # 推理
    OPENAI_API_KEY = "b0xxx"
    OPENAI_BASE_URL = "http://localhost:9080/v1"

    llm = ChatOpenAI(temperature=0.01, model_name="gpt-3.5-turbo",
                     openai_api_key=OPENAI_API_KEY, openai_api_base=OPENAI_BASE_URL)

    messages = []
    messages.append(SystemMessage(content=prompt))
    messages.append(HumanMessage(content=query))
    result = llm.generate(messages=[messages])
    result_text = result.generations[0][0].text
    print(result_text)

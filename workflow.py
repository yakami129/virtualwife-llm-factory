import argparse
import logging
import os
import time
import traceback
from typing import Optional

import yaml

from analysis.process import DatasetProcess

# 加载analysis模块
dataset_process = DatasetProcess()


def init_logs(work_dir_path: str):
    time_str = time.strftime("%Y-%m-%d")
    logs_dir_path = create_work_dir(work_dir_path + "/logs")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # 创建用于写入日志文件的处理器
    file_handler = logging.FileHandler(logs_dir_path + "/" + f"workflow_{time_str}.log", mode='a')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    # 将处理器添加到logger
    logger.addHandler(file_handler)


def load_args():
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description="Workflow script.")
    parser.add_argument('--config_path',
                        type=str,
                        required=True,
                        help='指定你需要运行任务的配置文件路径')
    parser.add_argument('--run_task_id', type=str, help='直接运行任务的id')
    return parser.parse_args()


def get_work_dir(config_path: str) -> str:
    return os.path.basename(config_path)


def create_work_dir(work_dir: str) -> str:
    if not os.path.exists(work_dir):
        os.makedirs(work_dir, exist_ok=True)
        logging.info(f"=> makedirs {work_dir} success")
    return work_dir


def get_root_path(config_path: str) -> str:
    if not os.path.isabs(config_path):
        config_path = os.path.join(os.getcwd(), config_path)
    return config_path


def load_config(config_path: str) -> Optional[dict]:
    try:
        with open(config_path + "/config.yaml", 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)
            return data
    except Exception as e:
        logging.info(f"加载配置文件失败: {config_path} # error:{e}")


def run_analysis(config: Optional[dict]):
    source_file_path = config["dataset"]["source_file_path"]
    config["dataset"]["source_file_path"] = get_root_path(source_file_path)
    work_dataset_train_path, work_dataset_test_path = dataset_process.run(work_dir_path, item)
    return work_dataset_train_path, work_dataset_test_path


if __name__ == "__main__":

    logging.info("======================== start workflow ======================== ")

    # 加载workflow脚本参数
    args = load_args()
    root_config_path = get_root_path(args.config_path)

    # 加载workflow运行参数
    config = load_config(root_config_path)

    # 获取工作目录，并且创建工作目录
    time_str = time.strftime("%Y-%m-%d-%H-%M-%S")
    work_dir = get_work_dir(root_config_path)
    root_work_dir = get_root_path("output/" + work_dir + "_" + time_str)
    work_dir_path = create_work_dir(root_work_dir)

    # 配置日志
    init_logs(work_dir_path)
    logging.info(f"=> 1结果输出目录：{work_dir_path}")
    try:
        # TOOD 这里后续可以优化成任务队列形式
        for item in config:
            logging.info(f"run finetune task => task_id:{item['task_id']}")
            logging.info("======================== run data analysis ======================== ")
            work_dataset_train_path, work_dataset_test_path = run_analysis(item)
            logging.info("======================== run finetune ============================= ")
            ## TODO

    except Exception as e:
        logging.error(f"workflow执行失败: {e}")
        # 打印异常的堆栈信息
        traceback.print_exc()
    finally:
        logging.info("send email to .....")

    logging.info("======================== end workflow ======================== ")

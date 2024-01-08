import argparse
import logging
import os
import traceback
from typing import Optional

from tools.gen_dataset import GenerateDataset

logger = logging.getLogger(__name__)


def create_work_dir(work_dir: str) -> str:
    # 如果目录不存在，则创建目录
    if not os.path.exists(work_dir):
        os.makedirs(work_dir, exist_ok=True)
        logging.info(f"=> makedirs {work_dir} success")
    return work_dir


def get_root_path(config_path: str) -> str:
    # 检查路径是否为绝对路径
    if not os.path.isabs(config_path):
        # 如果是相对路径，转换为绝对路径
        config_path = os.path.join(os.getcwd(), config_path)
    return config_path


def load_args():
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description="tools script.")
    parser.add_argument('--tool_name',
                        type=str,
                        required=True,
                        help='工具名称，例如：gen_dataset_tool')
    parser.add_argument('--batch',
                        type=str,
                        default="5",
                        required=False,
                        help='执行批次')
    parser.add_argument('--batch_size',
                        type=str,
                        default="5",
                        required=False,
                        help='每批大小，控制chatgpt每次生成的数据量大小')
    parser.add_argument('--example_path',
                        type=str,
                        required=False,
                        help="示例文件位置，example.txt")
    parser.add_argument('--work_dir_path',
                        type=str,
                        default="output",
                        required=False,
                        help='工作目录')

    # azure openai 的配置
    parser.add_argument('--azure_openai_url',
                        type=str,
                        default=False,
                        help='azure_openai_url')
    parser.add_argument('--azure_openai_key',
                        type=str,
                        default=False,
                        help='azure_openai_key')
    parser.add_argument('--azure_deployment_name',
                        type=str,
                        default=False,
                        help='azure_deployment_name')

    return parser.parse_args()


def init_gen_dataset_tool(config: Optional[dict], work_dir_path: str):
    example_path = config["example_path"]
    with open(example_path, 'r') as file:
        file_content = file.read()
    config["example"] = file_content
    config["output_path"] = work_dir_path
    return GenerateDataset()


if __name__ == "__main__":

    try:

        # 加载workflow脚本参数
        args = load_args()

        # 构建config
        config = args.__dict__

        work_dir_path = get_root_path(args.work_dir_path)

        # 获取tool
        tool_name = config["tool_name"]
        if tool_name == 'gen_dataset_tool':
            tool = init_gen_dataset_tool(config, work_dir_path)
        else:
            raise ValueError(f"不存在 {tool_name} 工具")

        print("======================== start tools ======================== ")

        # 开始执行
        tool.run(config)

        print("======================== end tools ======================== ")

    except Exception as e:
        logging.error(f"tools执行失败: {e}")
        # 打印异常的堆栈信息
        traceback.print_exc()

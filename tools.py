import argparse
import logging
import os
import traceback
from typing import Optional

from tools.bilibili_video_to_audio_tool import BiliBiliVideo2AudioTool
from tools.bilibili_video_to_csv_tool import BiliBiliVideo2CsvTool
from tools.gen_dataset import GenerateDataset
from tools.gen_role_package_tool import GenRolePackageTool
from tools.qa_csv2dataset_tool import QAcsv2datasetTool

logger = logging.getLogger(__name__)

os.environ['TOKENIZERS_PARALLELISM'] = "false"


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
                        help='工具名称，例如：bilibili_video_to_csv')

    parser.add_argument('--video_ids',
                        type=str,
                        default="",
                        required=False,
                        help='bilibili_video_to_csv - B站视频id，多个视频使用`,`分开')
    parser.add_argument('--bilibili_cookie',
                        type=str,
                        default="",
                        required=False,
                        help='bilibili_video_to_csv - B站账号cookie')

    parser.add_argument('--csv_dir_path',
                        type=str,
                        default="",
                        required=False,
                        help='qa_csv2dataset_tool - csv文件夹地址')

    parser.add_argument('--embed_model_path',
                        type=str,
                        default="",
                        required=False,
                        help='gen_role_package_tool - embed模型地址')

    parser.add_argument('--role_name',
                        type=str,
                        default="",
                        required=False,
                        help='gen_role_package_tool - 角色名称')

    parser.add_argument('--system_prompt_path',
                        type=str,
                        default="",
                        required=False,
                        help='gen_role_package_tool - 角色核心定义Prompt地址，仅支持txt')

    parser.add_argument('--dataset_path',
                        type=str,
                        default="",
                        required=False,
                        help='gen_role_package_tool - 角色对话语料地址')

    parser.add_argument('--output_path',
                        type=str,
                        required=True,
                        help="结果输出地址")

    return parser.parse_args()


def init_gen_dataset_tool(config: Optional[dict], output_path: str):
    example_path = config["example_path"]
    with open(example_path, 'r') as file:
        file_content = file.read()
    config["example"] = file_content
    config["output_path"] = output_path
    return GenerateDataset()


def init_bilibili_video_to_csv_tool(config: Optional[dict], output_path: str):
    return BiliBiliVideo2CsvTool()


def init_bilibili_video_to_audio_tool(config: Optional[dict], output_path: str):
    return BiliBiliVideo2AudioTool()


def init_qa_csv2dataset_tool(config: Optional[dict], output_path: str):
    return QAcsv2datasetTool()


def init_gen_role_package_tool(config, output_path):
    return GenRolePackageTool(config)


if __name__ == "__main__":

    try:

        # 加载workflow脚本参数
        args = load_args()

        # 构建config
        config = args.__dict__

        output_path = get_root_path(args.output_path)

        # 获取tool
        tool_name = config["tool_name"]
        if tool_name == 'gen_dataset_tool':
            tool = init_gen_dataset_tool(config, output_path)
        elif tool_name == 'bilibili_video_to_csv':
            tool = init_bilibili_video_to_csv_tool(config, output_path)
        elif tool_name == "bilibili_video_to_audio":
            tool = init_bilibili_video_to_audio_tool(config, output_path)
        elif tool_name == 'qa_csv2dataset_tool':
            tool = init_qa_csv2dataset_tool(config, output_path)
        elif tool_name == 'gen_role_package_tool':
            tool = init_gen_role_package_tool(config, output_path)
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

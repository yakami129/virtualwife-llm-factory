import logging
import os
import subprocess
import time
from abc import abstractmethod, ABC
from typing import Optional

# 构建sft训练命令
llama_merge_lora_command: str = """
   CUDA_VISIBLE_DEVICES={cuda_visible_devices} && python {finetune_path}/src/export_model.py \
    --model_name_or_path {model_path} \
    --adapter_name_or_path {adapter_path} \
    --template default \
    --finetuning_type lora \
    --export_dir {output_path} \
    --export_size 2 \
    --export_legacy_format False
   """


class MergeLora:
    def merge(self, adapter_path: str, output_path: str, config: Optional[dict]):

        finetune_path = self.get_finetune_script_path() + '/framework/llama_factory'
        task_id = config["task_id"]
        finetune_config = config["finetune"]
        cuda_visible_devices = os.environ["CUDA_VISIBLE_DEVICES"]
        model_path = finetune_config["model_path"]
        model_dir_name = os.path.basename(model_path)
        model_name = f"{model_dir_name}-{task_id}"
        new_model_directory = output_path + "/models/" + model_name

        lora_command = llama_merge_lora_command.format(finetune_path=finetune_path,
                                                       model_path=model_path,
                                                       adapter_path=adapter_path,
                                                       output_path=new_model_directory,
                                                       cuda_visible_devices=cuda_visible_devices)

        # 使用Popen执行命令并实时捕获输出
        process = subprocess.Popen(lora_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True)

        # 实时打印输出到日志，同时减少CPU占用
        while True:
            output = process.stdout.readline()
            if output:
                logging.info(output.strip())
            elif process.poll() is not None:
                break
            # 适当的延时以减少CPU占用
            time.sleep(0.5)

        rc = process.poll()
        return new_model_directory

    def get_finetune_script_path(self):
        # 获取当前文件的路径
        current_file_path = __file__
        # 转换为绝对路径
        absolute_path = os.path.abspath(current_file_path)
        return os.path.dirname(os.path.abspath(current_file_path))

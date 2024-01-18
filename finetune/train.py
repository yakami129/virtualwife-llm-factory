import logging
import os
import subprocess
import time
from abc import abstractmethod, ABC
from typing import Optional

# 构建sft训练命令
llama_sft_command: str = """
   CUDA_VISIBLE_DEVICES={cuda_visible_devices} && python {finetune_path}/src/train_bash.py \
    --stage sft \
    --do_train \
    --model_name_or_path {model_path} \
    --dataset {dataset_name}} \
    --dataset_dir {dataset_dir}\
    --template default \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --output_dir {output_dir} \
    --overwrite_cache \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate {learning_rate} \
    --num_train_epochs {num_train_epochs} \
    --plot_loss 
    --fp16
   """


class BaseFinetuneCommand(ABC):
    @abstractmethod
    def parser(self, work_dir_path: str, config: Optional[dict]):
        pass


class LLamaSftFinetuneCommand(BaseFinetuneCommand):

    def parser(self, work_dir_path: str, config: Optional[dict]):

        finetune_path = self.get_finetune_script_path() + '/framework/llama_factory'
        finetune_config = config["finetune"]
        cuda_visible_devices = os.environ["CUDA_VISIBLE_DEVICES"]
        num_train_epochs = finetune_config["num_train_epochs"]
        learning_rate = finetune_config["learning_rate"]
        model_path = finetune_config["model_path"]
        dataset_config = config["dataset"]
        dataset_name = dataset_config["dataset_name"]
        dataset_dir = dataset_config["dataset_dir"]
        output_dir = work_dir_path + "/lora"

        lora_command = llama_sft_command.format(finetune_path=finetune_path, num_train_epochs=num_train_epochs,
                                                learning_rate=learning_rate,
                                                cuda_visible_devices=cuda_visible_devices,
                                                model_path=model_path, output_dir=output_dir,
                                                dataset_name=dataset_name, dataset_dir=dataset_dir)

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
        return output_dir

    def get_finetune_script_path(self):
        # 获取当前文件的路径
        current_file_path = __file__
        # 转换为绝对路径
        absolute_path = os.path.abspath(current_file_path)
        return os.path.dirname(absolute_path)

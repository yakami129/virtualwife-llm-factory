# virtualwife-llm-factory

virtualwife-llm-factory
是一个llm训练框架，用于解决虚拟角色训练入门门槛高的问题，该框架具备自动生成语料，性格塑造评估，基于国产llm微调训练等核心能力，目前还在开发，可以点个star~
关注一下

# 项目目录说明

```text
├── analysis            # 数据集自动分割工具
├── config              # 训练配置
├── dataset             # 数据集存储位置、数据集自动生成工具
├── evaluate            # 数据集评估工具（性格、人设等）、模型评分工具
├── finetune            # 微调训练框架
├── models              # 模型集
├── output              # 结构输出位置
└── tools               # 工具集
```

# 初始化

- 安装机器学习相关依赖
    - 注意：需要确认操作系统和cuda版本，本示例是以linux系统为准，具体请查阅 https://pytorch.org/

```shell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

- 安装项目依赖

```shell
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple some-package
```

# tools.py 使用示例

- 查询tools参数明细

```shell
python tools.py -h
```

## 示例一：bilibili_video_to_csv_tool

- 工具说明：
    - 作用：从B站下载视频，并且将视频中的内容转换为csv文件，输出的csv文件需要人工标注对话所属角色
    - 用途：准备角色训练集
- 使用示例

```shell
 python tools.py --tool_name bilibili_video_to_csv \
 --video_ids BV1AC4y1U77e \
 --bilibili_cookie "<Your cookie>" \
 --output_path output
```

## 示例二：qa_csv2dataset_tool

- 工具说明：
    - 作用：将标记好的csv文件转换为标准数据集格式，方便fine-tuning
    - 用途：准备角色训练集
- 使用示例

```shell
python tools.py --tool_name qa_csv2dataset_tool \
--csv_dir_path output/example \
--output_path output
```

## 示例二：gen_role_package_tool

- 工具说明：
    - 作用：生成角色安装包
    - 用途：将角色Prompt和对话语料打包，结合RAG技术使用，具体使用方式请参考example/run_role_package.py，未来virtualwife会支持直接导入角色安装包
- 使用示例

```shell
python tools.py --tool_name gen_role_package_tool \
--embed_model_path models/baai/bge-large-zh-v1.5 \
--role_name example \
--system_prompt_path dataset/example_system_prompt.txt \
--dataset_path dataset/example_dataset.xlsx \
--output_path output
```

# workflow.py 使用示例（开发中，暂不可用）

- 一键微调模型

```shell
export CUDA_VISIBLE_DEVICES=6 && python workflow.py --config_path config/example
```

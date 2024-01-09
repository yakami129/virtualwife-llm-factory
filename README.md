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
    - 作用：从B站下载视频，并且将视频中的内容转换为csv文件
    - 用途：准备角色训练集
- 使用示例

```shell
 python tools.py --tool_name bilibili_video_to_csv \
 --video_ids BV1AC4y1U77e \
 --bilibili_cookie "<Your cookie>" \
 --output_path output
```

# workflow.py 使用示例

TODO 开发中

import pandas as pd
import json
from typing import Optional


class GenIndex_1_9b_DatasetTool:

    def __init__(self):
        pass

    def run(self, args: Optional[dict] = None, **kwargs):

        pd_dataset_path = args.get('pd_dataset_path')
        output_path = args.get('output_path')

        if not pd_dataset_path or not output_path:
            raise ValueError("pd_dataset_path and output_path must be provided")

        # 读取xlsx文件
        df = pd.read_excel(pd_dataset_path, engine='openpyxl')

        # 数据转换
        result = []
        for _, row in df.iterrows():
            item = {
                "system": f"请按照{row['openness']}-{row['level']}的风格回复我，内容请不要超过15个字符。",
                "human": row['question'],
                "assistant": row['answer']
            }
            result.append(item)

        # 输出JSON
        json_output = json.dumps(result, ensure_ascii=False, indent=2)

        # 将JSON输出到文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(json_output)

        print(f"Data has been written to {output_path}")


if __name__ == "__main__":
    # 示例：假设xlsx文件路径为"./data.xlsx"，输出路径为"./output.json"
    args = {
        "pd_dataset_path": "/Users/zhangyajun/Documents/CodeWorkSpace/skyjun/virtualwife-llm-factory/output/pd/pd-dataset.xlsx",
        "output_path": "/Users/zhangyajun/Documents/CodeWorkSpace/skyjun/virtualwife-llm-factory/output/dataset/index_1_9b/output.json"
    }
    tool = GenIndex_1_9b_DatasetTool()
    tool.run(args=args)

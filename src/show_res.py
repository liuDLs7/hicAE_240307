import pandas as pd
import json

result_path = '../PC_datas/results.json'

with open(result_path, 'r') as f:
    res = json.load(f)

# 将JSON数据加载到DataFrame中，并进行行列转置
df_transposed = pd.DataFrame(res)

# 将DataFrame转换为Markdown格式的表格
table = df_transposed.to_markdown()

# 打印输出表格
print(table)

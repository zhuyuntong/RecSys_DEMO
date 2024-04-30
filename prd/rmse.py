import sys
import pandas as pd
import numpy as np

# 检查命令行参数
if len(sys.argv) != 3:
    print("Usage: python script.py <test_data_file> <output_file>")
    sys.exit(1)

# 从命令行参数读取文件路径
test_data_file = sys.argv[1]
output_file = sys.argv[2]

# 读取CSV文件
test_data = pd.read_csv(test_data_file)
predictions = pd.read_csv(output_file)

# 合并两个dataframe，基于'user_id'和'business_id'
merged_data = pd.merge(test_data, predictions, on=['user_id', 'business_id'])

# 计算RMSE
rmse = np.sqrt(((merged_data['stars'] - merged_data['prediction']) ** 2).mean())
print(f'The RMSE between predictions and actual stars is: {rmse}')

import pandas as pd
import numpy as np
import re

# 尝试使用UTF-8-SIG编码读取文件
data = pd.read_csv('restaurant_data.csv', encoding='UTF-8-SIG')
pd.set_option('display.max_columns', None)  # 显示所有列
# 定义一个函数来检查是否包含乱码字符
def has_garbled_chars(text):
    if isinstance(text, str):
        # 检查是否包含非ASCII字符
        return bool(re.search(r'[^\x00-\x7F]+', text))
    return False

# 检查每行数据是否包含乱码字符，如果有则删除该行
def remove_garbled_rows(df):
    for column in df.columns:
        df = df[~df[column].apply(has_garbled_chars)]
    return df

# 应用函数删除包含乱码字符的行
cleaned_data = remove_garbled_rows(data)

# 打印前50行查看清理后的数据
print(cleaned_data.head(50))

# 保存清理后的数据到新的CSV文件
cleaned_data.to_csv('cleaned_restaurant_data.csv', index=False, encoding='UTF-8-SIG')
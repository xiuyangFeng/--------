import pickle  # 导入Pickle模块，用于序列化和反序列化Python对象

# 打开并读取pickle文件
with open('pv_model.pkl', 'rb') as file:
    """
    使用pickle.load()函数从文件中加载数据
    'rb'模式表示以二进制只读方式打开文件
    """
    data = pickle.load(file)

# 打印加载的数据
print(data)
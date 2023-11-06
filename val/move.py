import os
import shutil

source_dir = '/path/to/source_directory'
target_dir = '/home/special/user/jijun/Purified/cifar10'

# 确保目标目录存在
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# 获取源目录中的所有文件
files = os.listdir(source_dir)

for filename in files:
    # 获取文件的首字符
    first_char = filename[0]

    # 构建目标目录路径
    destination_dir = os.path.join(target_dir, first_char)

    # 如果目标目录不存在，创建它
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # 构建源文件路径和目标文件路径
    source_file = os.path.join(source_dir, filename)
    destination_file = os.path.join(destination_dir, filename)

    # 移动文件
    shutil.move(source_file, destination_file)

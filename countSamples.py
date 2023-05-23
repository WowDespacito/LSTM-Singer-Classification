import os
from collections import defaultdict
from tqdm import tqdm

def get_mp3_files(directory):
    """获取目录下所有mp3文件的文件名"""
    mp3_files = []
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
        if os.path.isdir(file_path):
            mp3_files.extend(get_mp3_files(file_path))
        elif file_name.endswith(".mp3"):
            mp3_files.append(file_path)
    return mp3_files

def calculate_directory_mp3_count(directory):
    """计算目录下每个子目录内的mp3文件数量"""
    directory_mp3_count = {}
    for directory_name, subdirectories, files in tqdm(os.walk(directory), desc="Walking through directories"):
        mp3_files = [file_name for file_name in files if file_name.endswith(".mp3")]
        mp3_count = len(mp3_files)
        directory_mp3_count[directory_name] = mp3_count
    return directory_mp3_count

directory = "D:\Temporary files\graduateDesign\data"
directory_mp3_count = calculate_directory_mp3_count(directory)
for directory_name, mp3_count in directory_mp3_count.items():
    print("{}目录下有{}个MP3文件".format(directory_name, mp3_count))
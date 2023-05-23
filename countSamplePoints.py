from pydub import AudioSegment
import os
from collections import defaultdict
from tqdm import tqdm

def get_audio_samples(file_path):
    """获取音频文件所有采样点"""
    audio = AudioSegment.from_file(file_path)
    return audio.get_array_of_samples()

def get_mp3_files(directory):
    """获取目录下所有MP3文件的文件名"""
    mp3_files = []
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
        if os.path.isdir(file_path):
            mp3_files.extend(get_mp3_files(file_path))
        elif file_name.endswith(".mp3"):
            mp3_files.append(file_path)
    return mp3_files

def calculate_samples(mp3_files):
    """计算目录下所有mp3文件的采样点数量"""
    sample_count = 0
    for file_path in tqdm(mp3_files, desc="处理MP3文件中"):
        samples = get_audio_samples(file_path)
        sample_count += len(samples)
    return sample_count

def calculate_directory_samples(directory):
    """计算目录下每个子目录中所有MP3文件的采样点数量"""
    directory_samples = defaultdict(int)
    for directory_name, subdirectories, files in tqdm(os.walk(directory), desc="遍历目录中"):
        mp3_files = [os.path.join(directory_name, file_name) for file_name in files if file_name.endswith(".mp3")]
        directory_sample_count = calculate_samples(mp3_files)
        directory_samples[directory_name] = directory_sample_count
    return directory_samples

directory = "D:\Temporary files\graduateDesign\data" # 替换成你要遍历的目录路径
directory_samples = calculate_directory_samples(directory)
for directory_name, sample_count in directory_samples.items():
    print("{}目录下所有MP3文件的采样点数量为：{}".format(directory_name, sample_count))

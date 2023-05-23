from pydub import AudioSegment
import os
from collections import defaultdict
from tqdm import tqdm

def get_audio_duration(file_path):
    """获取音频文件时长，单位为毫秒"""
    audio = AudioSegment.from_file(file_path)
    return audio.duration_seconds * 1000

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

def calculate_total_duration(mp3_files):
    """计算目录下所有mp3文件的总时长，单位为毫秒"""
    total_duration = 0
    for file_path in tqdm(mp3_files, desc="Processing mp3 files"):
        file_duration = get_audio_duration(file_path)
        total_duration += file_duration
    return total_duration

def calculate_directory_durations(directory):
    """计算目录下每个子目录内所有mp3文件的总时长，单位为毫秒"""
    directory_durations = defaultdict(int)
    for directory_name, subdirectories, files in tqdm(os.walk(directory), desc="Walking through directories"):
        mp3_files = [os.path.join(directory_name, file_name) for file_name in files if file_name.endswith(".mp3")]
        directory_duration = calculate_total_duration(mp3_files)
        directory_durations[directory_name] = directory_duration
    return directory_durations        

directory = "D:\Temporary files\graduateDesign\data"
directory_durations = calculate_directory_durations(directory)
for directory_name, duration in directory_durations.items():
    print("{}目录下所有MP3文件的总时长为：{}毫秒".format(directory_name, duration))
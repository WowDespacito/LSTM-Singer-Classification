##本脚本用于分割数据集
#--注意--#
##使用前需要安装ffmpeg，或将ffmpeg.exe和ffprobe.exe与该文件放置在同一目录下


from pydub import AudioSegment
from pydub.utils import make_chunks
import os

##path为文件路径；size为每个样本的时长，单位s；exFormat为最终样本的格式，默认为mp3
def audioSegment(path, size, exFormat="mp3"):
    size = size * 1000
    name = os.path.basename(path)
    fileFormat = name.split(".")[1]
    name = name.split(".")[0]
    audio = AudioSegment.from_file(path, format=fileFormat)
    chunks = make_chunks(audio, size)
    for i, chunk in enumerate(chunks):
        chunk_name = name + "_{0}.".format(i) + exFormat
        dir = os.path.dirname("../../data_duty/") + "/" + os.path.basename(os.path.dirname(path))
        if not os.path.exists(dir):
            os.makedirs(dir)
        chunk.export(dir + '/' + chunk_name, format=exFormat)
        print("输出：", dir + '/' + chunk_name)


if __name__ == "__main__":
    file_list = []
    for root, dir, files in os.walk("../../data_back/vocals"):
        for file in files:
            file_list.append(root+'/'+file)
    for file in file_list:
        print("正在处理：", file)
        audioSegment(file, 8)

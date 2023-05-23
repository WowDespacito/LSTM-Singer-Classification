import os

def RenameFile(path, index:str):
    file_name = os.path.basename(path)
    while (len(index) != 4):
        index = '0' + index
    file_name = file_name.split(' ')[0] + '.' + index + '.mp3'
    print(file_name)
    os.rename(path, os.path.dirname(path)+'/'+file_name)


if __name__ == '__main__':
    file_list = []
    for root, dirs, files in os.walk('../data'):
        for file in files:
            file_list.append(root+'/'+ file)
    i = 0
    cur_name = 'SunYanzi'
    for file in file_list:
        name = os.path.basename(file).split(' ')[0]
        if cur_name != name:
            i = 0
            cur_name = name
        RenameFile(file, str(i))
        i += 1



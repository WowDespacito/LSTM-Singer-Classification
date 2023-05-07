from sklearn.model_selection import train_test_split
import os
from shutil import copyfile
import random



file_list = []
for root, dir, files in os.walk('../../data'):
    for file in files:
        file_list.append(root + '/' + file)
    if file_list == []:
        continue
    num = len(file_list) - 472
    names = random.sample(file_list, num)
    for name in names:
        os.remove(name)
    file_list = []
    names = []

file_list = []
label_list = []
for root, dir, files in os.walk('D:\Temporary files\graduateDesign\data'):
    for file in files:
        file_list.append(root + '/' + file)
        label_list.append(file.split('(')[0])
X_train, X_test, y_train, y_test = train_test_split(file_list, label_list, test_size=0.3)
for file in X_train:
    copyfile(file, "../data/train_data/"+os.path.basename(file))
for file in X_test:
    copyfile(file, "../data/validation_data/"+os.path.basename(file))
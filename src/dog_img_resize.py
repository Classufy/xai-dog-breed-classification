import csv
from typing import DefaultDict
from PIL import Image
import glob

dic = DefaultDict()
target = set()

data_path = './dog_data' 
with open(f'{data_path}/labels.csv', 'r') as data:
    for line in csv.reader(data): 
        dic[line[0]] = line[1]
        target.add(line[1])

target = list(target)

dir_path = [f'{data_path}/test', f'{data_path}/train']

for dir in dir_path:
    resized_path = dir.split('/')[-1]
    files = glob.glob(f'{dir}/*.jpg')
    # print(resized_path)
    for file in files:
        img = Image.open(file)
        file = file.split('/')[-1]
        # print(f'{data_path}/{resized_path}_resized/{file}')
        img = img.resize((128, 128))
        img.save(f'{data_path}/{resized_path}_resized/{file}')

# for i in range(3213, 4278):
#     img = Image.open(f'./data/{animal[2]}/{animal[2]}_{i}.jpg')
#     # img = Image.open(f'./data/{animal[2]}/{animal[2]}_{i}.jpg')
#     img = img.resize((128, 128))
#     img.save(f'./data/{animal[2]}_resize/{animal[2]}_{i}.jpg')

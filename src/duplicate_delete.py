import imagehash
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np  # linear algebra
import os
import glob

target = [
    'beagle', 'cocker_spaniel', 'maltese', 
    'pomeranian', 'poodle', 'golden_retriever',
    'samoyed', 'shih_tzu', 'white_terrier', 'pekinese']

for breed in target:
    jpg_list = glob.glob(f'./data/{breed}/*.jpg')
    hash_set = set()
    for path in jpg_list:
        img = Image.open(path)
        hash = imagehash.phash(img)
        if hash in hash_set:
            if os.path.isfile(path):
                os.remove(path)
                print(f'Delete {path}')
        else: 
            hash_set.add(hash)

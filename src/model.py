from PIL import Image
import numpy as np
import glob
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
from typing import DefaultDict
import csv

dir_path = './dog_data'
image_dir = [f'{dir_path}/test_resize', f'{dir_path}/train_resize']

dic = DefaultDict()
image_target = set()

data_path = './dog_data'
with open(f'{data_path}/labels.csv', 'r') as data:
    for filename, breed in csv.reader(data):
        dic[f'{filename}.jpg'] = breed
        image_target.add(breed)

target = list(image_target)

image_target = DefaultDict()

for i, breed in enumerate(target):
    image_target[breed] = i    

X = []
y = []
dic

jpg_list = glob.glob(f'{dir_path}/train_resized/*.jpg')
for img in jpg_list:
    fname = img.split('/')[-1]
    img = Image.open(img)
    img = img.convert('RGB')
    X.append(np.asarray(img))
    y.append(image_target[dic[fname]])

X = np.array(X)
y = np.array(y)

X.shape
y.shape

df_y = pd.Series(y)
df_y.value_counts() / len(df_y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=1, test_size=0.2, stratify=y)


# X_train = X_train.reshape(-1, 128 * 128 * 3)
# X_test = X_test.reshape(-1, 128 * 128 * 3)
X_train = X_train / 255
X_test = X_test / 255

X_train.shape
y_train.shape

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, random_state=1, test_size=0.2)

model = tf.keras.Sequential([
  tf.keras.layers.Conv2D((128), (3, 3), activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D((128), (3, 3), activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D((128), (3, 3), activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(len(image_target), activation='softmax')
])

# optimizer = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, bete_2=0.999)

model.compile(
    optimizer='adam',
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

early_stopping_cb = tf.keras.callbacks.EarlyStopping(
    patience=10, restore_best_weights=True)

history = model.fit(
    X_train, y_train, epochs=100, 
    validation_data=(X_val, y_val),
    callbacks=[early_stopping_cb])

model.evaluate(X_test, y_test)

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()

model.summary()

# model.save('./saved_model.h5')


prob = model.predict(X_test)
y_pred = prob.argmax(axis=-1)

### 결과 확인

dic = {
    '0-0':0,
    '0-1':0,
    '0-2':0,
    '1-0':0,
    '1-1':0,
    '1-2':0,
    '2-0':0,
    '2-1':0,
    '2-2':0
}
for i in range(len(y_pred)):
    pred, ans = y_pred[i], y_test[i]
    # if y_pred[i] != y_test[i]: 
    #     wrong.append((image_target[pred], image_target[ans]))
        # cnt[ans] += 1
    dic[f'{pred}-{ans}'] += 1
print('예측 결과 - 출력결과')
for k in dic:
    pred = int(k[0])
    ans = int(k[2])
    print(f'{image_target[pred]} - {image_target[ans]} : {dic[k]}')



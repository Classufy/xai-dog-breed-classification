from PIL import Image
import numpy as np
import glob
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd

image_dir = ['./data/cheetah_resize', './data/leopard_resize', './data/tiger_resize']
image_target = ['cheetah', 'leopard', 'tiger']

X = []
y = []

for i, dir in enumerate(image_dir):
    jpg_list = glob.glob(f'{dir}/*.jpg')
    for img in jpg_list:
        img = Image.open(img)
        img = img.convert('RGB')
        X.append(np.asarray(img))        
    # data.append(jpg_list)
    for _ in range(len(jpg_list)): y.append(i)

X = np.array(X)
y = np.array(y)
len(X)

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

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()

model.evaluate(X_test, y_test)
model.summary()

model.save('./saved_model.h5')


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



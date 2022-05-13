from PIL import Image
import numpy as np
import glob
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd


data_path = './data'

target = [
    'beagle', 'cocker_spaniel', 'golden_retriever',
    'maltese', 'pekinese', 'pomeranian', 'poodle',
    'samoyed', 'shih_tzu', 'white_terrier']

image_dir = [f'{data_path}/{breed}' for breed in target]


X = []
y = []

for i, breed in enumerate(target):
    jpg_list = glob.glob(f'{data_path}/{breed}/*.jpg')
    for img in jpg_list:
        img = Image.open(img)
        img = img.resize((224, 224))
        img = img.convert('RGB')
        X.append(np.array(img))
        y.append(i) 
    print(f'{breed} * {len(jpg_list)} images are converted')

X = np.array(X)
y = np.array(y)

X.shape
y.shape

unique, count = np.unique(y, return_counts=True)
print(np.asarray((unique, count)))

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

base_model = tf.keras.applications.xception.Xception(
    weights='imagenet', include_top=False)

avg = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
output = tf.keras.layers.Dense(len(target), activation='softmax')(avg)
model = tf.keras.Model(inputs=base_model.input, outputs=output)

# 가중치들 다 고정하고 전이 학습
for layer in base_model.layers:
    layer.trainable = False

len(model.trainable_weights)

optimizer = tf.keras.optimizers.SGD(learning_rate=0.2, momentum=0.9, decay=0.01)
early_stopping_cb = tf.keras.callbacks.EarlyStopping(
    patience=2, restore_best_weights=True)

model.compile(
    optimizer=optimizer,
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])


history = model.fit(
    X_train, y_train, epochs=5,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping_cb])

# 마지막 5 레이어의 가중치만 열고 다시 학습
for layer in base_model.layers[-5:]:
    layer.trainable = True

len(model.trainable_weights)

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, decay=0.001)

model.compile(
    optimizer=optimizer,
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

history = model.fit(
    X_train, y_train, epochs=100,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping_cb])

print('test loss / accuracy')
print(model.evaluate(X_test, y_test))

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()


model.save('./saved_model.h5')




from PIL import Image
import numpy as np
import glob
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd


data_path = './data'

target = [
    'beagle', 'cocker_spaniel',
    'maltese', 'pomeranian', 'poodle',
    'samoyed', 'shih_tzu', 'white_terrier']

image_dir = [f'{data_path}/{breed}' for breed in target]


X = []
y = []

for i, breed in enumerate(target):
    jpg_list = glob.glob(f'{data_path}/{breed}/*.jpg')
    for img in jpg_list:
        img = Image.open(img)
        img = img.resize((128, 128))
        img = img.convert('RGB')
        X.append(np.array(img))
        y.append(i) 

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

model = tf.keras.Sequential([
  tf.keras.layers.Conv2D((128), (3, 3), activation='relu', input_shape=(128, 128, 3)),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D((128), (3, 3), activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Dense(len(target), activation='softmax')
])

# optimizer = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, bete_2=0.999)

model.compile(
    optimizer='adam',
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

early_stopping_cb = tf.keras.callbacks.EarlyStopping(
    patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train, epochs=100, 
    validation_data=(X_val, y_val),
    callbacks=[early_stopping_cb])

model.evaluate(X_test, y_test)

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()


model.save('./saved_model.h5')




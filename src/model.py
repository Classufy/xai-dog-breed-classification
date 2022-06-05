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


x = []
y = []

for i, breed in enumerate(target):
    jpg_list = glob.glob(f'{data_path}/{breed}/*.jpg')
    for img in jpg_list:
        img = Image.open(img)
        img = img.resize((224, 224))
        img = img.convert('RGB')
        x.append(np.array(img))
        y.append(i) 
    print(f'{breed} * {len(jpg_list)} images are converted')

x = np.array(x)
y = np.array(y)

x.shape
y.shape

unique, count = np.unique(y, return_counts=True)
print(np.asarray((unique, count)))

<<<<<<< HEAD
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=1, test_size=0.2, stratify=y)


=======
# X = X / 255.

x_train, x_test, y_train, y_test = train_test_split(
    X, y, random_state=1, test_size=0.2, stratify=y)

X = []
y = []

# X_train = X_train.reshape(-1, 128 * 128 * 3)
# X_test = X_test.reshape(-1, 128 * 128 * 3)
>>>>>>> 179a14741ac26badd7139201242217fad445806e
x_train = x_train / 255
x_test = x_test / 255
x_train.shape
y_train.shape

x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, random_state=1, test_size=0.2)
<<<<<<< HEAD

print('splited')
df = pd.DataFrame(columns=['test_loss', 'test_accuracy', 'train_loss', 'train_accuracy'])

for first_lr in [0.4, 0.3, 0.2]:
    for second_lr in [0.1, 0.05, 0.01]:
        for trainable_layers in [3, 5, 7]:
            base_model = tf.keras.applications.xception.Xception(
                weights='imagenet', include_top=False, input_shape=(224, 224, 3))

            avg = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
            output = tf.keras.layers.Dense(len(target), activation='softmax')(avg)
            model = tf.keras.Model(inputs=base_model.input, outputs=output)

            # 가중치들 다 고정하고 전이 학습
            for layer in base_model.layers:
                layer.trainable = False

            len(model.trainable_weights)

            optimizer = tf.keras.optimizers.SGD(learning_rate=first_lr, momentum=0.9, decay=0.01)
            early_stopping_cb = tf.keras.callbacks.EarlyStopping(
                patience=2, restore_best_weights=True)

            
            model.compile(
                optimizer=optimizer,
                loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False),
                metrics=['accuracy'])


            history = model.fit(
                x_train, y_train, epochs=7,
                validation_data=(x_val, y_val),
                callbacks=[early_stopping_cb])

            print('test loss / accuracy')
            test_loss, test_accuracy = model.evaluate(x_test, y_test)
            print(f'{test_loss}, {test_accuracy}')

            # 마지막 5 레이어의 가중치만 열고 다시 학습
            for layer in base_model.layers[-trainable_layers:]:
                layer.trainable = True


            len(model.trainable_weights)

            optimizer = tf.keras.optimizers.SGD(learning_rate=second_lr, momentum=0.9, decay=0.001)

            model.compile(
                optimizer=optimizer,
                loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False),
                metrics=['accuracy'])

            history = model.fit(
                x_train, y_train, epochs=100,
                validation_data=(x_val, y_val),
                callbacks=[early_stopping_cb])


            print('test loss / accuracy')
            test_loss, test_accuracy = model.evaluate(x_test, y_test)
            print(f'{test_loss}, {test_accuracy}')
            
            print('train loss / accuracy')
            train_loss, train_accuracy = model.evaluate(x_train, y_train)
            print(f'{train_loss}, {train_accuracy}')

            append = pd.DataFrame({
                'test_loss': [test_loss], 
                'test_accuracy': [test_accuracy],
                'train_loss': [train_loss], 
                'train_accuracy': [train_accuracy]
            })
            df = pd.concat([df, append], ignore_index=True)
            

        
df.to_csv('../assets/output.csv')

# pd.DataFrame(history.history).plot(figsize=(8, 5))
# plt.grid(True)
# plt.gca().set_ylim(0, 1)
# plt.show()


model.save('./assets/best_model.h5')
=======

print('splited')
df = pd.DataFrame(columns=['test_loss', 'test_accuracy', 'train_loss', 'train_accuracy'])

for first_lr in [0.4, 0.3, 0.2]:
    for second_lr in [0.1, 0.05, 0.01]:
        for train_layers in [3, 5, 7]:
            base_model = tf.keras.applications.xception.Xception(
                weights='imagenet', include_top=False, input_shape=(224, 224, 3))

            avg = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
            output = tf.keras.layers.Dense(len(target), activation='softmax')(avg)
            model = tf.keras.Model(inputs=base_model.input, outputs=output)

            # 가중치들 다 고정하고 전이 학습
            for layer in base_model.layers:
                layer.trainable = False

            len(model.trainable_weights)

            optimizer = tf.keras.optimizers.SGD(learning_rate=first_lr, momentum=0.9, decay=0.01)
            early_stopping_cb = tf.keras.callbacks.EarlyStopping(
                patience=2, restore_best_weights=True)

            
            model.compile(
                optimizer=optimizer,
                loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False),
                metrics=['accuracy'])


            history = model.fit(
                x_train, y_train, epochs=7,
                validation_data=(x_val, y_val),
                callbacks=[early_stopping_cb])

            print('test loss / accuracy')
            test_loss, test_accuracy = model.evaluate(x_test, y_test)
            print(f'{test_loss}, {test_accuracy}')
            # model.save('./middle_model_lr0_2.h5')

            # model = tf.keras.models.load_model('./middle_model_lr0_2.h5')
            # 마지막 5 레이어의 가중치만 열고 다시 학습
            for layer in base_model.layers[-train_layers:]:
                layer.trainable = True


            len(model.trainable_weights)

            optimizer = tf.keras.optimizers.SGD(learning_rate=second_lr, momentum=0.9, decay=0.001)

            model.compile(
                optimizer=optimizer,
                loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False),
                metrics=['accuracy'])

            history = model.fit(
                x_train, y_train, epochs=100,
                validation_data=(x_val, y_val),
                callbacks=[early_stopping_cb])


            print('test loss / accuracy')
            test_loss, test_accuracy = model.evaluate(x_test, y_test)
            print(f'{test_loss}, {test_accuracy}')
            
            print('train loss / accuracy')
            train_loss, train_accuracy = model.evaluate(x_train, y_train)
            print(f'{train_loss}, {train_accuracy}')

            append = pd.DataFrame({
                'test_loss': [test_loss], 
                'test_accuracy': [test_accuracy],
                'train_loss': [train_loss], 
                'train_accuracy': [train_accuracy]
            })
            df = pd.concat([df, append], ignore_index=True)
            

        
df.to_csv('output.csv')

# pd.DataFrame(history.history).plot(figsize=(8, 5))
# plt.grid(True)
# plt.gca().set_ylim(0, 1)
# plt.show()


# model.save('./saved_model.h5')
>>>>>>> 179a14741ac26badd7139201242217fad445806e


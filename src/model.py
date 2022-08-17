from PIL import Image
import numpy as np
import glob
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd

SEED = 13

target = {
  'beagle': 0, 
  'cocker_spaniel': 1, 
  'golden_retriever': 2, 
  'maltese': 3, 
  'pekinese': 4,
  'pomeranian': 5, 
  'poodle': 6, 
  'samoyed': 7, 
  'shih_tzu': 8, 
  'white_terrier': 9
}

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    './data/train',
    target_size=(224, 224),
    batch_size=128,
    class_mode='sparse',
    classes=target,
    seed=SEED
)

val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255,
    horizontal_flip=True
)

val_generator = val_datagen.flow_from_directory(
    './data/val',
    target_size=(224, 224),
    batch_size=128,
    class_mode='sparse',
    classes=target,
    seed=SEED
)

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255,
    horizontal_flip=True
)

test_generator = test_datagen.flow_from_directory(
    './data/test',
    target_size=(224, 224),
    class_mode='sparse',
    classes=target,
    seed=SEED
)

df = pd.DataFrame(
    columns=[
        'first_lr', 'second_lr', 'trainable_layers', 
        'test_loss', 'test_accuracy', 
        'train_loss', 'train_accuracy'
    ])

best_model = None
best_score = -1
best_param = None

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
                train_generator,
                epochs=1,
                validation_data=val_generator,
                callbacks=[early_stopping_cb])

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
                train_generator, 
                epochs=1,
                validation_data=val_generator,
                callbacks=[early_stopping_cb])


            test_loss, test_accuracy = model.evaluate(test_generator)
            print(f'test loss / accuracy : {test_loss}, {test_accuracy}')
            
            train_loss, train_accuracy = model.evaluate(train_generator)

            param = {
                'first_lr': [first_lr],
                'second_lr': [second_lr],
                'trainable_layers': [trainable_layers],
                'test_loss': [test_loss],
                'test_accuracy': [test_accuracy],
                'train_loss': [train_loss],
                'train_accuracy': [train_accuracy]
            }

            if best_score > test_accuracy:
                best_model = model
                best_score = test_accuracy
                best_param = param

            append = pd.DataFrame(param)
            df = pd.concat([df, append], ignore_index=True)
            

        
df.to_csv('../assets/output.csv')

# pd.DataFrame(history.history).plot(figsize=(8, 5))
# plt.grid(True)
# plt.gca().set_ylim(0, 1)
# plt.show()


best_model.save('./assets/best_model.h5')


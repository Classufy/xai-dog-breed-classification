from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sns

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

def load_data():
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        rotation_range=40,
        width_shift_range = 0.2,
        height_shift_range = 0.2,
        shear_range = 0.5,
    )

    train_generator = train_datagen.flow_from_directory(
        './data/train',
        target_size=(299, 299),
        batch_size=128,
        class_mode='sparse',
        classes=target,
        seed=SEED,
        shuffle=True
    )

    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    val_generator = val_datagen.flow_from_directory(
        './data/val',
        target_size=(299, 299),
        batch_size=128,
        class_mode='sparse',
        classes=target,
        seed=SEED,
        shuffle=True
    )

    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    test_generator = test_datagen.flow_from_directory(
        './data/test',
        target_size=(299, 299),
        class_mode='sparse',
        classes=target,
        shuffle=False,
    )
    return train_generator, val_generator, test_generator


def train_model(first_lr, second_lr, trainable_layers, train_generator, val_generator):
    base_model = tf.keras.applications.xception.Xception(
        weights='imagenet', include_top=False, input_shape=(299, 299, 3))

    avg = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    output = tf.keras.layers.Dense(len(target), activation='softmax')(avg)

    model = tf.keras.Model(inputs=base_model.input, outputs=output)

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


    first_history = model.fit(
        train_generator,
        epochs=10,
        validation_data=val_generator,
        callbacks=[early_stopping_cb])

    for layer in base_model.layers[-trainable_layers:]:
        layer.trainable = True


    len(model.trainable_weights)

    optimizer = tf.keras.optimizers.SGD(learning_rate=second_lr, momentum=0.9, decay=0.001)

    model.compile(
        optimizer=optimizer,
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy'])

    second_history = model.fit(
        train_generator, 
        epochs=1000,
        validation_data=val_generator,
        callbacks=[early_stopping_cb])

    return model, first_history, second_history

def show_confusion_matrix_plot(y_true, y_pred, test_generator):
    cm = confusion_matrix(y_true, y_pred)

    length = 10 # two classes
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, vmin=0, fmt='g', cmap='Blues', cbar=False)
    classes=list(test_generator.class_indices.keys())
    plt.xticks(np.arange(length)+.5, classes, rotation= 90)
    plt.yticks(np.arange(length)+.5, classes, rotation=0)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()


def main():
    train_generator, val_generator, test_generator = load_data()
    first_lr = 0.2
    second_lr = 0.01
    trainable_layers = 7
    model, first_history, second_history = train_model(
        first_lr, second_lr, trainable_layers, train_generator, val_generator)

    pd.DataFrame(first_history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()

    pd.DataFrame(second_history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()

    model.save('./assets/best_model.h5')
    

    model = tf.keras.models.load_model('./assets/best_model.h5')

    model.evaluate(test_generator)


    pred = model.predict(test_generator, verbose=1)
    y_pred = [np.argmax(p) for p in pred]
    y_true = test_generator.classes

    show_confusion_matrix_plot(y_true, y_pred, test_generator)

if __name__ == "__main__": main()
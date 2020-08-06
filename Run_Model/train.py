import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from google.colab import drive
import urllib.request
import os.path
from typing import Optional, List, Callable
from PIL import Image
import shutil, os
import zipfile
import matplotlib.pyplot as plt
from typing import Optional, List, Callable
import numpy as np
import pandas as pd
import shutil, os
import pprint
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from collections import Counter

def get_k_splits(train_dsu): #, n_folds = 10):
  all_images = []
  all_labels = []
  for image, label in train_dsu.take(-1):
    all_images.append(image.numpy())
    all_labels.append(int(label))
  all_images = np.array(all_images)
  all_labels = np.array(all_labels)
  return (all_images, all_labels)#, skf, fold_inds)

data_augmentation = keras.Sequential([
  layers.experimental.preprocessing.RandomFlip('horizontal'),
  layers.experimental.preprocessing.RandomRotation(0.1),
])

# setup
# read the data in, save some for the test set
image_size = (160, 160) ####
image_shape = image_size + (3,)
batch_size = 32

# training set
train_ds = tf.keras.preprocessing.image_dataset_from_directory('images_directory',
    validation_split=0.25, subset='training', seed=1337,
    image_size=image_size, batch_size=batch_size)

# test set ##### MAKE THIS THE TESTING SET -> 25% of the data
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
   'images_directory', validation_split=0.25, subset='validation', seed=1337,
    image_size=image_size, batch_size=batch_size)

train_dsu = tf.data.Dataset.unbatch(train_ds)
del train_ds

x = get_k_splits(train_dsu)

# pretrained model with SMOTE
from keras.models import model_from_json
import time  # for time stamp on file names
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
from collections import Counter

modelname = "P6SMOTEfinal"  ##### MODEL NAME
# K-fold Cross Validation model evaluation

num_folds = 3  # THREE FOLDS
acc_per_fold = []
loss_per_fold = []
kfold = StratifiedKFold(n_splits=num_folds, shuffle=True)
print("got kfold")
input_shape = image_shape
no_classes = 3

fold_no = 0
for train, val in kfold.split(x[0], x[1]):
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')
    unique, counts = np.unique(x[1][train], return_counts=True)
    print("COUNTTTTTTTTS: ", dict(zip(unique, counts)))

    # SMOTE
    sm = SMOTE(random_state=42)
    hold_shape = x[0][train].shape
    X_train = x[0][train].reshape(hold_shape[0], hold_shape[1] * hold_shape[2] * hold_shape[3])
    y_train = x[1][train].reshape(hold_shape[0], 1)
    X_smote, y_smote = sm.fit_resample(X_train, y_train)
    new_shape = X_smote.shape[0]
    a = X_smote.reshape(new_shape, hold_shape[1], hold_shape[2], hold_shape[3])
    b = y_smote.reshape(new_shape, )
    X_smote = a
    y_smote = b
    del a
    del b
    del X_train
    del y_train

    print("SMOTE X SHAPE:", X_smote.shape)
    print("SMOTE Y SHAPE:", X_smote.shape)

    unique, counts = np.unique(y_smote, return_counts=True)
    print("SMOTE COUNTTTTTTTTS: ", dict(zip(unique, counts)))

    # batch it up right before running
    train_generator = batch_generator(X_smote, y_smote, 32)
    del X_smote
    del y_smote

    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    prediction_layer = tf.keras.layers.Dense(3, activation='softmax')
    base_model = tf.keras.applications.MobileNetV2(input_shape=image_shape,
                                                   include_top=False,
                                                   weights='imagenet')

    model = tf.keras.Sequential([layers.experimental.preprocessing.RandomFlip('horizontal'),
                                 layers.experimental.preprocessing.RandomRotation(0.1),
                                 base_model,
                                 tf.keras.layers.GlobalAveragePooling2D(),
                                 tf.keras.layers.Dense(3, activation='softmax')])

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(0.0001),  # smaller learning rate
                  loss="SparseCategoricalCrossentropy",  # weighted_ce ,
                  metrics=['accuracy'])

    history = model.fit(train_generator,  # X_smote, y_smote,
                        batch_size=batch_size,
                        validation_data=(x[0][val], x[1][val]),
                        epochs=100,  # WENT BACK TO 100####################################################250 before
                        steps_per_epoch=100,
                        class_weight={0: 4., 1: 26.,
                                      2: 16.})  ### DON'T NEED CLASS WEIGHTS IF ALL OF THEM HAVE EQUAL COUNTS

    model_json = model.to_json()
    with open("model_" + modelname + time.strftime("%Y%m%d-%H%M%S") + ".json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("model_" + modelname + time.strftime("%Y%m%d-%H%M%S") + ".h5")
    print(time.strftime("%Y%m%d-%H%M%S") + "Saved model to disk \n")

    # Generate generalization metrics
    scores = model.evaluate(x[0][val], x[1][val], verbose=0)
    print(
        f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1] * 100}%')
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])
    #### get the classification report
    predictions = model.predict(x[0][val])
    print(predictions)
    labels = np.argmax(predictions, axis=-1)
    print(labels)
    target_names = ['class 0', 'class 1', 'class 2']
    print(classification_report(y_true=x[1][val], y_pred=labels, target_names=target_names))  # , output_dict=True))
    my_eval = classification_report(y_true=x[1][val], y_pred=labels, target_names=target_names, output_dict=True)
    my_eval_df = pd.DataFrame(my_eval).transpose()
    pairs = [(i, j) for i, j in zip(labels, x[1][val])]
    print(Counter(elem for elem in pairs))
    temp = dict(Counter(elem for elem in pairs))

    for j in range(3):
        name = "pred_" + str(j)
        my_eval_df[name] = [0, 0, 0, "-", "-", "-"]
        for i in range(3):
            if (j, i) in temp.keys():
                my_eval_df[name][i] = temp[(j, i)]
    print(my_eval_df)
    my_eval_df.to_csv(modelname + '_classification_report_' + str(fold_no) + '.csv', index=True)
    my_eval_df = 0  # save space
    del my_eval
    del my_eval_df
    del temp

    print(history.history.keys())
    #  "Accuracy"
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(modelname + "_accuracy_" + str(fold_no) + ".png")
    plt.show()

    # "Loss"
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(modelname + "_loss_" + str(fold_no) + ".png")
    plt.show()

    fold_no = fold_no + 1

# == Provide average scores ==
print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(acc_per_fold)):
    print('------------------------------------------------------------------------')
    print(f'> Fold {i + 1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
print(f'> Loss: {np.mean(loss_per_fold)}')

# save results
temp = pd.DataFrame(list(zip(acc_per_fold, loss_per_fold)),
                    columns=['Accuracy', 'Loss'])
# temp = temp.T
temp.loc['mean'] = temp.mean()
print(temp)
temp.to_csv(modelname + '.csv', index=True)
del temp



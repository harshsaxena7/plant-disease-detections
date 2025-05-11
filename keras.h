#ifndef TRAIN_PLANT_DISEASE_H
#define TRAIN_PLANT_DISEASE_H

// This file is auto-generated from Train_plant_disease.ipynb

// Code cell 1
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

// Code cell 2
training_set = tf.keras.utils.image_dataset_from_directory(
    'train',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(128, 128),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False
)

// Code cell 3
validation_set = tf.keras.utils.image_dataset_from_directory(
    'valid',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(128, 128),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False
)

// Code cell 4
cnn = tf.keras.models.Sequential()

// Code cell 5
cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,padding='same',activation='relu',input_shape=[128,128,3]))
cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))

// Code cell 6
cnn.add(tf.keras.layers.Conv2D(filters=64,kernel_size=3,padding='same',activation='relu'))
cnn.add(tf.keras.layers.Conv2D(filters=64,kernel_size=3,activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))

// Code cell 7
cnn.add(tf.keras.layers.Conv2D(filters=128,kernel_size=3,padding='same',activation='relu'))
cnn.add(tf.keras.layers.Conv2D(filters=128,kernel_size=3,activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))

// Code cell 8
cnn.add(tf.keras.layers.Conv2D(filters=256,kernel_size=3,padding='same',activation='relu'))
cnn.add(tf.keras.layers.Conv2D(filters=256,kernel_size=3,activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))

// Code cell 9
cnn.add(tf.keras.layers.Conv2D(filters=512,kernel_size=3,padding='same',activation='relu'))
cnn.add(tf.keras.layers.Conv2D(filters=512,kernel_size=3,activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))

// Code cell 10
cnn.add(tf.keras.layers.Dropout(0.25))

// Code cell 11
cnn.add(tf.keras.layers.Flatten())

// Code cell 12
cnn.add(tf.keras.layers.Dense(units=1500,activation='relu'))

// Code cell 13
cnn.add(tf.keras.layers.Dropout(0.4)) #To avoid overfitting

// Code cell 14
#Output Layer
cnn.add(tf.keras.layers.Dense(units=38,activation='softmax'))

// Code cell 15
cnn.compile(optimizer=tf.keras.optimizers.legacy.Adam(
    learning_rate=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])

// Code cell 16
cnn.summary()

// Code cell 17
training_history = cnn.fit(x=training_set,validation_data=validation_set,epochs=10)

// Code cell 18


// Code cell 19
#Training set Accuracy
train_loss, train_acc = cnn.evaluate(training_set)
print('Training accuracy:', train_acc)

// Code cell 20
#Validation set Accuracy
val_loss, val_acc = cnn.evaluate(validation_set)
print('Validation accuracy:', val_acc)

// Code cell 21
cnn.save('trained_plant_disease_model.keras')

// Code cell 22
training_history.history #Return Dictionary of history

// Code cell 23
#Recording History in json
import json
with open('training_hist.json','w') as f:
  json.dump(training_history.history,f)

// Code cell 24
print(training_history.history.keys())

// Code cell 25
epochs = [i for i in range(1,11)]
plt.plot(epochs,training_history.history['accuracy'],color='red',label='Training Accuracy')
plt.plot(epochs,training_history.history['val_accuracy'],color='blue',label='Validation Accuracy')
plt.xlabel('No. of Epochs')
plt.title('Visualization of Accuracy Result')
plt.legend()
plt.show()

// Code cell 26


// Code cell 27
class_name = validation_set.class_names

// Code cell 28
test_set = tf.keras.utils.image_dataset_from_directory(
    'valid',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=1,
    image_size=(128, 128),
    shuffle=False,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False
)

// Code cell 29


// Code cell 30
y_pred = cnn.predict(test_set)
predicted_categories = tf.argmax(y_pred, axis=1)

// Code cell 31
true_categories = tf.concat([y for x, y in test_set], axis=0)
Y_true = tf.argmax(true_categories, axis=1)

// Code cell 32
Y_true

// Code cell 33
predicted_categories

// Code cell 34


// Code cell 35
from sklearn.metrics import confusion_matrix,classification_report
cm = confusion_matrix(Y_true,predicted_categories)

// Code cell 36
# Precision Recall Fscore
print(classification_report(Y_true,predicted_categories,target_names=class_name))

// Code cell 37
plt.figure(figsize=(40, 40))
sns.heatmap(cm,annot=True,annot_kws={"size": 10})

plt.xlabel('Predicted Class',fontsize = 20)
plt.ylabel('Actual Class',fontsize = 20)
plt.title('Plant Disease Prediction Confusion Matrix',fontsize = 25)
plt.show()

// Code cell 38


#endif // TRAIN_PLANT_DISEASE_H

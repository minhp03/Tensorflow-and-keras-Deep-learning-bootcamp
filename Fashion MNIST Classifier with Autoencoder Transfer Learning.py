
import numpy as np
import tensorflow as tf

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


unsupervised_minh = { 'images': train_images}

supervised_minh = {
    'images': test_images,
    'labels': test_labels
}


# Normalize the images to be between 0 and 1
unsupervised_minh['images'] = unsupervised_minh['images'].astype('float32') / 255.0
supervised_minh['images'] = supervised_minh['images'].astype('float32') / 255.0

supervised_minh['labels'] = tf.keras.utils.to_categorical(supervised_minh['labels'])



print("\n =========================")

print(unsupervised_minh['images'].shape)
print("\n =========================")

print(supervised_minh['labels'].shape)
print("\n =========================")
print(supervised_minh['images'])




import pandas as pd
from sklearn.model_selection import train_test_split

#Using Sklearn's train_test_split() method split the unsupervised dataset 
#into training (57,000 samples) and validation (3,000 samples).
seed = 56  
unsupervised_train_minh, unsupervised_val_minh = train_test_split(unsupervised_minh['images'], 
                                                                   test_size=3000, random_state=seed)

supervised_minh_discard_img = supervised_minh['images'][-3000:]
supervised_minh_discard_lbl = supervised_minh['labels'][-3000:]


from sklearn.model_selection import train_test_split

x_train_minh, x_temp, y_train_minh, y_temp = train_test_split(
    supervised_minh_discard_img.reshape(-1, 28*28),  
    supervised_minh_discard_lbl, 
    train_size=1800, random_state=seed)

x_val_minh, x_test_minh, y_val_minh, y_test_minh = train_test_split(
    x_temp, y_temp, 
    test_size=600, 
    random_state=seed)


x_train_minh= pd.DataFrame(x_train_minh.reshape(-1, 28*28))
x_val_minh = pd.DataFrame(x_val_minh.reshape(-1, 28*28))
x_test_minh = pd.DataFrame(x_test_minh.reshape(-1, 28*28))


print("Unsup Train Set Shape:", unsupervised_train_minh.shape)
print("Unsupe Val Set Shape:", unsupervised_val_minh.shape)

print(" Training  Shape:", x_train_minh.shape, "Labels Shape:", y_train_minh.shape)
print(" Validation  Shape:", x_val_minh.shape, "Labels Shape:", y_val_minh.shape)
print(" Testing  Shape:", x_test_minh.shape, "Labels Shape:", y_test_minh.shape)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras import layers, models


cnn_v1_model_minh = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=2),
    layers.Conv2D(8, (3, 3), activation='relu', padding='same', strides=2),
    layers.Flatten(),
    layers.Dense(100, activation='relu'),
    layers.Dense(10, activation='softmax')
])

#compile
cnn_v1_model_minh.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
cnn_v1_model_minh.summary()



cnn_v1_history_minh = cnn_v1_model_minh.fit(
    x_train_minh.values.reshape(-1, 28, 28, 1),  # Reshaping to match the CNN input shape
    y_train_minh,
    epochs=10,
    batch_size=256,
    validation_data=(x_val_minh.values.reshape(-1, 28, 28, 1), y_val_minh)
)


import matplotlib.pyplot as plt

plt.plot(cnn_v1_history_minh.history['accuracy'], label='Training Accuracy')
plt.plot(cnn_v1_history_minh.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training Vs Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


test_loss, test_accuracy = cnn_v1_model_minh.evaluate(x_test_minh.values.reshape(-1, 28, 28, 1), y_test_minh)
print(f"Test Accuracy:",+ test_accuracy)



from sklearn.metrics import confusion_matrix
import seaborn as sns

# Predict
cnn_predictions_minh = cnn_v1_model_minh.predict(x_test_minh.values.reshape(-1, 28, 28, 1))

# Convert predictions to label indices
predictions_labels = np.argmax(cnn_predictions_minh, axis=1)

# True labels
true_labels = np.argmax(y_test_minh, axis=1)

# Confusion Matrix
cm = confusion_matrix(true_labels, predictions_labels)

# Plot
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

print(cm)


#f random noise to unsupervised data set
noise_factor = 0.2
x_train_noisy_minh = unsupervised_train_minh + noise_factor * tf.random.normal(shape=unsupervised_train_minh.shape, seed=56)
x_val_noisy_minh = unsupervised_val_minh + noise_factor * tf.random.normal(shape=unsupervised_val_minh.shape, seed=56)

x_train_noisy_minh = tf.clip_by_value(x_train_noisy_minh, clip_value_min=0., clip_value_max=1.)
x_val_noisy_minh = tf.clip_by_value(x_val_noisy_minh, clip_value_min=0., clip_value_max=1.)


fig, axes = plt.subplots(1, 10, figsize=(20, 2))
for i, ax in enumerate(axes.flat):
    ax.imshow(x_val_noisy_minh[i], cmap='pink')
    ax.axis('off')
plt.show()


#g. Build and pretrain Autoencoder


from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, Conv2DTranspose

# Input layer
inputs_minh = Input(shape=(28, 28, 1))

# Encoder
e_minh = Conv2D(16, (3, 3), activation='relu', padding='same', strides=2)(inputs_minh)
e_minh = Conv2D(8, (3, 3), activation='relu', padding='same', strides=2)(e_minh)

# Decoder
d_minh = Conv2DTranspose(8, (3, 3), activation='relu', padding='same', strides=2)(e_minh)
d_minh = Conv2DTranspose(16, (3, 3), activation='relu', padding='same', strides=2)(d_minh)


decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(d_minh)

autoencoder_minh = Model(inputs_minh, decoded)
autoencoder_minh.compile(optimizer='adam', loss='mean_squared_error')

autoencoder_minh.summary()

#prediction
autoencoder_history_minh = autoencoder_minh.fit(
    x_train_noisy_minh, unsupervised_train_minh, 
    epochs=10,
    batch_size=256,
    shuffle=True,
    validation_data=(x_val_noisy_minh, unsupervised_val_minh)
)

autoencoder_predictions_minh = autoencoder_minh.predict(x_val_noisy_minh)


print(autoencoder_predictions_minh)
#display the first 10
fig, axes = plt.subplots(1, 10, figsize=(20, 2))
for i, ax in enumerate(axes.flat):
    ax.imshow(autoencoder_predictions_minh[i].reshape(28, 28), cmap='gray')
    ax.axis('off')
plt.show()


#H. Build and perform transfer learning on a CNN with the Autoencoder

encoder_input = autoencoder_minh.input
encoder_output = autoencoder_minh.layers[-4].output  

cnn_v2_minh = Sequential([
    Model(inputs=encoder_input, outputs=encoder_output),  
    Flatten(),  
    Dense(100, activation='relu'), 
    Dense(10, activation='softmax')  
])

#JCompare the performance of the baseline CNN model to the pretrained model in 
#your report


#########################################################
cnn_v2_minh.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
cnn_v2_minh.summary()


cnn_v2_history_minh= cnn_v2_minh.fit(
    x_train_minh.values.reshape(-1, 28, 28, 1),  # Ensure input shape matches
    y_train_minh,
    epochs=10,
    batch_size=256,
    validation_data=(x_val_minh.values.reshape(-1, 28, 28, 1), y_val_minh)
)
plt.plot(cnn_v2_history_minh.history['accuracy'], label='Training Accuracy')
plt.plot(cnn_v2_history_minh.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training Vs Val')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

test_loss_v2, test_accuracy_v2 = cnn_v2_minh.evaluate(x_test_minh.values.reshape(-1, 28, 28, 1), y_test_minh)
print(test_accuracy_v2)


# Predict
cnn_predictions_v2_minh = cnn_v2_minh.predict(x_test_minh.values.reshape(-1, 28, 28, 1))

predictions_labels_v2 = np.argmax(cnn_predictions_v2_minh, axis=1)
cm_v2 = confusion_matrix(true_labels, predictions_labels_v2)
#########################################################
print(cm_v2)

plt.figure(figsize=(10, 8))
sns.heatmap(cm_v2, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
#########################################################
 
plt.plot(cnn_v1_history_minh.history['val_accuracy'], label='Baseline Model Val')
plt.plot(cnn_v2_history_minh.history['val_accuracy'], label='Pretrained Model Val')
plt.title('ValBaseline / Pretrained Model')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


import tensorflow as tf
import pathlib
import keras
from keras.applications import MobileNetV2
from keras import layers, Model
import matplotlib.pyplot as plt

print(tf.config.list_physical_devices('GPU'))

tf.random.set_seed(123)


data_dir = pathlib.Path('D:/programming stuff/ml cnn/data')
train_dir = data_dir / 'train'
test_dir = data_dir / 'test'

img_height = 224
img_width = 224
batch_size = 32

train_ds = keras.utils.image_dataset_from_directory(
    train_dir,
    validation_split = 0.2,
    subset = 'training',
    seed = 123,
    image_size = (img_height, img_width),
    batch_size = batch_size,
    label_mode = 'binary'
)

val_ds = keras.utils.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset='validation',
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode='binary'
)

test_ds = keras.utils.image_dataset_from_directory(
  test_dir,
  image_size=(img_height, img_width),
  batch_size=batch_size,
  label_mode='binary',
  shuffle=False 
)

class_names = train_ds.class_names
print("Class names: ", class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE) #Caching the dataset, shuffling it and allowing multithreading. This piece of code is mostly to speed up training.
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

base_model = MobileNetV2(input_shape=(img_height, img_width, 3),
                         include_top=False, 
                         weights='imagenet')


base_model.trainable = False

inputs = keras.Input(shape=(img_height, img_width, 3))

#x = keras.applications.mobilenet_v2.preprocess_input(inputs)
x = layers.Rescaling((1./127.5), offset=-1)(inputs)

x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.2)(x)


outputs = layers.Dense(1, activation='sigmoid')(x)


model = Model(inputs, outputs)


model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), 
              loss=keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])


model.summary()

initial_epochs = 20

history = model.fit(train_ds, epochs=initial_epochs, validation_data=val_ds)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(initial_epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

loss, accuracy = model.evaluate(test_ds)
print(f"Test accuracy: {accuracy:.4f}")
print(f"Test loss: {loss:.4f}")

#model.save('D:/programming stuff/ml cnn/savedmodel')
#print(f"Model saved in SavedModel format at: D:/programming stuff/ml cnn/savedmodel")
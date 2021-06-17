import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


# Global variables
image_height = 14
image_width = 14
num_channels = 1
num_samples = 60000
num_classes = 10

# Makes the program not crash somehow
allow_growth = True
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)


# Resizes all number images from 28x28 to 14x14
# Rounds each pixel above 128 to 1.0 and below to 0.0
def resize_number_image(number):
    new_number = np.zeros((14, 14))
    for i in range(14):
        for j in range(14):
            avg_val = (int(number[i*2, j*2]) + int(number[i*2 + 1, j*2]) + int(number[i*2, j*2 + 1]) + int(number[i*2 + 1, j*2 + 1])) / 4
            new_number[i, j] = 1.0 if avg_val > 60 else 0.0  # avg_val / 255

    return new_number


# The model architecture
def define_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform',
                     input_shape=(image_height, image_width, num_channels)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(50, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    # compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
proc_x_train = []
proc_x_test = []
# decrease image size to 14x14
for number, label in zip(x_train, y_train):
    proc_x_train.append(resize_number_image(number).flatten().reshape(14, 14))
for number, label in zip(x_test, y_test):
    proc_x_test.append(resize_number_image(number).flatten().reshape(14, 14))

# Convert to numpy arrays
proc_x_train = np.asarray(proc_x_train)
proc_x_test = np.asarray(proc_x_test)


# fig = plt.figure(figsize=(14, 14))
# columns = 4
# rows = 4
# for i in range(1, columns*rows+1):
#     img = proc_x_train[i]
#     fig.add_subplot(rows, columns, i)
#     plt.imshow(img)
#
# plt.show()
# exit()

# create train and test data/labels
train_data = np.reshape(proc_x_train, (proc_x_train.shape[0], image_height, image_width, num_channels))
test_data = np.reshape(proc_x_test, (proc_x_test.shape[0], image_height, image_width, num_channels))
train_labels = to_categorical(y_train, num_classes)
test_labels = to_categorical(y_test, num_classes)



# Shuffle training data
for _ in range(5):
    indexes = np.random.permutation(len(train_data))
train_data = train_data[indexes]
train_labels = train_labels[indexes]

# set aside 10% of training data as cross-validation set
val_perc = 0.10
val_count = int(val_perc * len(train_data))

# pick validations set from train_data/labels
val_data = train_data[:val_count,:]
val_labels = train_labels[:val_count,:]

# leave rest in training set
train_data2 = train_data[val_count:,:]
train_labels2 = train_labels[val_count:,:]

# build the model
model = define_model()

print("shape::", test_data.shape)

# Fit the model
results = model.fit(train_data2, train_labels2,
                    epochs=15, batch_size=128,
                    validation_data=(val_data, val_labels))

print(test_data.shape)

# Evaluate the model on the test data
test_loss, test_accuracy = \
  model.evaluate(test_data, test_labels, batch_size=64)
print('Test loss: %.4f accuracy: %.4f' % (test_loss, test_accuracy))
model.save("mnist_FFN")

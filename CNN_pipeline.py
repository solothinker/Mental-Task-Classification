import tensorflow as tf
import numpy as np
import os
import glob
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras import Input
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
np.random.seed(1111)

#------------------------------
time_window_size =5000
def tf_data_generator(file_list, batch_size = 20):
    i = 0
    N_FEATURES = 1
    segments = []
    time_steps = 5000
    step = 512#int(time_steps/2)
    label_name = 'Action'
    while True:
        if i*batch_size >= len(file_list):  
            i = 0
            np.random.shuffle(file_list)
        else:
            file_chunk = file_list[i*batch_size:(i+1)*batch_size] 
            data = []
            labels = []
            label_classes = ['EEG Fp1','Action']
##            label_classes = tf.constant(['EEG Fp1','Action']) # This line has changed.
            for file in file_chunk:
                df = pd.read_csv(open(file,'r'),usecols=label_classes)
                for ii in range(0, len(df) - time_steps, step):
                    xs = df['EEG Fp1'].values[ii: ii + time_steps]
                    label = stats.mode(df[label_name][ii: ii + time_steps])[0][0]
                    segments.append([xs])
                    labels.append(label)
##                data.append(segments)
##                print(df)
##                labels=1
##                print('###################################')
##                data.append(temp.values.reshape(32,32,1)) 
##                pattern = tf.constant(eval("file[14:21]"))  # This line has changed
##                for j in range(len(label_classes)):
##                    if re.match(pattern.numpy(), label_classes[j].numpy()):  # This line has changed.
##                        labels.append(j)
            data = np.asarray(segments, dtype= np.float32).reshape(-1, time_steps, N_FEATURES)#np.asarray(data)#.reshape(-1,32,32,1)
            labels = np.asarray(labels)
            yield data, labels
            i = i + 1
##        if len(file_list)==i:
##            break
            
# collecting the data
##path  = "secondWave\*"
##files = glob.glob(path)
##check_data = tf_data_generator(files, batch_size = 1)
##num = 0
##for data, labels in check_data:
##    print(data.shape, labels.shape)
##    print(labels, "<--Labels")
##    print()
##    num = num + 1
##    if num > 2: break
path = "eeg_csv/*"
files =  glob.glob(path)
train, test = train_test_split(files, test_size = 20, random_state = 5)
train, val = train_test_split(train, test_size = 10, random_state = 1)
np.random.shuffle(train)
print("Number of train_files:" ,len(train))
print("Number of validation_files:" ,len(val))
print("Number of test_files:" ,len(test))

batch_size = 20
train_dataset = tf.data.Dataset.from_generator(tf_data_generator, args = [train, batch_size],output_types = (tf.float32, tf.float32), 
                                              output_shapes = ((None, 5000,1),(None,)))


validation_dataset = tf.data.Dataset.from_generator(tf_data_generator, args = [val, batch_size],output_types = (tf.float32, tf.float32),
                                                   output_shapes = ((None, 5000,1),(None,)))


test_dataset = tf.data.Dataset.from_generator(tf_data_generator, args = [test, batch_size],output_types = (tf.float32, tf.float32),
                                             output_shapes = ((None,5000,1),(None,)))

time_window_size =5000
model = Sequential(name='EEG')
# model.add(Input(shape=(time_window_size,1),name='Input'))
model.add(Conv1D(filters=16,kernel_size=5,activation='relu',input_shape = (time_window_size,1),name='Layer1'))
model.add(Conv1D(filters=32,kernel_size=5,activation='relu',name='Layer2'))
model.add(MaxPooling1D(pool_size=2,strides=2,name='Maxpool1d'))
model.add(Flatten(name='Flatten'))
model.add(Dense(1,activation='sigmoid',name='Output'))
model.summary()

tf.keras.utils.plot_model(model, show_shapes=True, rankdir="LR")
##os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
##run_opts = tf.compat.v1.RunOptions(report_tensor_allocations_upon_oom = True)
optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['accuracy'])#, options = run_opts)tf.keras.losses.BinaryCrossentropy()

steps_per_epoch = int(np.ceil(len(train)/batch_size))
validation_steps = int(np.ceil(len(val)/batch_size))
steps = int(np.ceil(len(test)/batch_size))
print("steps_per_epoch = ", steps_per_epoch)
print("validation_steps = ", validation_steps)
print("steps = ", steps)
history = model.fit(train_dataset, validation_data = validation_dataset,
                    steps_per_epoch = steps_per_epoch,
                    validation_steps = validation_steps,
                    epochs = 1,verbose=1
                    )
test_loss, test_accuracy = model.evaluate(test, steps = 10)
print("Test loss: ", test_loss)
print("Test accuracy:", test_accuracy)




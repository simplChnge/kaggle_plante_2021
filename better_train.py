
IMAGE_SHAPE = (150, 150)
​
TRAIN_FROM_NEW = True

from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

from pylab import rcParams

rcParams['figure.figsize'] = 18, 8

train_path = '../input/plant-pathology-2021-fgvc8/train_images'
test_path = '../input/plant-pathology-2021-fgvc8/test_images'
df = pd.read_csv('../input/plant-pathology-2021-fgvc8/train.csv')
dct = defaultdict(list)

for i, label in enumerate(df.labels):
    for category in label.split():
        dct[category].append(i)
 
dct = {key: np.array(val) for key, val in dct.items()}
new_df = pd.DataFrame(np.zeros((df.shape[0], len(dct.keys())), dtype = np.int8), columns = dct.keys())

for key, val in dct.items():
    new_df.loc[val, key] = 1


cats_total = new_df.sum(axis = 1).values
from collections import Counter

Counter(df.labels)
ks = {k: i for i,k in enumerate(Counter(df.labels).keys())}
new_df['labs'] = np.array([ks[v] for v in df.labels])

new_df = pd.concat([df, new_df], axis = 1)

new_df.to_csv('better_train.csv', index = False)

import keras
from keras.models import Sequential, Input, Model
from keras.layers import BatchNormalization
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.convolutional import Convolution1D, Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import Callback,ModelCheckpoint
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
import keras.backend as K

target_columns = list(new_df.columns[3:-1])
def f1(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val
	
	
	
input_layer = Input(shape= tuple(list(IMAGE_SHAPE) + [3]), name="input")

# convolutional block 1
conv1 = Convolution2D(32, kernel_size=(7, 7), activation="relu", name="conv_1")(input_layer)
batch1 = BatchNormalization(name="batch_norm_1")(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2), name="pool_1")(batch1)

# convolutional block 2
conv2 = Convolution2D(64, kernel_size=(5, 5), activation="relu", name="conv_2")(pool1)
batch2 = BatchNormalization(name="batch_norm_2")(conv2)
pool2 = MaxPooling2D(pool_size=(2, 2), name="pool_2")(batch2)

# convolutional block 3
conv3 = Convolution2D(128, kernel_size=(3, 3), activation="elu", name="conv_3")(pool2)
batch3 = BatchNormalization(name="batch_norm_3")(conv3)
pool3 = MaxPooling2D(pool_size=(2, 2), name="pool_3")(batch3)


# convolutional block 4
conv4 = Convolution2D(256, kernel_size=(2, 2), activation="elu", name="conv_4")(pool3)
batch4 = BatchNormalization(name="batch_norm_4")(conv4)
pool4 = MaxPooling2D(pool_size=(2, 2), name="pool_4")(batch4)

# fully connected layers
flatten = Flatten()(pool4)

fc1 = Dense(256, activation="relu", name="fc1")(flatten)
batch4 = BatchNormalization(name="batch_norm_5")(fc1)
d1 = Dropout(rate=0.2, name="dropout1")(batch4)

fc2 = Dense(128, activation="relu", name="fc2")(d1)
batch5 = BatchNormalization(name="batch_norm_6")(fc2)
d2 = Dropout(rate=0.4, name="dropout2")(batch5)

# output layer
output = Dense(5, activation="sigmoid", name="classif")(d2)


if TRAIN_FROM_NEW:
    model = Model(input_layer, output)
else:
    model = keras.models.load_model('../input/better-train-csv-format-keras-starter/data/model', compile=False)


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[f1])

model.summary()

skf = StratifiedKFold(n_splits=4, shuffle = True, random_state = 1488)

for train_ind, test_ind in skf.split(new_df, new_df['labs']):
    print(train_ind)
    print(Counter(new_df.loc[train_ind, 'labels']))
    print()

batch_size = 64

image_generator = keras.preprocessing.image.ImageDataGenerator(
    rescale = 1/255,
    
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    
    rotation_range=30, 
    zoom_range=0.15,
    width_shift_range=0.2, 
    height_shift_range=0.2, 
    shear_range=0.15,
    horizontal_flip=True, 
    vertical_flip=False,
    
    fill_mode="reflect"#,
    #validation_split = 0.3
  )



new_df = new_df.iloc[np.random.choice(new_df.shape[0], new_df.shape[0], replace = False),:]

skf = StratifiedKFold(n_splits = 8, shuffle = True, random_state = 1488)

train_ind, test_ind = next(skf.split(new_df, new_df['labs']))

train_generator = image_generator.flow_from_dataframe(
    dataframe = new_df.iloc[train_ind,:],
    directory = train_path,
    x_col="image",
    y_col = target_columns,
    weight_col=None,
    target_size = IMAGE_SHAPE,
    color_mode="rgb",

    class_mode="raw",
    batch_size=batch_size,
    shuffle=True,
    seed=None,

    #subset='training',
    interpolation="box"
)

validation_generator = image_generator.flow_from_dataframe(
    dataframe = new_df.iloc[test_ind,:],
    directory = train_path,
    x_col="image",
    y_col = target_columns,
    weight_col=None,
    target_size = IMAGE_SHAPE,
    color_mode="rgb",

    class_mode="raw",
    batch_size=batch_size,
    shuffle=True,
    seed=None,

    #subset='validation',
    interpolation="box"
)


batch = next(validation_generator)

print(batch[0].shape)

# train the network
history = model.fit(

    train_generator,
    steps_per_epoch = train_generator.samples // batch_size,
    validation_data = validation_generator, 
    validation_steps = validation_generator.samples // batch_size,
    
    epochs = 2
)

# list all data in history
print(history.history.keys())

# summarize history for accuracy
plt.plot(history.history['f1'])
plt.plot(history.history['val_f1'])
plt.title('model f1 score')
plt.ylabel('f1')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

model.save('data/model')

test_df = pd.read_csv('../input/plant-pathology-2021-fgvc8/sample_submission.csv')


test_generator = keras.preprocessing.image.ImageDataGenerator(
    rescale = 1/255
)


test_generator = test_generator.flow_from_dataframe(
    dataframe= test_df,
    directory = test_path,
    x_col = "image",
    y_col = 'labels',
    target_size = IMAGE_SHAPE,
    color_mode="rgb",

    batch_size=1,
    shuffle=False,
    seed=None,

    subset=None,
    
    interpolation="box"
)

answer = model.predict(test_generator)
​
sub = (answer > 0.5)
​
health = sub.sum(axis = 1) == 0
​
tot = []
for i in range(answer.shape[0]):
    tmp = []
    if health[i]:
        tmp = ['healthy']
    else:
        for j, c in enumerate(target_columns):
            if sub[i, j]:
                tmp.append(c)
    
    tot.append(tmp)
​
tot = [' '.join(t) for t in tot]
​
test_df['labels'] = np.array(tot)

test_df.to_csv('submission.csv', index = False)


















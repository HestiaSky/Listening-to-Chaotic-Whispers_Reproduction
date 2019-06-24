from keras.models import Model
from keras.layers import Dense, Input, Activation, multiply, Lambda
from keras.layers import TimeDistributed, GRU, Bidirectional
from keras import backend as K
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical


def han():
    input1 = Input(shape=(50, 500), dtype='float32')

    # Attention Layer
    dense_layer = Dense(500, activation='tanh')(input1)
    softmax_layer = Activation('softmax')(dense_layer)
    attention_mul = multiply([softmax_layer, input1])
    # end attention layer

    vec_sum = Lambda(lambda x: K.sum(x, axis=1))(attention_mul)
    pre_model1 = Model(input1, vec_sum)
    pre_model2 = Model(input1, vec_sum)

    input2 = Input(shape=(11, 50, 500), dtype='float32')

    pre_gru = TimeDistributed(pre_model1)(input2)
    # bidirectional gru
    l_gru = Bidirectional(GRU(250, return_sequences=True))(pre_gru)

    post_gru = TimeDistributed(pre_model2)(l_gru)
    # MLP to perform classification
    dense1 = Dense(100, activation='tanh')(post_gru)
    dense2 = Dense(3, activation='tanh')(dense1)
    final = Activation('softmax')(dense2)
    final_model = Model(input2, final)
    final_model.summary()

    return final_model


def twin_creation(x_train_folder, y_train_folder):
    x_train_list = os.listdir(x_train_folder)
    x_train_list = sorted(x_train_list)

    y_train_list = os.listdir(y_train_folder)
    y_train_list = sorted(y_train_list)

    duo_list = []
    for i in range(len(y_train_list)):
        duo = [x_train_list[i], y_train_list[i]]
        duo_list.append(duo)
    duo_list = [duo for duo in duo_list if duo[0][-4:] == '.npy']
    return duo_list


def training(x_name, y_name, model):
    x_train = np.load('/home/lixinhang/data/x_train/' + x_name)
    y_train = np.load('/home/lixinhang/data/y_train/' + y_name)

    # Encoding y
    encoder = LabelEncoder()
    encoder.fit(y_train)
    encoded_Y = encoder.transform(y_train)
    y_train_end = to_categorical(encoded_Y)

    if y_train_end.shape[1] != 3:
        pass
    else:
        model.train_on_batch(x_train, y_train_end)
        print("model fitting on " + x_name)


if __name__ == "__main__":
    model = han()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # Put your training data folder path
    x_train_folder = '/home/lixinhang/data/x_train'
    y_train_folder = '/home/lixinhang/data/y_train'
    epochs = 60

    duo_list = twin_creation(x_train_folder, y_train_folder)
    for epoch in range(epochs):
        for k, duo in enumerate(duo_list):
            print('fitting on firm {} out of epoch {}'.format(duo[0][:-12], epoch))
            training(duo[0], duo[1], model)

    model.save('your_model_{}epochs.hdf5'.format(epochs))

    print('Model saved!')








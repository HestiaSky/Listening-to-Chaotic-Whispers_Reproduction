import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Model, load_model
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder


def open_pickle(url):
    with open(url, 'rb') as file:
        return pickle.load(file)


def open_numpy(url):
    with open(url, 'rb') as file:
        f = np.load(file)
        return f


def compute_accuracy(real_value, prediction):
    ech_len = len(real_value)
    rights = 0
    assert len(real_value) == len(prediction)
    for i, j in enumerate(real_value):
        if real_value[i] == prediction[i]:
            rights += 1
    if ech_len > 0:
        return rights, ech_len, (rights/ech_len)*100
    else:
        return 0, 0, 0

'''
def plot_data(real_value, prediction, accuracy):
    assert len(real_value) == len(prediction)
    fig, ax = plt.subplots()
    fig.suptitle('Plot real variation of stock prices')
    y = []
    y_pred = []
    for i in range(0, 50):
        y.append(real_value[i])
        y_pred.append(prediction[i])
    x = np.arange(0, 50, 1)
    ax.set_title('Accuracy = ' + str(accuracy) + " %")
    ax.set_xlabel("Days")
    ax.set_ylabel("Variation of stock prices")
    ax.plot(x, y)
    ax.plot(x, y_pred)
    plt.show()
'''

def load_data(model, x_train_file, y_train_file, x_test_file, y_test_file):

    '''
    x_train = np.load(x_train_file)
    y_train = np.load(y_train_file)

    encoder = LabelEncoder()
    encoder.fit(y_train)
    encoded_Y = encoder.transform(y_train)
    y_train = to_categorical(encoded_Y)
    '''

    x_test = np.load(x_test_file)
    y_test = np.load(y_test_file)

    print("model compiling - Hierachical attention network")
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    #print("model fitting - Hierachical attention network")
    #model.fit(x_train, y_train, epochs=200)

    print("validation_test")
    final_x_test_predict = model.predict(x_test)
    print("Prediction Y ", final_x_test_predict)
    print("Real Y ", y_test)

    real = []
    for i in y_test:
        if i[0] == 1:
            real.append(-1)
        elif i[1] == 1:
            real.append(0)
        elif i[2] == 1:
            real.append(1)

    print("Real ", real)

    prediction = np.argmax(final_x_test_predict, axis=1)
    prediction = [i-1 for i in prediction]

    print("Prediction", prediction)

    rights, ech_len, accuracy = compute_accuracy(real, prediction)
    print('Name = ', x_test_file[:-11])
    print("Accuracy = ", str(accuracy) + " %")
    result_dic[x_test_file[:-11]] = accuracy
    #plot_data(real, prediction, accuracy)

    return rights, ech_len


if __name__ == '__main__':

    print('Start testing')
    model = load_model('/home/lixinhang/LCW_mac_linux/your_model_60epochs.hdf5')
    file_list = os.listdir('/home/lixinhang/data/x_test')
    file_list = [i[:-11] for i in file_list]
    result_dic = {}
    rights = 0
    ech_len = 0
    for code in file_list:
        x_train_path = '/home/lixinhang/data/x_train/' + code + '_x_train.npy'
        y_train_path = '/home/lixinhang/data/y_train/' + code + '_y_train.npy'
        x_test_path = '/home/lixinhang/data/x_test/' + code + '_x_test.npy'
        y_test_path = '/home/lixinhang/data/y_test/' + code + '_y_test.npy'
        rp, ep = load_data(model, x_train_path, y_train_path, x_test_path, y_test_path)
        rights += rp
        ech_len += ep
    print('Test finished!')
    print(result_dic)
    print((rights/ech_len)*100)

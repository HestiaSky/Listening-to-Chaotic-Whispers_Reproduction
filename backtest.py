import pickle
import csv
import os
import numpy as np
from datetime import datetime
from keras.models import Model, load_model


def open_pickle(url):
    with open(url, 'rb') as file:
        return pickle.load(file)


def open_numpy(url):
    with open(url, 'rb') as file:
        f = np.load(file)
        return f


def compute_accuracy(real_value, prediction):
    ech_len = len(real_value)
    pos_true = 0
    pos = 0
    neg_true = 0
    neg = 0
    assert len(real_value) == len(prediction)
    for i, j in enumerate(real_value):
        if real_value[i] == 1:
            pos += 1
            if prediction[i] == 1:
                pos_true += 1
        elif real_value[i] == -1:
            neg += 1
            if prediction[i] == -1:
                neg_true += 1
    if pos == 0:
        pos = 1
    if neg == 0:
        neg = 1
    if ech_len > 0:
        return (pos/ech_len)*100, (pos_true/pos)*100, (neg/ech_len)*100, \
               (neg_true/neg)*100, ((ech_len-pos-neg)/ech_len)*100
    else:
        return


def backtest_data(model, code, x_test_file, y_test_file, dates_test):
    x_test = open_numpy(x_test_file)
    y_test = np.load(y_test_file)
    dates = open_pickle(dates_test)
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    final_x_test_predict = model.predict(x_test)

    real = []
    for i in y_test:
        if i[0] == 1:
            real.append(-1)
        elif i[1] == 1:
            real.append(0)
        elif i[2] == 1:
            real.append(1)

    pred = np.argmax(final_x_test_predict, axis=1)
    pred = [i - 1 for i in pred]

    pos, pos_recall, neg, neg_recall, equal = compute_accuracy(real, pred)

    accline = {}
    accline['code'] = code
    accline['pos'] = pos
    accline['pos_recall'] = pos_recall
    accline['neg'] = neg
    accline['neg_recall'] = neg_recall
    accline['equal'] = equal

    if pos_recall < 35 or neg_recall < 35:
        return accline, []

    prediction = []
    for i in range(0, len(final_x_test_predict)):
        line = {}
        proba = -1 * final_x_test_predict[i][0] + 1 * final_x_test_predict[i][2]
        date = dates[i]
        date = datetime.strptime(date, '%Y-%m-%d')
        date = int(str(date.year) + str(date.month).zfill(2) + str(date.day).zfill(2))
        line['date'] = date
        line['ticker'] = code
        line['proba'] = proba
        prediction.append(line)

    print(code, ' finished!')
    return accline, prediction


if __name__ == '__main__':
    headers = ['date', 'ticker', 'proba']
    headers2 = ['code', 'pos', 'pos_recall', 'neg', 'neg_recall', 'equal']
    data = []
    accuracy = []
    
    model = load_model('/home/lixinhang/LCW_mac_linux/your_model_60epochs.hdf5')
    file_list = os.listdir('/home/lixinhang/data/x_test')
    file_list = [i[:-11] for i in file_list]
    for code in file_list:
        x_test_file = '/home/lixinhang/data/x_test/' + code + '_x_test.npy'
        y_test_file = '/home/lixinhang/data/y_test/' + code + '_y_test.npy'
        dates_test = '/home/lixinhang/data/dates/' + code + '.pkl'
        acc, prediction = backtest_data(model, code, x_test_file, y_test_file, dates_test)
        accuracy.append(acc)
        for line in prediction:
            data.append(line)

    print('All done!')
    with open('/home/lixinhang/result.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, headers)
        writer.writeheader()
        writer.writerows(data)
        print('Data saved!')
    with open('/home/lixinhang/acc.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, headers2)
        writer.writeheader()
        writer.writerows(accuracy)
        print('Acc saved!')









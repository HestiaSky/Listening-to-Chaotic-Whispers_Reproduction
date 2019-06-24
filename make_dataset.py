import pickle
import numpy as np
from datetime import datetime, timedelta
from gensim.models.doc2vec import Doc2Vec
from sklearn.model_selection import train_test_split
import os
import multiprocessing as mp


stock_dir = '/home/lixinhang/move_folder'
article_dir = '/home/lixinhang/firm_pickle_folder'
data_dir = '/home/lixinhang/data'
mylist = mp.Manager().list()
num = mp.Value('i', 0)


def create_x_y(name, model, article, stock_trend, k_max=40, test_size=0.33, window=10):

    with open(article, 'rb') as dict_article:
        dict_article = pickle.load(dict_article)
    with open(stock_trend, 'rb') as dict_stock:
        dict_stock = pickle.load(dict_stock)

    if (len(dict_article) > 0) and (len(dict_stock) > 0):
        data = np.zeros((1, 11, k_max, 500), dtype='float32')
        y = []
        dates = []

        for i in range(int(1500)):
            today = datetime(2014, 1, 1)+timedelta(days=i)
            tomorrow = today+timedelta(days=1)
            to_add = False

            if tomorrow in dict_stock:
                y_i = int(dict_stock[tomorrow])
                new_row = np.zeros((1, 11, k_max, 500), dtype='float32')

                for j in range(11):
                    day = datetime(2014, 1, 1)+timedelta(days=i-j)
                    if day.strftime("%Y-%m-%d") in dict_article:
                        list_article = dict_article[day.strftime("%Y-%m-%d")]
                        to_add = True

                        for k in range(k_max):
                            if k < len(dict_article[day.strftime("%Y-%m-%d")]):
                                article_id = str(list_article[k])
                                vector = model.docvecs[article_id]
                                new_row[0, j, k, :] = vector
                            else:
                                new_row[0, j, k, :] = np.zeros(500)

                if to_add:
                    data = np.vstack([data, new_row])
                    y.append(y_i)
                    dates.append(tomorrow.strftime("%Y-%m-%d"))

        y_vec = np.asarray(y)
        x_mat = np.delete(data, 0, axis=0)
        x_train, x_test, y_train, y_test = train_test_split(
            x_mat, y_vec, test_size=test_size, random_state=42, shuffle=False)

        y_train_size = y_train.shape[0]
        dates_test = dates[y_train_size:]

        with open(data_dir + '/dates/' + name + '.pkl', 'wb') as handle:
            pickle.dump(dates_test, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return x_train, x_test, y_train, y_test

    else:
        print('File Error!  ', name)
        return [], [], [], []


def create_dataset(stock_list):

    model = Doc2Vec.load('/home/lixinhang/d2v_model')
    article_list = os.listdir(article_dir)

    for i, j in enumerate(stock_list):

        if stock_list[i] in article_list:

            name = stock_list[i][:-4]
            stock_trend = stock_dir + '/' + stock_list[i]
            article = article_dir + '/' + stock_list[i]

            x_train, x_test, y_train, y_test = create_x_y(name, model, article, stock_trend,
                                                          50, 0.33, window=10)
            print(len(x_train), '  ', len(x_test))

            if len(x_train) > 0:
                y_test_encode_start = list()

                for trend in y_test:
                    new_value = trend+1
                    code = [0 for _ in range(3)]
                    code[new_value] = 1
                    y_test_encode_start.append(code)

                np.save(data_dir + '/x_train/' + name + '_x_train.npy', x_train)
                np.save(data_dir + '/x_test/' + name + '_x_test.npy', x_test)
                np.save(data_dir + '/y_train/' + name + '_y_train.npy', y_train)
                np.save(data_dir + '/y_test/' + name + '_y_test.npy', y_test_encode_start)

            num.value += 1
            print('Process: ', num.value, ' Code is ', name)

        else:
            pass


if __name__ == '__main__':

    nb_process = 10

    stock_list = os.listdir(stock_dir)
    stock_list.sort()

    stock_repartition = list(np.array_split(stock_list, nb_process))
    stock_repartition = [x.tolist() for x in stock_repartition]

    process_list = [mp.Process(target=create_dataset, args=(stock, ))
                    for stock in stock_repartition]

    for p in process_list:
        p.start()

    for p in process_list:
        p.join()


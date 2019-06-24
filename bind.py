import pandas as pd
import os
import numpy as np
import multiprocessing as mp
import pickle

mylist = mp.Manager().list()
num = mp.Value('i', 0)


def bind(firm):

    for x in firm:
        firm_name = str(x[1])

        for day in os.listdir(dirname):
            day_path = os.path.join(dirname, day)
            df = pd.read_csv(day_path, encoding='gbk')
            df_list_ID = df['newsID'].values.tolist()
            df_list_body = df['newsBody'].values.tolist()

            if len(df_list_body) > 0:
                for number in range(0, len(df_list_body)):
                    try:
                        if firm_name in df_list_body[number]:
                            mylist.append(df_list_ID[number])
                    except:
                        pass

        for day in os.listdir(dirname2):
            day_path = os.path.join(dirname2, day)
            df = pd.read_csv(day_path, encoding='gbk')
            df_list_ID = df['newsID'].values.tolist()
            df_list_body = df['newsBody'].values.tolist()

            if len(df_list_body) > 0:
                for number in range(0, len(df_list_body)):
                    try:
                        if firm_name in df_list_body[number]:
                            mylist.append(df_list_ID[number])
                    except:
                        pass

        print(firm_name)
        print(len(mylist))


def bind_process():

    # Loading of files
    firm = 'firm_name.csv'
    firm_df = pd.read_csv(firm)
    firm_list = firm_df[['code', 'name']].values.tolist()

    # Parallelization of the task
    nb_process = 30
    l = list(np.array_split(firm_list, nb_process))
    l = [x.tolist() for x in l]
    process_list = [mp.Process(target=bind,
                               args=(f,)) for f in l]

    for p in process_list:
        p.start()

    for p in process_list:
        p.join()

    print('Data get!!!')
    data = set()
    for i in mylist:
        data.add(i)
    print(len(data))
    with open('bind.pkl', 'wb') as f:
        pickle.dump(data, f)


def bind_news(news_list):

    for news in news_list:
        if int(news[:-4]) in bind_list:
            reader = open(news_folder+news, 'r').read()
            writer = open(bind_news_folder+news, 'w')
            writer.write(reader)
            num.value += 1
            print(num.value)


def bind_news_process():

    news_list = os.listdir(news_folder)
    nb_process = 30
    l = list(np.array_split(news_list, nb_process))
    l = [x.tolist() for x in l]
    process_list = [mp.Process(target=bind_news,
                               args=(news_list, )) for news_list in l]

    for p in process_list:
        p.start()

    for p in process_list:
        p.join()


if __name__ == '__main__':

    dirname = '/data/share_data/remotedata/News/Eastmoney/body'
    dirname2 = '/data/share_data/remotedata/News/SinaFinance/body'
    news_folder = '/home/lixinhang/news_folder/'
    bind_news_folder = '/home/lixinhang/bind_news_folder/'

    bind_list = pickle.load(open('bind.pkl', 'rb'))
    print(len(bind_list))

    bind_news_process()

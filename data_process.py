import pandas as pd
import os
import numpy as np
import multiprocessing as mp
import pickle


def create_pickle_firm(spnas):

    #Create a pickle containing articles associated with a company each day

    for x in spnas:
        firm_name = str(x[1])
        data = {}

        for day in os.listdir(dirname):
            day_path = os.path.join(dirname, day)
            df = pd.read_csv(day_path, encoding='gbk')
            df_list_ID = df['newsID'].values.tolist()
            df_list_body = df['newsBody'].values.tolist()

            newsID = []
            hasNews = False
            if len(df_list_body) > 0:
                for number in range(0, len(df_list_body)):
                    try:
                        if firm_name in df_list_body[number]:
                            hasNews = True
                            newsID.append(df_list_ID[number])
                    except:
                        pass

            if hasNews:
                data[day[:-4]] = newsID


        for day in os.listdir(dirname2):
            day_path = os.path.join(dirname2, day)
            df = pd.read_csv(day_path, encoding='gbk')
            df_list_ID = df['newsID'].values.tolist()
            df_list_body = df['newsBody'].values.tolist()

            if type(df_list_body) == list:

                if day[:-4] in data:
                    if len(df_list_body) > 0:
                        for number in range(0, len(df_list_body)):
                            try:
                                if firm_name in df_list_body[number]:
                                    newsID.append(df_list_ID[number])
                            except:
                                pass

                else:
                    newsID = []
                    hasNews = False
                    if len(df_list_body) > 0:
                        for number in range(0, len(df_list_body)):
                            try:
                                if firm_name in df_list_body[number]:
                                    hasNews = True
                                    newsID.append(df_list_ID[number])
                            except:
                                pass

                    if hasNews:
                        data[day[:-4]] = newsID

        print(data)
        with open('{}{}{}'.format(firm_pickle_folder, str(x[0]), ".pkl"), 'wb') as f:
            pickle.dump(data, f)


if __name__ == '__main__':

    # Loading of files
    spnas = 'firm_name.csv'
    spnas_df = pd.read_csv(spnas)
    spnas_list = spnas_df[['code', 'name']].values.tolist()
    #dirname = '../Eastmoney'
    #dirname2 = '../SinaFinance'
    dirname = '/data/share_data/remotedata/News/Eastmoney/body'
    dirname2 = '/data/share_data/remotedata/News/SinaFinance/body'
    firm_pickle_folder = '/home/lixinhang/firm_pickle_folder/'

    # Parallelization of the task
    nb_process = 100
    l = list(np.array_split(spnas_list, nb_process))
    l = [x.tolist() for x in l]
    process_list = [mp.Process(target=create_pickle_firm,
                               args=(spnas,)) for spnas in l]

    for p in process_list:
        p.start()

    for p in process_list:
        p.join()

import pandas as pd
import os
import multiprocessing as mp


def process_news(date):
    day_path = os.path.join(dirname, date)
    df = pd.read_csv(day_path, encoding='gbk')
    df_list_ID = df['newsID'].values.tolist()
    df_list_body = df['newsBody'].values.tolist()

    for i in range(0, len(df_list_ID)):
        try:
            id = df_list_ID[i]
            news = df_list_body[i]
            f = open(news_folder+str(id)+'.txt', 'w')
            f.write(news)
        except: pass


if __name__ == '__main__':

    dirname = '/data/share_data/remotedata/News/SinaFinance/body'
    news_folder = '/home/lixinhang/news_folder/'
    date_list = os.listdir(dirname)

    # Parallelization of the task
    nb_process = 100
    process_list = [mp.Process(target=process_news,
                               args=(date,)) for date in date_list]

    for p in process_list:
        p.start()

    for p in process_list:
        p.join()


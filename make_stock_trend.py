import os
import pickle
import pandas as pd
import numpy as np


def get_stock_value(dirname, value_folder):

    stock = {}
    i = 0
    day_list = []
    dl = os.listdir(dirname)
    for d in dl:
        ds = pd.to_datetime(d[:-4])
        if ds > pd.datetime(2014, 1, 1):
            day_list.append(d)

    total = len(day_list)
    print('Total days:', total)
    for day in day_list:
        day_path = os.path.join(dirname, day)
        df = pd.read_csv(day_path)
        df = df.iloc[:, 1:]
        df_list_ID = df['SecuCode'].tolist()

        for number in df_list_ID:
            if number not in stock:
                stock[number] = []
            stock[number].append(df.loc[df['SecuCode'] == number,
                                        ['TradingDay', 'OpenPrice']].values.tolist())

        i += 1
        print('Get day:', i, '/', total)

    i = 0
    total = len(stock)
    print('Total stocks:', total)
    for number in stock:
        with open('{}{}{}'.format(value_folder, number, ".pkl"), 'wb') as f:
            data = pd.DataFrame(np.array(stock[number]).reshape(len(stock[number]), 2),
                                columns=['TradingDay', 'OpenPrice'])
            data['TradingDay'] = pd.to_datetime(data['TradingDay'])
            data.sort_values('TradingDay', inplace=True)
            pickle.dump(data, f)

        i += 1
        print('Get stock:', i, '/', total)


def get_stock_move(dirname, move_folder):

    i = 0
    total = len(os.listdir(dirname))
    print('Total stocks:', total)
    for firm in os.listdir(dirname):
        stock_move = {}
        firm_value = os.path.join(dirname, firm)
        df = pickle.load(open(firm_value, 'rb'))
        print(firm[:-4], 'is running')
        for index in range(0, len(df)-1):
            past = float(df.iloc[index]['OpenPrice'])
            now = float(df.iloc[index+1]['OpenPrice'])
            if past == 0 or now == 0:
                pass
            else:
                raise_t = (now-past)/past
                if raise_t > 0.0087:
                    stock_move[df.iloc[index]['TradingDay']] = 1
                elif raise_t < -0.0041:
                    stock_move[df.iloc[index]['TradingDay']] = -1
                else:
                    stock_move[df.iloc[index]['TradingDay']] = 0

        with open('{}{}'.format(move_folder, firm), 'wb') as f:
            pickle.dump(stock_move, f)

        i += 1
        print('Get move:', i, '/', total)


if __name__ == '__main__':

    #print('Pipeline Start!')
    #get_stock_value('/home/lixinhang/pv', '/home/lixinhang/value_folder/')
    #print('Get stock value done!')
    get_stock_move('/home/lixinhang/value_folder', '/home/lixinhang/move_folder/')
    print('Get stock move done!')

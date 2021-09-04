import pandas as pd 
import numpy as np

from os import listdir
from os.path import isfile, join

import constants


path = constants.DATA
list_data = [f for f in listdir(path) if isfile(join(path, f))]

noise_type = 2

for data_file in list_data:

    print(data_file)
  
    df = pd.read_csv(constants.DATA+data_file)

    # df['time'] = pd.to_datetime(df['time'])
    # df = df.set_index("time")

    df = df.assign(close_noise=0)
    noise = np.random.normal(0,df["close"].std(),len(df)) * 0.01
    df['close_noise'] = df['close'] + noise
    if noise_type==2:
        df['close_noise'] = df['close']

    for i in range(2):

        if i==1:
            postfix = "_true"
            df['close_noise'+postfix] = df['close']
        else:
            postfix = ""

        df = df.assign(diff_consecutive=df['close_noise'+postfix].shift(-1) - df['close_noise'+postfix])
        max_consecutive_diff = df['diff_consecutive'].max()
        min_consecutive_diff = df['diff_consecutive'].min()
        df = df.drop(['diff_consecutive'], axis=1)

        df['shift12'] = df['close_noise'+postfix].shift(-12)
        df['shift11'] = df['close_noise'+postfix].shift(-11)
        df['shift10'] = df['close_noise'+postfix].shift(-10)
        df['shift13'] = df['close_noise'+postfix].shift(-13)
        df['shift14'] = df['close_noise'+postfix].shift(-14)
        maximum = df[['shift12', 'shift11', 'shift10', 'shift13', 'shift14']].max(axis=1)
        minimum = df[['shift12', 'shift11', 'shift10', 'shift13', 'shift14']].min(axis=1)
        df = df.drop(['shift12', 'shift11', 'shift10', 'shift13', 'shift14'], axis=1)
        df = df.assign(diff_consecutive=maximum - df['close_noise'+postfix])
        max_consecutive_diff_mid = df['diff_consecutive'].max()
        df = df.drop(['diff_consecutive'], axis=1)
        df = df.assign(diff_consecutive=minimum - df['close_noise'+postfix])
        min_consecutive_diff_mid = df['diff_consecutive'].min()
        df = df.drop(['diff_consecutive'], axis=1)

        df['shift48'] = df['close_noise'+postfix].shift(-48)
        df['shift47'] = df['close_noise'+postfix].shift(-47)
        df['shift46'] = df['close_noise'+postfix].shift(-46)
        df['shift45'] = df['close_noise'+postfix].shift(-45)
        df['shift44'] = df['close_noise'+postfix].shift(-44)
        df['shift43'] = df['close_noise'+postfix].shift(-43)
        df['shift49'] = df['close_noise'+postfix].shift(-49)
        df['shift50'] = df['close_noise'+postfix].shift(-50)
        df['shift51'] = df['close_noise'+postfix].shift(-51)
        df['shift52'] = df['close_noise'+postfix].shift(-52)
        df['shift53'] = df['close_noise'+postfix].shift(-53)
        maximum = df[['shift48', 'shift47', 'shift46', 'shift45', 'shift44', 'shift43', 'shift49', 'shift50', 'shift51', 'shift52', 'shift53']].max(axis=1)
        minimum = df[['shift48', 'shift47', 'shift46', 'shift45', 'shift44', 'shift43', 'shift49', 'shift50', 'shift51', 'shift52', 'shift53']].min(axis=1)
        df = df.drop(['shift48', 'shift47', 'shift46', 'shift45', 'shift44', 'shift43', 'shift49', 'shift50', 'shift51', 'shift52', 'shift53'], axis=1)
        df = df.assign(diff_consecutive=maximum - df['close_noise'+postfix])
        max_consecutive_diff_long = df['diff_consecutive'].max()
        df = df.drop(['diff_consecutive'], axis=1)
        df = df.assign(diff_consecutive=minimum - df['close_noise'+postfix])
        min_consecutive_diff_long = df['diff_consecutive'].min()
        df = df.drop(['diff_consecutive'], axis=1)

        df = df.assign(prediction_short=0)
        df.loc[(df['close_noise'+postfix].shift(-1) - df['close_noise'+postfix])>0, 'prediction_short'] = 2
        df.loc[(df['close_noise'+postfix].shift(-1) - df['close_noise'+postfix])<0, 'prediction_short'] = 1

        df = df.assign(prediction_mid=0)
        df.loc[((df['close_noise'+postfix].shift(-12) - df['close_noise'+postfix])>0) | 
                ((df['close_noise'+postfix].shift(-11) - df['close_noise'+postfix])>0) |
                ((df['close_noise'+postfix].shift(-10) - df['close_noise'+postfix])>0) |
                ((df['close_noise'+postfix].shift(-13) - df['close_noise'+postfix])>0) |
                ((df['close_noise'+postfix].shift(-14) - df['close_noise'+postfix])>0), 'prediction_mid'] = 2
        df.loc[((df['close_noise'+postfix].shift(-12) - df['close_noise'+postfix])<0) | 
                ((df['close_noise'+postfix].shift(-11) - df['close_noise'+postfix])<0) |
                ((df['close_noise'+postfix].shift(-10) - df['close_noise'+postfix])<0) |
                ((df['close_noise'+postfix].shift(-13) - df['close_noise'+postfix])<0) |
                ((df['close_noise'+postfix].shift(-14) - df['close_noise'+postfix])<0), 'prediction_mid'] = 1

        df = df.assign(prediction_long=0)
        df.loc[((df['close_noise'+postfix].shift(-48) - df['close_noise'+postfix])>0) | 
                ((df['close_noise'+postfix].shift(-49) - df['close_noise'+postfix])>0) |
                ((df['close_noise'+postfix].shift(-50) - df['close_noise'+postfix])>0) |
                ((df['close_noise'+postfix].shift(-51) - df['close_noise'+postfix])>0) |
                ((df['close_noise'+postfix].shift(-52) - df['close_noise'+postfix])>0) |
                ((df['close_noise'+postfix].shift(-53) - df['close_noise'+postfix])>0) |
                ((df['close_noise'+postfix].shift(-47) - df['close_noise'+postfix])>0) |
                ((df['close_noise'+postfix].shift(-46) - df['close_noise'+postfix])>0) |
                ((df['close_noise'+postfix].shift(-45) - df['close_noise'+postfix])>0) |
                ((df['close_noise'+postfix].shift(-44) - df['close_noise'+postfix])>0) |
                ((df['close_noise'+postfix].shift(-43) - df['close_noise'+postfix])>0), 'prediction_long'] = 2
        df.loc[((df['close_noise'+postfix].shift(-48) - df['close_noise'+postfix])<0) | 
                ((df['close_noise'+postfix].shift(-49) - df['close_noise'+postfix])<0) |
                ((df['close_noise'+postfix].shift(-50) - df['close_noise'+postfix])<0) |
                ((df['close_noise'+postfix].shift(-51) - df['close_noise'+postfix])<0) |
                ((df['close_noise'+postfix].shift(-52) - df['close_noise'+postfix])<0) |
                ((df['close_noise'+postfix].shift(-53) - df['close_noise'+postfix])<0) |
                ((df['close_noise'+postfix].shift(-47) - df['close_noise'+postfix])<0) |
                ((df['close_noise'+postfix].shift(-46) - df['close_noise'+postfix])<0) |
                ((df['close_noise'+postfix].shift(-45) - df['close_noise'+postfix])<0) |
                ((df['close_noise'+postfix].shift(-44) - df['close_noise'+postfix])<0) |
                ((df['close_noise'+postfix].shift(-43) - df['close_noise'+postfix])<0), 'prediction_long'] = 1

        df = df.assign(probability_short=0)
        df.loc[df['prediction_short']==2, 'probability_short'] = (df['close_noise'+postfix].shift(-1) - df['close_noise'+postfix])/max_consecutive_diff
        df.loc[df['prediction_short']==1, 'probability_short'] = (df['close_noise'+postfix].shift(-1) - df['close_noise'+postfix])/min_consecutive_diff
        df = df.assign(probability_mid=0)
        df['shift12'] = df['close_noise'+postfix].shift(-12)
        df['shift11'] = df['close_noise'+postfix].shift(-11)
        df['shift10'] = df['close_noise'+postfix].shift(-10)
        df['shift13'] = df['close_noise'+postfix].shift(-13)
        df['shift14'] = df['close_noise'+postfix].shift(-14)
        maximum = df[['shift12', 'shift11', 'shift10', 'shift13', 'shift14']].max(axis=1)
        minimum = df[['shift12', 'shift11', 'shift10', 'shift13', 'shift14']].min(axis=1)
        df = df.drop(['shift12', 'shift11', 'shift10', 'shift13', 'shift14'], axis=1)
        df.loc[df['prediction_mid']==2, 'probability_mid'] = (maximum - df['close_noise'+postfix])/max_consecutive_diff_mid
        df.loc[df['prediction_mid']==1, 'probability_mid'] = (minimum - df['close_noise'+postfix])/min_consecutive_diff_mid
        df = df.assign(probability_long=0)
        df['shift48'] = df['close_noise'+postfix].shift(-48)
        df['shift47'] = df['close_noise'+postfix].shift(-47)
        df['shift46'] = df['close_noise'+postfix].shift(-46)
        df['shift45'] = df['close_noise'+postfix].shift(-45)
        df['shift44'] = df['close_noise'+postfix].shift(-44)
        df['shift43'] = df['close_noise'+postfix].shift(-43)
        df['shift49'] = df['close_noise'+postfix].shift(-49)
        df['shift50'] = df['close_noise'+postfix].shift(-50)
        df['shift51'] = df['close_noise'+postfix].shift(-51)
        df['shift52'] = df['close_noise'+postfix].shift(-52)
        df['shift53'] = df['close_noise'+postfix].shift(-53)
        maximum = df[['shift48', 'shift47', 'shift46', 'shift45', 'shift44', 'shift43', 'shift49', 'shift50', 'shift51', 'shift52', 'shift53']].max(axis=1)
        minimum = df[['shift48', 'shift47', 'shift46', 'shift45', 'shift44', 'shift43', 'shift49', 'shift50', 'shift51', 'shift52', 'shift53']].min(axis=1)
        df = df.drop(['shift48', 'shift47', 'shift46', 'shift45', 'shift44', 'shift43', 'shift49', 'shift50', 'shift51', 'shift52', 'shift53'], axis=1)
        df.loc[df['prediction_long']==2, 'probability_long'] = (maximum - df['close_noise'+postfix])/max_consecutive_diff_long
        df.loc[df['prediction_long']==1, 'probability_long'] = (minimum - df['close_noise'+postfix])/min_consecutive_diff_long

        if i == 0:
            df = df.assign(buy_probability_short=0)
            df = df.assign(sell_probability_short=0)
            df = df.assign(buy_probability_mid=0)
            df = df.assign(sell_probability_mid=0)
            df = df.assign(buy_probability_long=0)
            df = df.assign(sell_probability_long=0)
        else:
            df = df.assign(buy_probability_short_true=0)
            df = df.assign(sell_probability_short_true=0)
            df = df.assign(buy_probability_mid_true=0)
            df = df.assign(sell_probability_mid_true=0)
            df = df.assign(buy_probability_long_true=0)
            df = df.assign(sell_probability_long_true=0)

        if noise_type==2:
            if i==0:
                df = df.assign(random=0)
                random = np.random.random_sample(len(df))
                df['random'] = df['random'] + random
                df.loc[(df['prediction_short'] == 1) & (df['random'] < 0.35), 'prediction_short'] = -1
                df['random'] = df['random'] - random
                random = np.random.random_sample(len(df))
                df['random'] = df['random'] + random
                df.loc[(df['prediction_short'] == 2) & (df['random'] < 0.35), 'prediction_short'] = -2
                df['random'] = df['random'] - random
                random = np.random.random_sample(len(df))
                df['random'] = df['random'] + random
                df.loc[(df['prediction_mid'] == 1) & (df['random'] < 0.4), 'prediction_mid'] = -1
                df['random'] = df['random'] - random
                random = np.random.random_sample(len(df))
                df['random'] = df['random'] + random
                df.loc[(df['prediction_mid'] == 2) & (df['random'] < 0.4), 'prediction_mid'] = -2
                df['random'] = df['random'] - random
                random = np.random.random_sample(len(df))
                df['random'] = df['random'] + random
                df.loc[(df['prediction_long'] == 1) & (df['random'] < 0.45), 'prediction_long'] = -1
                df['random'] = df['random'] - random
                random = np.random.random_sample(len(df))
                df['random'] = df['random'] + random
                df.loc[(df['prediction_long'] == 2) & (df['random'] < 0.45), 'prediction_long'] = -2
                df['random'] = df['random'] - random
                df = df.drop(['random'], axis=1)
                df.loc[(df['prediction_short'] == -1), 'prediction_short'] = 2
                df.loc[(df['prediction_short'] == -2), 'prediction_short'] = 1
                df.loc[(df['prediction_mid'] == -1), 'prediction_mid'] = 2
                df.loc[(df['prediction_mid'] == -2), 'prediction_mid'] = 1
                df.loc[(df['prediction_long'] == -1), 'prediction_long'] = 2
                df.loc[(df['prediction_long'] == -2), 'prediction_long'] = 1

        df['buy_probability_short'+postfix] += df['probability_short']*(df['prediction_short']-1)
        df['sell_probability_short'+postfix] += (-1)*df['probability_short']*(df['prediction_short']-2)
        df['buy_probability_mid'+postfix] += df['probability_mid']*(df['prediction_mid']-1)
        df['sell_probability_mid'+postfix] += (-1)*df['probability_mid']*(df['prediction_mid']-2)
        df['buy_probability_long'+postfix] += df['probability_long']*(df['prediction_long']-1)
        df['sell_probability_long'+postfix] += (-1)*df['probability_long']*(df['prediction_long']-2)

        if i==0:
            if noise_type==1:
                df.loc[df['buy_probability_short'+postfix]<0.01, 'buy_probability_short'+postfix] = 0
                df.loc[df['sell_probability_short'+postfix]<0.01, 'sell_probability_short'+postfix] = 0
                df.loc[df['buy_probability_mid'+postfix]<0.02, 'buy_probability_mid'+postfix] = 0
                df.loc[df['sell_probability_mid'+postfix]<0.02, 'sell_probability_mid'+postfix] = 0
                df.loc[df['buy_probability_long'+postfix]<0.03, 'buy_probability_long'+postfix] = 0
                df.loc[df['sell_probability_long'+postfix]<0.03, 'sell_probability_long'+postfix] = 0
            else:
                df.loc[df['buy_probability_short'+postfix]<0.001, 'buy_probability_short'+postfix] = 0
                df.loc[df['sell_probability_short'+postfix]<0.001, 'sell_probability_short'+postfix] = 0
                df.loc[df['buy_probability_mid'+postfix]<0.002, 'buy_probability_mid'+postfix] = 0
                df.loc[df['sell_probability_mid'+postfix]<0.002, 'sell_probability_mid'+postfix] = 0
                df.loc[df['buy_probability_long'+postfix]<0.003, 'buy_probability_long'+postfix] = 0
                df.loc[df['sell_probability_long'+postfix]<0.003, 'sell_probability_long'+postfix] = 0

            bf = df[(df['close'].shift(-1) - df['close'] > 0)]
            print(bf.shape[0])
            bf = df[(df['close'].shift(-1) - df['close'] < 0)]
            print(bf.shape[0])
            print(df[df['buy_probability_short'+postfix] > 0].shape[0])
            print(df[df['sell_probability_short'+postfix] > 0].shape[0])
            print(df[df['buy_probability_mid'+postfix] > 0].shape[0])
            print(df[df['sell_probability_mid'+postfix] > 0].shape[0])
            print(df[df['buy_probability_long'+postfix] > 0].shape[0])
            print(df[df['sell_probability_long'+postfix] > 0].shape[0])

        df = df.drop(['probability_short', 'prediction_short'], axis=1)
        df = df.drop(['probability_mid', 'prediction_mid'], axis=1)
        df = df.drop(['probability_long', 'prediction_long'], axis=1)

    count1 = 0
    count2 = 0
    count3 = 0
    count_total1 = 0
    count_total2 = 0
    count_total3 = 0
    for index, row in df.iterrows():
        if(row['buy_probability_short'] > 0) or (row['sell_probability_short'] > 0):
            count_total1 += 1
        if(row['buy_probability_mid'] > 0) or (row['sell_probability_mid'] > 0):
            count_total2 += 1
        if(row['buy_probability_long'] > 0) or (row['sell_probability_long'] > 0):
            count_total3 += 1
        if(row['buy_probability_short'] > 0) and (row['buy_probability_short_true'] == 0):
            count1 += 1
        if(row['sell_probability_short'] > 0) and (row['sell_probability_short_true'] == 0):
            count1 += 1
        if(row['buy_probability_mid'] > 0) and (row['buy_probability_mid_true'] == 0):
            count2 += 1
        if(row['sell_probability_mid'] > 0) and (row['sell_probability_mid_true'] == 0):
            count2 += 1
        if(row['buy_probability_long'] > 0) and (row['buy_probability_long_true'] == 0):
            count3 += 1
        if(row['sell_probability_long'] > 0) and (row['sell_probability_long_true'] == 0):
            count3 += 1

    print(count1, count2, count3)
    print(count1/count_total1, count2/count_total2, count3/count_total3)
    print((count1+count2+count3)/(count_total1+count_total2+count_total3))

    df = df.drop(['buy_probability_short_true', 'sell_probability_short_true'], axis=1)
    df = df.drop(['buy_probability_mid_true', 'sell_probability_mid_true'], axis=1)
    df = df.drop(['buy_probability_long_true', 'sell_probability_long_true'], axis=1)
    df = df.drop(['close_noise_true'], axis=1)

    # normalize data in signal columns
    from sklearn import preprocessing
    min_max_scaler = preprocessing.MinMaxScaler()
    column_names_to_normalize = ['buy_probability_short', 'sell_probability_short', 'buy_probability_mid', 'sell_probability_mid', 'buy_probability_long', 'sell_probability_long']
    x = df[column_names_to_normalize].values
    x_scaled = min_max_scaler.fit_transform(x)
    df_temp = pd.DataFrame(x_scaled, columns=column_names_to_normalize, index = df.index)
    df[column_names_to_normalize] = df_temp

    for column in df:
        if column in column_names_to_normalize:
            df.loc[df[column] > 0.1, column] = 1
            df.loc[df[column] <= 0.1, column] = df[column] * 10

    # print(df.head(30))

    # print(df.count())

    if noise_type==2:
        df.to_csv(constants.DATA+data_file[:-4]+'_signal2.csv')
    else:
        df.to_csv(constants.DATA+data_file[:-4]+'_signal.csv')

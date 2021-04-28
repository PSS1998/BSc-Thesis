import pandas as pd 
import numpy as np

from os import listdir
from os.path import isfile, join

import constants


path = constants.DATA
list_data = [f for f in listdir(path) if isfile(join(path, f))]

for data_file in list_data:

    print(data_file)
  
    df = pd.read_csv("data/"+data_file)

    # df['time'] = pd.to_datetime(df['time'])
    # df = df.set_index("time")

    df = df.assign(close_noise=0)
    noise = np.random.normal(0,df["close"].std(),len(df)) * 0.015
    df['close_noise'] = df['close'] + noise
    # df['close_noise'] = df['close']

    df = df.assign(diff_consecutive=df['close_noise'].shift(-1) - df['close_noise'])
    max_consecutive_diff = df['diff_consecutive'].max()
    min_consecutive_diff = df['diff_consecutive'].min()
    df = df.drop(['diff_consecutive'], axis=1)

    df['shift12'] = df['close_noise'].shift(-12)
    df['shift11'] = df['close_noise'].shift(-11)
    df['shift10'] = df['close_noise'].shift(-10)
    df['shift13'] = df['close_noise'].shift(-13)
    df['shift14'] = df['close_noise'].shift(-14)
    maximum = df[['shift12', 'shift11', 'shift10', 'shift13', 'shift14']].max(axis=1)
    minimum = df[['shift12', 'shift11', 'shift10', 'shift13', 'shift14']].min(axis=1)
    df = df.drop(['shift12', 'shift11', 'shift10', 'shift13', 'shift14'], axis=1)
    df = df.assign(diff_consecutive=maximum - df['close_noise'])
    max_consecutive_diff_mid = df['diff_consecutive'].max()
    df = df.drop(['diff_consecutive'], axis=1)
    df = df.assign(diff_consecutive=minimum - df['close_noise'])
    min_consecutive_diff_mid = df['diff_consecutive'].min()
    df = df.drop(['diff_consecutive'], axis=1)

    df['shift48'] = df['close_noise'].shift(-48)
    df['shift47'] = df['close_noise'].shift(-47)
    df['shift46'] = df['close_noise'].shift(-46)
    df['shift45'] = df['close_noise'].shift(-45)
    df['shift44'] = df['close_noise'].shift(-44)
    df['shift43'] = df['close_noise'].shift(-43)
    df['shift49'] = df['close_noise'].shift(-49)
    df['shift50'] = df['close_noise'].shift(-50)
    df['shift51'] = df['close_noise'].shift(-51)
    df['shift52'] = df['close_noise'].shift(-52)
    df['shift53'] = df['close_noise'].shift(-53)
    maximum = df[['shift48', 'shift47', 'shift46', 'shift45', 'shift44', 'shift43', 'shift49', 'shift50', 'shift51', 'shift52', 'shift53']].max(axis=1)
    minimum = df[['shift48', 'shift47', 'shift46', 'shift45', 'shift44', 'shift43', 'shift49', 'shift50', 'shift51', 'shift52', 'shift53']].min(axis=1)
    df = df.drop(['shift48', 'shift47', 'shift46', 'shift45', 'shift44', 'shift43', 'shift49', 'shift50', 'shift51', 'shift52', 'shift53'], axis=1)
    df = df.assign(diff_consecutive=maximum - df['close_noise'])
    max_consecutive_diff_long = df['diff_consecutive'].max()
    df = df.drop(['diff_consecutive'], axis=1)
    df = df.assign(diff_consecutive=minimum - df['close_noise'])
    min_consecutive_diff_long = df['diff_consecutive'].min()
    df = df.drop(['diff_consecutive'], axis=1)

    df = df.assign(prediction_short=0)
    df.loc[(df['close_noise'].shift(-1) - df['close_noise'])>0, 'prediction_short'] = 2
    df.loc[(df['close_noise'].shift(-1) - df['close_noise'])<0, 'prediction_short'] = 1

    df = df.assign(prediction_mid=0)
    df.loc[((df['close_noise'].shift(-12) - df['close_noise'])>0) | 
           ((df['close_noise'].shift(-11) - df['close_noise'])>0) |
           ((df['close_noise'].shift(-10) - df['close_noise'])>0) |
           ((df['close_noise'].shift(-13) - df['close_noise'])>0) |
           ((df['close_noise'].shift(-14) - df['close_noise'])>0), 'prediction_mid'] = 2
    df.loc[((df['close_noise'].shift(-12) - df['close_noise'])<0) | 
           ((df['close_noise'].shift(-11) - df['close_noise'])<0) |
           ((df['close_noise'].shift(-10) - df['close_noise'])<0) |
           ((df['close_noise'].shift(-13) - df['close_noise'])<0) |
           ((df['close_noise'].shift(-14) - df['close_noise'])<0), 'prediction_mid'] = 1

    df = df.assign(prediction_long=0)
    df.loc[((df['close_noise'].shift(-48) - df['close_noise'])>0) | 
           ((df['close_noise'].shift(-49) - df['close_noise'])>0) |
           ((df['close_noise'].shift(-50) - df['close_noise'])>0) |
           ((df['close_noise'].shift(-51) - df['close_noise'])>0) |
           ((df['close_noise'].shift(-52) - df['close_noise'])>0) |
           ((df['close_noise'].shift(-53) - df['close_noise'])>0) |
           ((df['close_noise'].shift(-47) - df['close_noise'])>0) |
           ((df['close_noise'].shift(-46) - df['close_noise'])>0) |
           ((df['close_noise'].shift(-45) - df['close_noise'])>0) |
           ((df['close_noise'].shift(-44) - df['close_noise'])>0) |
           ((df['close_noise'].shift(-43) - df['close_noise'])>0), 'prediction_long'] = 2
    df.loc[((df['close_noise'].shift(-48) - df['close_noise'])<0) | 
           ((df['close_noise'].shift(-49) - df['close_noise'])<0) |
           ((df['close_noise'].shift(-50) - df['close_noise'])<0) |
           ((df['close_noise'].shift(-51) - df['close_noise'])<0) |
           ((df['close_noise'].shift(-52) - df['close_noise'])<0) |
           ((df['close_noise'].shift(-53) - df['close_noise'])<0) |
           ((df['close_noise'].shift(-47) - df['close_noise'])<0) |
           ((df['close_noise'].shift(-46) - df['close_noise'])<0) |
           ((df['close_noise'].shift(-45) - df['close_noise'])<0) |
           ((df['close_noise'].shift(-44) - df['close_noise'])<0) |
           ((df['close_noise'].shift(-43) - df['close_noise'])<0), 'prediction_long'] = 1

    df = df.assign(probability_short=0)
    df.loc[df['prediction_short']==2, 'probability_short'] = (df['close_noise'].shift(-1) - df['close_noise'])/max_consecutive_diff
    df.loc[df['prediction_short']==1, 'probability_short'] = (df['close_noise'].shift(-1) - df['close_noise'])/min_consecutive_diff
    df = df.assign(probability_mid=0)
    df['shift12'] = df['close_noise'].shift(-12)
    df['shift11'] = df['close_noise'].shift(-11)
    df['shift10'] = df['close_noise'].shift(-10)
    df['shift13'] = df['close_noise'].shift(-13)
    df['shift14'] = df['close_noise'].shift(-14)
    maximum = df[['shift12', 'shift11', 'shift10', 'shift13', 'shift14']].max(axis=1)
    minimum = df[['shift12', 'shift11', 'shift10', 'shift13', 'shift14']].min(axis=1)
    df = df.drop(['shift12', 'shift11', 'shift10', 'shift13', 'shift14'], axis=1)
    df.loc[df['prediction_mid']==2, 'probability_mid'] = (maximum - df['close_noise'])/max_consecutive_diff_mid
    df.loc[df['prediction_mid']==1, 'probability_mid'] = (minimum - df['close_noise'])/min_consecutive_diff_mid
    df = df.assign(probability_long=0)
    df['shift48'] = df['close_noise'].shift(-48)
    df['shift47'] = df['close_noise'].shift(-47)
    df['shift46'] = df['close_noise'].shift(-46)
    df['shift45'] = df['close_noise'].shift(-45)
    df['shift44'] = df['close_noise'].shift(-44)
    df['shift43'] = df['close_noise'].shift(-43)
    df['shift49'] = df['close_noise'].shift(-49)
    df['shift50'] = df['close_noise'].shift(-50)
    df['shift51'] = df['close_noise'].shift(-51)
    df['shift52'] = df['close_noise'].shift(-52)
    df['shift53'] = df['close_noise'].shift(-53)
    maximum = df[['shift48', 'shift47', 'shift46', 'shift45', 'shift44', 'shift43', 'shift49', 'shift50', 'shift51', 'shift52', 'shift53']].max(axis=1)
    minimum = df[['shift48', 'shift47', 'shift46', 'shift45', 'shift44', 'shift43', 'shift49', 'shift50', 'shift51', 'shift52', 'shift53']].min(axis=1)
    df = df.drop(['shift48', 'shift47', 'shift46', 'shift45', 'shift44', 'shift43', 'shift49', 'shift50', 'shift51', 'shift52', 'shift53'], axis=1)
    df.loc[df['prediction_long']==2, 'probability_long'] = (maximum - df['close_noise'])/max_consecutive_diff_long
    df.loc[df['prediction_long']==1, 'probability_long'] = (minimum - df['close_noise'])/min_consecutive_diff_long

    df = df.assign(buy_probability_short=0)
    df['buy_probability_short'] += df['probability_short']*(df['prediction_short']-1)
    df = df.assign(sell_probability_short=0)
    df['sell_probability_short'] += (-1)*df['probability_short']*(df['prediction_short']-2)
    df = df.assign(buy_probability_mid=0)
    df['buy_probability_mid'] += df['probability_mid']*(df['prediction_mid']-1)
    df = df.assign(sell_probability_mid=0)
    df['sell_probability_mid'] += (-1)*df['probability_mid']*(df['prediction_mid']-2)
    df = df.assign(buy_probability_long=0)
    df['buy_probability_long'] += df['probability_long']*(df['prediction_long']-1)
    df = df.assign(sell_probability_long=0)
    df['sell_probability_long'] += (-1)*df['probability_long']*(df['prediction_long']-2)

    df.loc[df['buy_probability_short']<0.05, 'buy_probability_short'] = 0
    df.loc[df['sell_probability_short']<0.05, 'sell_probability_short'] = 0
    df.loc[df['buy_probability_mid']<0.15, 'buy_probability_mid'] = 0
    df.loc[df['sell_probability_mid']<0.15, 'sell_probability_mid'] = 0
    df.loc[df['buy_probability_long']<0.25, 'buy_probability_long'] = 0
    df.loc[df['sell_probability_long']<0.25, 'sell_probability_long'] = 0

    # bf = df[(df['close'].shift(-1) - df['close'] > 0)]
    # print(bf[bf['buy_probability_short'] > 0].count())
    # print(bf[bf['sell_probability_short'] > 0].count())
    # bf = df[(df['close'].shift(-1) - df['close'] < 0)]
    # print(bf[bf['buy_probability_short'] > 0].count())
    # print(bf[bf['sell_probability_short'] > 0].count())

    df = df.drop(['probability_short', 'prediction_short'], axis=1)
    df = df.drop(['probability_mid', 'prediction_mid'], axis=1)
    df = df.drop(['probability_long', 'prediction_long'], axis=1)

    # print(df.head(30))

    df.to_csv('data/'+data_file[:-4]+'_signal_noise.csv')



    # signal without noise

    df = pd.read_csv("data/"+data_file)

    # df['time'] = pd.to_datetime(df['time'])
    # df = df.set_index("time")

    df = df.assign(close_noise=0)
    noise = np.random.normal(0,df["close"].std(),len(df)) * 0.015
    df['close_noise'] = df['close'] + noise
    df['close_noise'] = df['close']

    df = df.assign(diff_consecutive=df['close_noise'].shift(-1) - df['close_noise'])
    max_consecutive_diff = df['diff_consecutive'].max()
    min_consecutive_diff = df['diff_consecutive'].min()
    df = df.drop(['diff_consecutive'], axis=1)

    df['shift12'] = df['close_noise'].shift(-12)
    df['shift11'] = df['close_noise'].shift(-11)
    df['shift10'] = df['close_noise'].shift(-10)
    df['shift13'] = df['close_noise'].shift(-13)
    df['shift14'] = df['close_noise'].shift(-14)
    maximum = df[['shift12', 'shift11', 'shift10', 'shift13', 'shift14']].max(axis=1)
    minimum = df[['shift12', 'shift11', 'shift10', 'shift13', 'shift14']].min(axis=1)
    df = df.drop(['shift12', 'shift11', 'shift10', 'shift13', 'shift14'], axis=1)
    df = df.assign(diff_consecutive=maximum - df['close_noise'])
    max_consecutive_diff_mid = df['diff_consecutive'].max()
    df = df.drop(['diff_consecutive'], axis=1)
    df = df.assign(diff_consecutive=minimum - df['close_noise'])
    min_consecutive_diff_mid = df['diff_consecutive'].min()
    df = df.drop(['diff_consecutive'], axis=1)

    df['shift48'] = df['close_noise'].shift(-48)
    df['shift47'] = df['close_noise'].shift(-47)
    df['shift46'] = df['close_noise'].shift(-46)
    df['shift45'] = df['close_noise'].shift(-45)
    df['shift44'] = df['close_noise'].shift(-44)
    df['shift43'] = df['close_noise'].shift(-43)
    df['shift49'] = df['close_noise'].shift(-49)
    df['shift50'] = df['close_noise'].shift(-50)
    df['shift51'] = df['close_noise'].shift(-51)
    df['shift52'] = df['close_noise'].shift(-52)
    df['shift53'] = df['close_noise'].shift(-53)
    maximum = df[['shift48', 'shift47', 'shift46', 'shift45', 'shift44', 'shift43', 'shift49', 'shift50', 'shift51', 'shift52', 'shift53']].max(axis=1)
    minimum = df[['shift48', 'shift47', 'shift46', 'shift45', 'shift44', 'shift43', 'shift49', 'shift50', 'shift51', 'shift52', 'shift53']].min(axis=1)
    df = df.drop(['shift48', 'shift47', 'shift46', 'shift45', 'shift44', 'shift43', 'shift49', 'shift50', 'shift51', 'shift52', 'shift53'], axis=1)
    df = df.assign(diff_consecutive=maximum - df['close_noise'])
    max_consecutive_diff_long = df['diff_consecutive'].max()
    df = df.drop(['diff_consecutive'], axis=1)
    df = df.assign(diff_consecutive=minimum - df['close_noise'])
    min_consecutive_diff_long = df['diff_consecutive'].min()
    df = df.drop(['diff_consecutive'], axis=1)

    df = df.assign(prediction_short=0)
    df.loc[(df['close_noise'].shift(-1) - df['close_noise'])>0, 'prediction_short'] = 2
    df.loc[(df['close_noise'].shift(-1) - df['close_noise'])<0, 'prediction_short'] = 1

    df = df.assign(prediction_mid=0)
    df.loc[((df['close_noise'].shift(-12) - df['close_noise'])>0) | 
           ((df['close_noise'].shift(-11) - df['close_noise'])>0) |
           ((df['close_noise'].shift(-10) - df['close_noise'])>0) |
           ((df['close_noise'].shift(-13) - df['close_noise'])>0) |
           ((df['close_noise'].shift(-14) - df['close_noise'])>0), 'prediction_mid'] = 2
    df.loc[((df['close_noise'].shift(-12) - df['close_noise'])<0) | 
           ((df['close_noise'].shift(-11) - df['close_noise'])<0) |
           ((df['close_noise'].shift(-10) - df['close_noise'])<0) |
           ((df['close_noise'].shift(-13) - df['close_noise'])<0) |
           ((df['close_noise'].shift(-14) - df['close_noise'])<0), 'prediction_mid'] = 1

    df = df.assign(prediction_long=0)
    df.loc[((df['close_noise'].shift(-48) - df['close_noise'])>0) | 
           ((df['close_noise'].shift(-49) - df['close_noise'])>0) |
           ((df['close_noise'].shift(-50) - df['close_noise'])>0) |
           ((df['close_noise'].shift(-51) - df['close_noise'])>0) |
           ((df['close_noise'].shift(-52) - df['close_noise'])>0) |
           ((df['close_noise'].shift(-53) - df['close_noise'])>0) |
           ((df['close_noise'].shift(-47) - df['close_noise'])>0) |
           ((df['close_noise'].shift(-46) - df['close_noise'])>0) |
           ((df['close_noise'].shift(-45) - df['close_noise'])>0) |
           ((df['close_noise'].shift(-44) - df['close_noise'])>0) |
           ((df['close_noise'].shift(-43) - df['close_noise'])>0), 'prediction_long'] = 2
    df.loc[((df['close_noise'].shift(-48) - df['close_noise'])<0) | 
           ((df['close_noise'].shift(-49) - df['close_noise'])<0) |
           ((df['close_noise'].shift(-50) - df['close_noise'])<0) |
           ((df['close_noise'].shift(-51) - df['close_noise'])<0) |
           ((df['close_noise'].shift(-52) - df['close_noise'])<0) |
           ((df['close_noise'].shift(-53) - df['close_noise'])<0) |
           ((df['close_noise'].shift(-47) - df['close_noise'])<0) |
           ((df['close_noise'].shift(-46) - df['close_noise'])<0) |
           ((df['close_noise'].shift(-45) - df['close_noise'])<0) |
           ((df['close_noise'].shift(-44) - df['close_noise'])<0) |
           ((df['close_noise'].shift(-43) - df['close_noise'])<0), 'prediction_long'] = 1

    df = df.assign(probability_short=0)
    df.loc[df['prediction_short']==2, 'probability_short'] = (df['close_noise'].shift(-1) - df['close_noise'])/max_consecutive_diff
    df.loc[df['prediction_short']==1, 'probability_short'] = (df['close_noise'].shift(-1) - df['close_noise'])/min_consecutive_diff
    df = df.assign(probability_mid=0)
    df['shift12'] = df['close_noise'].shift(-12)
    df['shift11'] = df['close_noise'].shift(-11)
    df['shift10'] = df['close_noise'].shift(-10)
    df['shift13'] = df['close_noise'].shift(-13)
    df['shift14'] = df['close_noise'].shift(-14)
    maximum = df[['shift12', 'shift11', 'shift10', 'shift13', 'shift14']].max(axis=1)
    minimum = df[['shift12', 'shift11', 'shift10', 'shift13', 'shift14']].min(axis=1)
    df = df.drop(['shift12', 'shift11', 'shift10', 'shift13', 'shift14'], axis=1)
    df.loc[df['prediction_mid']==2, 'probability_mid'] = (maximum - df['close_noise'])/max_consecutive_diff_mid
    df.loc[df['prediction_mid']==1, 'probability_mid'] = (minimum - df['close_noise'])/min_consecutive_diff_mid
    df = df.assign(probability_long=0)
    df['shift48'] = df['close_noise'].shift(-48)
    df['shift47'] = df['close_noise'].shift(-47)
    df['shift46'] = df['close_noise'].shift(-46)
    df['shift45'] = df['close_noise'].shift(-45)
    df['shift44'] = df['close_noise'].shift(-44)
    df['shift43'] = df['close_noise'].shift(-43)
    df['shift49'] = df['close_noise'].shift(-49)
    df['shift50'] = df['close_noise'].shift(-50)
    df['shift51'] = df['close_noise'].shift(-51)
    df['shift52'] = df['close_noise'].shift(-52)
    df['shift53'] = df['close_noise'].shift(-53)
    maximum = df[['shift48', 'shift47', 'shift46', 'shift45', 'shift44', 'shift43', 'shift49', 'shift50', 'shift51', 'shift52', 'shift53']].max(axis=1)
    minimum = df[['shift48', 'shift47', 'shift46', 'shift45', 'shift44', 'shift43', 'shift49', 'shift50', 'shift51', 'shift52', 'shift53']].min(axis=1)
    df = df.drop(['shift48', 'shift47', 'shift46', 'shift45', 'shift44', 'shift43', 'shift49', 'shift50', 'shift51', 'shift52', 'shift53'], axis=1)
    df.loc[df['prediction_long']==2, 'probability_long'] = (maximum - df['close_noise'])/max_consecutive_diff_long
    df.loc[df['prediction_long']==1, 'probability_long'] = (minimum - df['close_noise'])/min_consecutive_diff_long

    df = df.assign(buy_probability_short=0)
    df['buy_probability_short'] += df['probability_short']*(df['prediction_short']-1)
    df = df.assign(sell_probability_short=0)
    df['sell_probability_short'] += (-1)*df['probability_short']*(df['prediction_short']-2)
    df = df.assign(buy_probability_mid=0)
    df['buy_probability_mid'] += df['probability_mid']*(df['prediction_mid']-1)
    df = df.assign(sell_probability_mid=0)
    df['sell_probability_mid'] += (-1)*df['probability_mid']*(df['prediction_mid']-2)
    df = df.assign(buy_probability_long=0)
    df['buy_probability_long'] += df['probability_long']*(df['prediction_long']-1)
    df = df.assign(sell_probability_long=0)
    df['sell_probability_long'] += (-1)*df['probability_long']*(df['prediction_long']-2)

    df.loc[df['buy_probability_short']<0.05, 'buy_probability_short'] = 0
    df.loc[df['sell_probability_short']<0.05, 'sell_probability_short'] = 0
    df.loc[df['buy_probability_mid']<0.15, 'buy_probability_mid'] = 0
    df.loc[df['sell_probability_mid']<0.15, 'sell_probability_mid'] = 0
    df.loc[df['buy_probability_long']<0.25, 'buy_probability_long'] = 0
    df.loc[df['sell_probability_long']<0.25, 'sell_probability_long'] = 0

    # bf = df[(df['close'].shift(-1) - df['close'] > 0)]
    # print(bf[bf['buy_probability_short'] > 0].count())
    # print(bf[bf['sell_probability_short'] > 0].count())
    # bf = df[(df['close'].shift(-1) - df['close'] < 0)]
    # print(bf[bf['buy_probability_short'] > 0].count())
    # print(bf[bf['sell_probability_short'] > 0].count())

    df = df.drop(['probability_short', 'prediction_short'], axis=1)
    df = df.drop(['probability_mid', 'prediction_mid'], axis=1)
    df = df.drop(['probability_long', 'prediction_long'], axis=1)

    # print(df.head(30))

    df.to_csv('data/'+data_file[:-4]+'_signal.csv')

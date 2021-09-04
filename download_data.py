import json
import datetime
import os.path

import pandas as pd

import api
import utility
import config
import constants




class data_downloader_csv():
    def __init__(self):
        self.API = api.API()

    def _ohlcv_load_from_dict(self, ticker):
        ticker = pd.DataFrame(ticker)
        if(ticker.columns[4] == 's'):
            ticker.drop(ticker.columns[4], axis=1, inplace=True)
            ticker.columns = ['close', 'high', 'low', 'open', 'date', 'volume']
            # ticker.columns = ['close', 'high', 'low', 'open', 'date']
        else:
            ticker.columns = ['close', 'high', 'low', 'open', 'date', 'volume']
            # ticker.columns = ['close', 'high', 'low', 'open', 'date']
        ticker.sort_values('date', inplace=True)
        return ticker

    def _fetch_ticker(self, ticker_name, from_date=0, to_date=0):
        timeframe = config.TIMEFRAME
        if to_date==0:
            to_time = datetime.datetime.now()
            to_time = int(to_time.replace(tzinfo=datetime.timezone.utc).timestamp())
        else:
            to_time = to_date
        if from_date==0:
            from_time = datetime.datetime.now() - datetime.timedelta(30)
            from_time = int(from_time.replace(tzinfo=datetime.timezone.utc).timestamp())
        else:
            from_time = from_date
        ticker = self.API.get_candles(ticker_name, timeframe, from_time, to_time)
        if(ticker['s']=="no_data"):
            raise exceptions.FinnhubRequestException("No Data!")
        ticker = self._ohlcv_load_from_dict(ticker)
        return ticker

    def _ohlcv_save_to_json(self, ticker, ticker_name):
        timeframe = config.TIMEFRAME
        ticker.sort_index(inplace=True)
        ticker['date'] = ticker['date'].apply(lambda x: datetime.datetime.utcfromtimestamp(x).strftime("%Y/%m/%d %H:%M:%S"))
        ticker.insert(0, "time", ticker['date'])
        ticker = ticker.drop('date', axis=1)
        name = constants.DATA+ticker_name+":"+timeframe+'.csv'
        name = name.replace(":","_")
        ticker.to_csv(name, index = False)

    def download_data(self, from_date=0):
        pair_list = config.PAIR_LIST
        pair_list = pair_list.split()
        for i in range(len(pair_list)):
            ticker_name = pair_list[i]
            ticker = self._fetch_ticker(ticker_name, from_date=from_date)
            self._ohlcv_save_to_json(ticker, ticker_name)
    

from_date = 0
utility_functions = utility.utility()
if config.FROM_DATE != 0:
    from_date = utility_functions.parse_date(config.FROM_DATE)
Data_provider = data_downloader_csv()
Data_provider.download_data(from_date)

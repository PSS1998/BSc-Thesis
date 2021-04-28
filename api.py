import config
import finnhub_api
import utility

import time

class API():
    def __init__(self):
        self.finnhub_client = finnhub_api.Client(api_key=config.API)
        self.utility = utility.utility()

    def get_candles(self, ticker_name, timeframe, from_time, to_time):
        constant_time = 500
        # print("getting candles for " + ticker_name)
        # time.sleep(5)
        diff_time = to_time-from_time
        time_list = []
        if (diff_time>(self.utility.timeframe_to_timestamp()*constant_time)):
            new_to_time = from_time+self.utility.timeframe_to_timestamp()*constant_time
            candles = self.finnhub_client.crypto_candles(ticker_name, timeframe, from_time, new_to_time)
            diff_time = to_time-new_to_time
            while (diff_time>(self.utility.timeframe_to_timestamp()*constant_time)):
                # time.sleep(1)
                # print("Getting candles from " + new_from_time + " to " + new_to_time)
                new_from_time = new_to_time
                new_to_time = new_from_time+self.utility.timeframe_to_timestamp()*constant_time
                candles_temp = self.finnhub_client.crypto_candles(ticker_name, timeframe, new_from_time, new_to_time)
                for key, value in candles_temp.items():
                    if key == "s":
                        continue
                    candles[key].extend(value)
                diff_time = to_time-new_to_time
            candles_temp = self.finnhub_client.crypto_candles(ticker_name, timeframe, new_to_time, to_time)
            for key, value in candles_temp.items():
                if key == "s":
                    continue
                candles[key].extend(value)
        else:
            candles = self.finnhub_client.crypto_candles(ticker_name, timeframe, from_time, to_time)
        return candles
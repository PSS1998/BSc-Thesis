import time
import datetime
import json

import matplotlib.pylab as plt

import config

class utility():

    # input date fromat is like 2010/12/01 
    @staticmethod
    def parse_date(date):        
        return int(time.mktime(datetime.datetime.strptime(date, "%Y/%m/%d").timetuple()))

    @staticmethod
    def timestamp_to_date(timestamp):        
        return datetime.datetime.utcfromtimestamp(timestamp).strftime("%Y/%m/%d")

    @staticmethod
    def timeframe_to_timestamp():
        timestamp = 0
        timeframe = config.TIMEFRAME
        if timeframe=="1":
            timestamp = 1*60
        elif timeframe=="5":
            timestamp = 5*60
        elif timeframe=="10":
            timestamp = 10*60
        elif timeframe=="30":
            timestamp = 30*60
        elif timeframe=="H":
            timestamp = 60*60
        elif timeframe=="D":
            timestamp = 60*24*60
        return timestamp

    @staticmethod
    def analyze_profit(trades, money):
        money_change,time_change = money
        money_change = [x for _, x in sorted(zip(time_change,money_change), key=lambda pair: pair[0])]
        time_change = sorted(time_change)
        final_money_change_list = []
        final_money_change = 1000
        for i in range(len(time_change)):
            time_change[i] = utility().timestamp_to_date(time_change[i])
            final_money_change += money_change[i]*100
            final_money_change_list.append(final_money_change)
        total_profit = 0
        total_trade_num = 0
        for trade_name, trade in trades.items():
            if trade.num!=0:
                total_profit += trade.profit
                total_trade_num += trade.num
                print(trade_name + " " + "number of trades: " + str(trade.num) + ", profit: " + str(trade.profit))
        print("total profit = " + str(total_profit) + ", total number of trade = " + str(total_trade_num))
        plt.plot(time_change, final_money_change_list)
        plt.show()

    
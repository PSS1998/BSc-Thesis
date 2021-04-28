
import json
import requests

import time

from exceptions import FinnhubAPIException
from exceptions import FinnhubRequestException


class Client():
    API_URL = "https://finnhub.io/api/v1"
    DEFAULT_TIMEOUT = 10

    def __init__(self, api_key):
        self._session = self._init_session(api_key)

    @staticmethod
    def _init_session(api_key):
        session = requests.session()
        session.headers.update({"Accept": "application/json",
                                "User-Agent": "finnhub/python"})
        session.params["token"] = api_key

        return session

    def close(self):
        self._session.close()

    def _enter_(self):
        return self

    def _exit_(self, *exc):
        self.close()

    def _request(self, method, path, **kwargs):
        uri = "{}/{}".format(self.API_URL, path)
        kwargs["timeout"] = kwargs.get("timeout", self.DEFAULT_TIMEOUT)
        kwargs["params"] = self._format_params(kwargs.get("params", {}))

        # print(uri, kwargs)
        time.sleep(1)
        response = getattr(self._session, method)(uri, **kwargs)
        return self._handle_response(response)

    @staticmethod
    def _handle_response(response):
        if not response.ok:
            raise FinnhubAPIException(response)

        try:
            content_type = response.headers.get('Content-Type', '')
            if 'application/json' in content_type:
                return response.json()
            if 'text/csv' in content_type:
                return response.text
            raise FinnhubRequestException("Invalid Response: {}".format(response.text))
        except ValueError:
            raise FinnhubRequestException("Invalid Response: {}".format(response.text))

    @staticmethod
    def _merge_two_dicts(first, second):
        result = first.copy()
        result.update(second)
        return result

    @staticmethod
    def _format_params(params):
        return {k: json.dumps(v) if isinstance(v, bool) else v for k, v in params.items()}

    def _get(self, path, **kwargs):
        return self._request("get", path, **kwargs)

    @property
    def api_key(self):
        return self._session.params.get("token")

    @api_key.setter
    def api_key(self, token):
        self._session.params["token"] = token

    def covid19(self):
        return self._get("/covid19/us")

    def company_profile(self, **params):
        return self._get("/stock/profile", params=params)

    def company_profile2(self, **params):
        return self._get("/stock/profile2", params=params)

    def aggregate_indicator(self, symbol, resolution):
        return self._get("/scan/technical-indicator", params={
            "symbol": symbol,
            "resolution": resolution,
        })

    def crypto_exchanges(self):
        return self._get("/crypto/exchange")

    def forex_exchanges(self):
        return self._get("/forex/exchange")

    def major_developments(self, symbol, _from=None, to=None):
        return self._get("/major-development", params={
            "symbol": symbol,
            "from": _from,
            "to": to
        })

    def company_executive(self, symbol):
        return self._get("/stock/executive", params={"symbol": symbol})

    def stock_dividends(self, symbol, _from=None, to=None):
        return self._get("/stock/dividend", params={
            "symbol": symbol,
            "from": _from,
            "to": to
        })

    def stock_symbols(self, exchange):
        return self._get("/stock/symbol", params={"exchange": exchange})

    def recommendation_trends(self, symbol):
        return self._get("/stock/recommendation", params={"symbol": symbol})

    def price_target(self, symbol):
        return self._get("/stock/price-target", params={"symbol": symbol})

    def upgrade_downgrade(self, **params):
        return self._get("/stock/upgrade-downgrade", params=params)

    def option_chain(self, **params):
        return self._get("/stock/option-chain", params=params)

    def company_peers(self, symbol):
        return self._get("/stock/peers", params={"symbol": symbol})

    def company_basic_financials(self, symbol, metric):
        return self._get("/stock/metric", params={
            "symbol": symbol,
            "metric": metric
        })

    def financials(self, symbol, statement, freq):
        return self._get("/stock/financials", params={
            "symbol": symbol,
            "statement": statement,
            "freq": freq
        })

    def financials_reported(self, **params):
        return self._get("/stock/financials-reported", params=params)

    def fund_ownership(self, symbol, limit=None):
        return self._get("/stock/fund-ownership", params={
            "symbol": symbol,
            "limit": limit
        })

    def company_earnings(self, symbol, limit=None):
        return self._get("/stock/earnings", params={
            "symbol": symbol,
            "limit": limit
        })

    def company_revenue_estimates(self, symbol, freq=None):
        return self._get("/stock/revenue-estimate", params={
            "symbol": symbol,
            "freq": freq
        })

    def company_eps_estimates(self, symbol, freq=None):
        return self._get("/stock/eps-estimate", params={
            "symbol": symbol,
            "freq": freq
        })

    def exchange(self):
        return self._get("/stock/exchange")

    def filings(self, **params):
        return self._get("/stock/filings", params=params)

    def stock_symbol(self, **params):
        return self._get("/stock/symbol", params=params)

    def quote(self, symbol):
        return self._get("/quote", params={
            "symbol": symbol
        })

    def transcripts(self, _id):
        return self._get("/stock/transcripts", params={"id": _id})

    def transcripts_list(self, symbol):
        return self._get("/stock/transcripts/list", params={"symbol": symbol})

    def sim_index(self, **params):
        return self._get("/stock/similarity-index", params=params)

    def stock_candles(self, symbol, resolution, _from, to, **kwargs):
        params = self._merge_two_dicts({
            "symbol": symbol,
            "resolution": resolution,
            "from": _from,
            "to": to
        }, kwargs)

        return self._get("/stock/candle", params=params)

    def stock_tick(self, symbol, date, limit, skip, _format='json', **kwargs):
        params = self._merge_two_dicts({
            "symbol": symbol,
            "date": date,
            "limit": limit,
            "skip": skip,
            "format": _format
        }, kwargs)

        return self._get("/stock/tick", params=params)

    def forex_rates(self, **params):
        return self._get("/forex/rates", params=params)

    def forex_symbols(self, exchange):
        return self._get("/forex/symbol", params={
            "exchange": exchange
        })

    def forex_candles(self, symbol, resolution, _from, to, _format='json'):
        return self._get("/forex/candle", params={
            "symbol": symbol,
            "resolution": resolution,
            "from": _from,
            "to": to,
            "format": _format
        })

    def crypto_symbols(self, exchange):
        return self._get("/crypto/symbol", params={"exchange": exchange})

    def crypto_candles(self, symbol, resolution, _from, to, _format='json'):
        return self._get("crypto/candle", params={
            "symbol": symbol,
            "resolution": resolution,
            "from": _from,
            "to": to,
            "format": _format
        })

    def pattern_recognition(self, symbol, resolution):
        return self._get("/scan/pattern", params={
            "symbol": symbol,
            "resolution": resolution
        })

    def support_resistance(self, symbol, resolution):
        return self._get("/scan/support-resistance", params={
            "symbol": symbol,
            "resolution": resolution
        })

    def technical_indicator(self, symbol, resolution, _from, to, indicator, indicator_fields=None):
        indicator_fields = indicator_fields or {}
        params = self._merge_two_dicts({
            "symbol": symbol,
            "resolution": resolution,
            "from": _from,
            "to": to,
            "indicator": indicator
        }, indicator_fields)

        return self._get("/indicator", params=params)

    def stock_splits(self, symbol, _from, to):
        return self._get("/stock/split", params={
            "symbol": symbol,
            "from": _from,
            "to": to
        })

    def general_news(self, category, min_id=0):
        return self._get("/news", params={
            "category": category,
            "minId": min_id
        })

    def company_news(self, symbol, _from, to):
        return self._get("/company-news", params={
            "symbol": symbol,
            "from": _from,
            "to": to
        })

    def news_sentiment(self, symbol):
        return self._get("/news-sentiment", params={
            "symbol": symbol
        })

    def investors_ownership(self, symbol, limit=None):
        return self._get("/stock/investor-ownership", params={
            "symbol": symbol,
            "limit": limit
        })

    def country(self):
        return self._get("/country")

    def merger_country(self):
        return self._get("/merger/country")

    def merger(self, **params):
        return self._get("/merger", params=params)

    def economic_code(self):
        return self._get("/economic/code")

    def economic_data(self, code):
        return self._get("/economic", params={"code": code})

    def calendar_economic(self):
        return self._get("/calendar/economic")

    def earnings_calendar(self, **params):
        return self._get("/calendar/earnings", params=params)

    def ipo_calendar(self, _from, to):
        return self._get("/calendar/ipo", params={
            "from": _from,
            "to": to
        })

    def calendar_ico(self):
        return self._get("/calendar/ico")

    def indices_const(self, **params):
        return self._get("/index/constituents", params=params)

    def indices_hist_const(self, **params):
        return self._get("/index/historical-constituents", params=params)

    def etfs_profile(self, symbol):
        return self._get("/etf/profile", params={"symbol":symbol})

    def etfs_holdings(self, symbol):
        return self._get("/etf/holdings", params={"symbol":symbol})

    def etfs_ind_exp(self, symbol):
        return self._get("/etf/holdings", params={"symbol":symbol})

    def etfs_country_exp(self, symbol):
        return self._get("/etf/country", params={"symbol": symbol})


# finnhub_client = Client(api_key="bsgta37rh5rdc8pmr8dg")

# Forex exchanges
# print(finnhub_client.forex_exchanges())

# Forex all pairs
# print(finnhub_client.forex_rates(base='USD'))

# Forex symbols
# print(finnhub_client.forex_symbols('fxpro'))

# Forex candles
# ticker = finnhub_client.forex_candles('FXPRO:4', '5', 1593604800, 1596110400)
# ticker = data_handler.ohlcv_load_from_dict(ticker)
# print(ticker)


# List of fxpro symbols
# [{'description': 'FXPRO Canadian Dollar vs Swiss Franc CADCHF', 'displaySymbol': 'CAD/CHF', 'symbol': 'FXPRO:1061'}, {'description': 'FXPRO Silver Ounce vs Euro XAGEUR', 'displaySymbol': 'XAG/EUR', 'symbol': 'FXPRO:1109'}, {'description': 'FXPRO Silver Ounce vs US Dollar XAGUSD', 'displaySymbol': 'XAG/USD', 'symbol': 'FXPRO:42'}, {'description': 'FXPRO Gold Ounce vs Euro XAUEUR', 'displaySymbol': 'XAU/EUR', 'symbol': 'FXPRO:1108'}, {'description': 'FXPRO Australian Dollar vs Canadian Dollar AUDCAD', 'displaySymbol': 'AUD/CAD', 'symbol': 'FXPRO:18'}, {'description': 'FXPRO Gold Ounce vs US Dollar XAUUSD', 'displaySymbol': 'XAU/USD', 'symbol': 'FXPRO:41'}, {'description': 'FXPRO Gold Grams vs US Dollar  XAUUSDgr', 'displaySymbol': 'GAU/USD', 'symbol': 'FXPRO:1110'}, {'description': 'FXPRO Australia 200 Spot Index #AUS200', 'displaySymbol': 'ASX200/AUD', 'symbol': 'FXPRO:1274'}, {'description': 'FXPRO Platinum Ounce vs US Dollar XPTUSD', 'displaySymbol': 'XPT/USD', 'symbol': 'FXPRO:1297'}, {'description': 'FXPRO BitcoinCash vs US Dollar BCHUSD', 'displaySymbol': 'BCH/USD', 'symbol': 'FXPRO:1314'}, {'description': 'FXPRO Canadian Dollar vs Japanese Yen CADJPY', 'displaySymbol': 'CAD/JPY', 'symbol': 'FXPRO:15'}, {'description': 'FXPRO Australian Dollar vs Swiss Franc AUDCHF', 'displaySymbol': 'AUD/CHF', 'symbol': 'FXPRO:1038'}, {'description': 'FXPRO Euro Stoxx 50 Spot Index #Euro50', 'displaySymbol': 'Euro50/EUR', 'symbol': 'FXPRO:1103'}, {'description': 'FXPRO Bitcoin vs US Dollar BTCUSD', 'displaySymbol': 'BTC/USD', 'symbol': 'FXPRO:1310'}, {'description': 'FXPRO Swiss Franc vs Japanese Yen CHFJPY', 'displaySymbol': 'CHF/JPY', 'symbol': 'FXPRO:13'}, {'description': 'FXPRO Australian Dollar vs Danish Krone AUDDKK', 'displaySymbol': 'AUD/DKK', 'symbol': 'FXPRO:1289'}, {'description': 'FXPRO Brent (UK) Spot Oil BRENT', 'displaySymbol': 'XBR/USD', 'symbol': 'FXPRO:1117'}, {'description': 'FXPRO France CAC 40 Spot Index #France40', 'displaySymbol': 'France40/EUR', 'symbol': 'FXPRO:1104'}, {'description': 'FXPRO FTSE China A50 Spot Index #ChinaA50', 'displaySymbol': 'ChinaA50/USD', 'symbol': 'FXPRO:1100'}, {'description': 'FXPRO Ethereum vs US Dollar ETHUSD', 'displaySymbol': 'ETH/USD', 'symbol': 'FXPRO:1311'}, {'description': 'FXPRO Euro vs Canadian Dollar EURCAD', 'displaySymbol': 'EUR/CAD', 'symbol': 'FXPRO:17'}, {'description': 'FXPRO Australian Dollar vs Japanese Yen AUDJPY', 'displaySymbol': 'AUD/JPY', 'symbol': 'FXPRO:11'}, {'description': 'FXPRO Germany 30 (DAX) Spot Index #Germany30', 'displaySymbol': 'Germany30/EUR', 'symbol': 'FXPRO:1106'}, {'description': 'FXPRO China H Shares Spot Index #ChinaHShar', 'displaySymbol': 'China_HShar/HKD', 'symbol': 'FXPRO:1115'}, {'description': 'FXPRO Litecoin vs US Dollar LTCUSD', 'displaySymbol': 'LTC/USD', 'symbol': 'FXPRO:1312'}, {'description': 'FXPRO Euro vs Swiss Franc EURCHF', 'displaySymbol': 'EUR/CHF', 'symbol': 'FXPRO:10'}, {'description': 'FXPRO Australian Dollar vs New Zealand Dollar AUDNZD', 'displaySymbol': 'AUD/NZD', 'symbol': 'FXPRO:1060'}, {'description': 'FXPRO Japan 225 Spot Index #Japan225', 'displaySymbol': 'Japan225/USD', 'symbol': 'FXPRO:1293'}, {'description': 'FXPRO France 120 Spot Index #France120', 'displaySymbol': 'France120/EUR', 'symbol': 'FXPRO:1098'}, {'description': 'FXPRO Ripple vs US Dollar XRPUSD', 'displaySymbol': 'XRP/USD', 'symbol': 'FXPRO:1315'}, {'description': 'FXPRO Euro vs British Pound EURGBP', 'displaySymbol': 'EUR/GBP', 'symbol': 'FXPRO:9'}, {'description': 'FXPRO Australian Dollar vs Polish Zloty AUDPLN', 'displaySymbol': 'AUD/PLN', 'symbol': 'FXPRO:1290'}, {'description': 'FXPRO FTSE 100 Spot Index #UK100', 'displaySymbol': 'UK100/GBP', 'symbol': 'FXPRO:1105'}, {'description': 'FXPRO Germany Technology 30 Spot Index #GerTech30', 'displaySymbol': 'GerTech30/EUR', 'symbol': 'FXPRO:1093'}, {'description': 'FXPRO Euro vs Japanese Yen EURJPY', 'displaySymbol': 'EUR/JPY', 'symbol': 'FXPRO:3'}, {'description': 'FXPRO Australian Dollar vs Singapore Dollar AUDSGD', 'displaySymbol': 'AUD/SGD', 'symbol': 'FXPRO:1085'}, {'description': 'FXPRO US Dow Jones 30 Spot Index #US30', 'displaySymbol': 'US30/USD', 'symbol': 'FXPRO:1287'}, {'description': 'FXPRO Germany 50 Mid Cap Spot Index #Germany50', 'displaySymbol': 'Germany50/EUR', 'symbol': 'FXPRO:1096'}, {'description': 'FXPRO Euro vs US Dollar EURUSD', 'displaySymbol': 'EUR/USD', 'symbol': 'FXPRO:1'}, {'description': 'FXPRO Australian Dollar vs US Dollar AUDUSD', 'displaySymbol': 'AUD/USD', 'symbol': 'FXPRO:5'}, {'description': 'FXPRO US Nasdaq 100 Spot Index #USNDAQ100', 'displaySymbol': 'US_NDAQ100/USD', 'symbol': 'FXPRO:1107'}, {'description': 'FXPRO British Pound vs Canadian Dollar GBPCAD', 'displaySymbol': 'GBP/CAD', 'symbol': 'FXPRO:1040'}, {'description': 'FXPRO Swiss Franc vs Polish Zloty CHFPLN', 'displaySymbol': 'CHF/PLN', 'symbol': 'FXPRO:1086'}, {'description': 'FXPRO US S&P 500 Spot Index #USSPX500', 'displaySymbol': 'USSPX500/USD', 'symbol': 'FXPRO:1288'}, {'description': 'FXPRO Netherlands 25 Spot Index #Holland25', 'displaySymbol': 'Holland25/EUR', 'symbol': 'FXPRO:1094'}, {'description': 'FXPRO British Pound vs Swiss Franc GBPCHF', 'displaySymbol': 'GBP/CHF', 'symbol': 'FXPRO:40'}, {'description': 'FXPRO Swiss Franc vs Singapore Dollar CHFSGD', 'displaySymbol': 'CHF/SGD', 'symbol': 'FXPRO:1291'}, {'description': 'FXPRO Hong Kong 50 Spot Index #HongKong50', 'displaySymbol': 'HongKong50/HKD', 'symbol': 'FXPRO:1114'}, {'description': 'FXPRO British Pound vs Japanese Yen GBPJPY', 'displaySymbol': 'GBP/JPY', 'symbol': 'FXPRO:7'}, {'description': 'FXPRO Euro vs Australian Dollar EURAUD', 'displaySymbol': 'EUR/AUD', 'symbol': 'FXPRO:14'}, {'description': 'FXPRO British Pound vs US Dollar GBPUSD', 'displaySymbol': 'GBP/USD', 'symbol': 'FXPRO:2'}, {'description': 'FXPRO Euro vs Czech Koruna EURCZK', 'displaySymbol': 'EUR/CZK', 'symbol': 'FXPRO:1080'}, {'description': 'FXPRO US Dollar vs Canadian Dollar USDCAD', 'displaySymbol': 'USD/CAD', 'symbol': 'FXPRO:8'}, {'description': 'FXPRO Euro vs Danish Krone EURDKK', 'displaySymbol': 'EUR/DKK', 'symbol': 'FXPRO:1062'}, {'description': 'FXPRO US Dollar vs Swiss Franc USDCHF', 'displaySymbol': 'USD/CHF', 'symbol': 'FXPRO:6'}, {'description': 'FXPRO Euro vs Hong Kong Dollar EURHKD', 'displaySymbol': 'EUR/HKD', 'symbol': 'FXPRO:1051'}, {'description': 'FXPRO Spain 35 Spot Index #Spain35', 'displaySymbol': 'Spain35/EUR', 'symbol': 'FXPRO:1285'}, {'description': 'FXPRO US Dollar vs Japanese Yen USDJPY', 'displaySymbol': 'USD/JPY', 'symbol': 'FXPRO:4'}, {'description': 'FXPRO Euro vs Hungarian Forint EURHUF', 'displaySymbol': 'EUR/HUF', 'symbol': 'FXPRO:1063'}, {'description': 'FXPRO Switzerland 20 Spot Index #Swiss20', 'displaySymbol': 'Swiss20/CHF', 'symbol': 'FXPRO:1286'}, {'description': 'FXPRO Euro vs Mexican Pesos EURMXN', 'displaySymbol': 'EUR/MXN', 'symbol': 'FXPRO:1064'}, {'description': 'FXPRO Euro vs Norwegian Krone EURNOK', 'displaySymbol': 'EUR/NOK', 'symbol': 'FXPRO:1039'}, {'description': 'FXPRO US Small Cap 2000 Spot Index #US2000', 'displaySymbol': 'US20001296/USD', 'symbol': 'FXPRO:1296'}, {'description': 'FXPRO Euro vs New Zealand Dollar EURNZD', 'displaySymbol': 'EUR/NZD', 'symbol': 'FXPRO:1065'}, {'description': 'FXPRO Euro vs Polish Zloty EURPLN', 'displaySymbol': 'EUR/PLN', 'symbol': 'FXPRO:1066'}, {'description': 'FXPRO Euro vs Russian Ruble EURRUB', 'displaySymbol': 'EUR/RUB', 'symbol': 'FXPRO:1083'}, {'description': 'FXPRO Euro vs Swedish Krona EURSEK', 'displaySymbol': 'EUR/SEK', 'symbol': 'FXPRO:1067'}, {'description': 'FXPRO Euro vs Singapore Dollar EURSGD', 'displaySymbol': 'EUR/SGD', 'symbol': 'FXPRO:1087'}, {'description': 'FXPRO Euro vs Turkish Lira EURTRY', 'displaySymbol': 'EUR/TRY', 'symbol': 'FXPRO:1001'}, {'description': 'FXPRO Euro vs South African Rand EURZAR', 'displaySymbol': 'EUR/ZAR', 'symbol': 'FXPRO:1052'}, {'description': 'FXPRO British Pound vs Australian Dollar GBPAUD', 'displaySymbol': 'GBP/AUD', 'symbol': 'FXPRO:16'}, {'description': 'FXPRO British Pound vs Danish Krone GBPDKK', 'displaySymbol': 'GBP/DKK', 'symbol': 'FXPRO:1088'}, {'description': 'FXPRO British Pound vs Norwegian Krone GBPNOK', 'displaySymbol': 'GBP/NOK', 'symbol': 'FXPRO:1041'}, {'description': 'FXPRO British Pound vs New Zealand Dollar  GBPNZD', 'displaySymbol': 'GBP/NZD', 'symbol': 'FXPRO:1042'}, {'description': 'FXPRO British Pound vs Polish Zloty GBPPLN', 'displaySymbol': 'GBP/PLN', 'symbol': 'FXPRO:1089'}, {'description': 'FXPRO British Pound vs Swedish Krona GBPSEK', 'displaySymbol': 'GBP/SEK', 'symbol': 'FXPRO:1090'}, {'description': 'FXPRO British Pound vs Singapore Dollar GBPSGD', 'displaySymbol': 'GBP/SGD', 'symbol': 'FXPRO:1043'}, {'description': 'FXPRO British Pound vs South African Rand GBPZAR', 'displaySymbol': 'GBP/ZAR', 'symbol': 'FXPRO:1068'}, {'description': 'FXPRO Norwegian Krone vs Swedish Krona NOKSEK', 'displaySymbol': 'NOK/SEK', 'symbol': 'FXPRO:1092'}, {'description': 'FXPRO New Zealand Dollar vs Canadian Dollar NZDCAD', 'displaySymbol': 'NZD/CAD', 'symbol': 'FXPRO:1044'}, {'description': 'FXPRO New Zealand Dollar vs Swiss Franc NZDCHF', 'displaySymbol': 'NZD/CHF', 'symbol': 'FXPRO:1045'}, {'description': 'FXPRO New Zealand Dollar vs Japanese Yen NZDJPY', 'displaySymbol': 'NZD/JPY', 'symbol': 'FXPRO:1046'}, {'description': 'FXPRO New Zealand Dollar vs Singapore Dollar NZDSGD', 'displaySymbol': 'NZD/SGD', 'symbol': 'FXPRO:1069'}, {'description': 'FXPRO New Zealand Dollar vs US Dollar NZDUSD', 'displaySymbol': 'NZD/USD', 'symbol': 'FXPRO:12'}, {'description': 'FXPRO Polish Zloty vs Japanese Yen PLNJPY', 'displaySymbol': 'PLN/JPY', 'symbol': 'FXPRO:1091'}, {'description': 'FXPRO Singapore Dollar vs Japanese Yen SGDJPY', 'displaySymbol': 'SGD/JPY', 'symbol': 'FXPRO:1053'}, {'description': 'FXPRO US Dollar vs China Offshore Spot USDCNH', 'displaySymbol': 'USD/CNH', 'symbol': 'FXPRO:1082'}, {'description': 'FXPRO US Dollar vs Czech Koruna USDCZK', 'displaySymbol': 'USD/CZK', 'symbol': 'FXPRO:1081'}, {'description': 'FXPRO US Dollar vs Danish Krone USDDKK', 'displaySymbol': 'USD/DKK', 'symbol': 'FXPRO:1070'}, {'description': 'FXPRO US Dollar vs Hong Kong Dollar USDHKD', 'displaySymbol': 'USD/HKD', 'symbol': 'FXPRO:1054'}, {'description': 'FXPRO US Dollar vs Hungarian Forint USDHUF', 'displaySymbol': 'USD/HUF', 'symbol': 'FXPRO:1071'}, {'description': 'FXPRO US Dollar vs Israeli Shekel USDILS', 'displaySymbol': 'USD/ILS', 'symbol': 'FXPRO:1292'}, {'description': 'FXPRO US Dollar vs Mexican Pesos USDMXN', 'displaySymbol': 'USD/MXN', 'symbol': 'FXPRO:1047'}, {'description': 'FXPRO US Dollar vs Norwegian Krone USDNOK', 'displaySymbol': 'USD/NOK', 'symbol': 'FXPRO:1048'}, {'description': 'FXPRO US Dollar vs Polish Zloty USDPLN', 'displaySymbol': 'USD/PLN', 'symbol': 'FXPRO:1049'}, {'description': 'FXPRO US Dollar vs Russian Ruble USDRUB', 'displaySymbol': 'USD/RUB', 'symbol': 'FXPRO:1084'}, {'description': 'FXPRO US Dollar vs Swedish Krona USDSEK', 'displaySymbol': 'USD/SEK', 'symbol': 'FXPRO:1050'}, {'description': 'FXPRO US Dollar vs Singapore Dollar USDSGD', 'displaySymbol': 'USD/SGD', 'symbol': 'FXPRO:28'}, {'description': 'FXPRO US Dollar vs Thai Baht USDTHB', 'displaySymbol': 'USD/THB', 'symbol': 'FXPRO:1313'}, {'description': 'FXPRO US Dollar vs Turkish Lira USDTRY', 'displaySymbol': 'USD/TRY', 'symbol': 'FXPRO:1000'}, {'description': 'FXPRO US Dollar vs South African Rand USDZAR', 'displaySymbol': 'USD/ZAR', 'symbol': 'FXPRO:1055'}, {'description': 'FXPRO Natural Gas (US) Spot NAT.GAS', 'displaySymbol': 'XNG/USD', 'symbol': 'FXPRO:1118'}, {'description': 'FXPRO WTI Spot Oil WTI', 'displaySymbol': 'XTI/USD', 'symbol': 'FXPRO:1116'}, {'description': 'FXPRO FTSE Mid 250 Spot Index #UKmid250', 'displaySymbol': 'UKmid250/GBP', 'symbol': 'FXPRO:1316'}]

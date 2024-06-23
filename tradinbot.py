from lumibot.brokers import Alpaca
from lumibot.backtesting import YahooDataBacktesting
from lumibot.strategies.strategy import Strategy
from lumibot.traders import Trader
from datetime import datetime
from alpaca_trade_api import REST
from timedelta import Timedelta
from finbert_utils import estimate_sentiment
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv('API_KEY')
API_SECRET = os.getenv('API_SECRET')
BASE_URL = os.getenv('BASE_URL')


ALPACA_CREDS = {
    'API_KEY': API_KEY,
    'API_SECRET': API_SECRET,
    'PAPER': True,
}


class MLTrader(Strategy):
    def initialize(self, symbol:str='SPY', sleeptime:str='1h', cash_at_risk:float=0.5):
        self.symbol = symbol
        self.sleeptime =  sleeptime
        self.last_trade = None
        self.cash_at_risk = cash_at_risk
        self.api = REST(base_url = BASE_URL, key_id = API_KEY, secret_key = API_SECRET)
        
    def position_sizing(self):
        cash = self.get_cash()
        last_price = self.get_last_price(self.symbol)
        quantity = round(cash * self.cash_at_risk / last_price) #Use 50% of cash
        return cash , last_price, quantity
    
    def get_dates(self, no_days_prior:int=3):
        today = self.get_datetime()
        days_prior = today - Timedelta(days = no_days_prior)
        return today.strftime('%Y-%m-%d'), days_prior.strftime('%Y-%m-%d')
    
    def get_sentiment(self):
        today, days_prior = self.get_dates(no_days_prior=3)
        news = self.api.get_news(symbol = self.symbol, start=days_prior, end=today)
        
        news = [ev.__dict__['_raw']['headline'] for ev in news]
        
        probability, sentiment = estimate_sentiment(news)
        
        return probability, sentiment
    
    def on_trading_iteration(self):
        cash, last_price, quantity = self.position_sizing()
        probability, sentiment = self.get_sentiment()
        
        if cash > last_price:
            if sentiment == 'positive' and probability > .99:
                if self.last_trade == 'sell':
                    self.sell_all()
                order = self.create_order(
                                        self.symbol, 
                                        quantity, 
                                        'buy', 
                                        type='bracket',
                                        take_profit_price=last_price*1.20,
                                        stop_loss_price=last_price*0.90,
                                        )
                self.submit_order(order)
                self.last_trade = 'buy'
                
            elif sentiment == 'negative' and probability > .99:
                if self.last_trade == 'buy':
                    self.sell_all()
                order = self.create_order(
                                        self.symbol, 
                                        quantity, 
                                        'sell', 
                                        type='bracket',
                                        take_profit_price=last_price*0.80,
                                        stop_loss_price=last_price*1.15,
                                        )
                self.submit_order(order)
                self.last_trade = 'sell'
    
    
broker = Alpaca(ALPACA_CREDS)
start_date = datetime(2020, 1, 1)
end_date = datetime(2023, 12, 31)

strategy = MLTrader(name='mlstrat', broker=broker, 
                    parameters={"symbol": "SPY", "sleeptime": "24h", "cash_at_risk": 0.5})
strategy.backtest(
    YahooDataBacktesting,
    start_date,
    end_date,
    parameters={},
)

#to deploy the strategy
# trader = Trader()
# trader.add_strategy(strategy)
# trader.run_all()
# and comment the backtest code (91-96)
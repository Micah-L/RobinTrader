#Use with Python3
#Installation:
#pip install robin_stocks pandas


###   Constants   ###
###~~~~~~~~~~~~~~~###

## Account related ## 
# Replace with your Robinhood login info
USERNAME = "YourUsernameHere"
PASSWORD = "YourPasswordHere" 

# Set True to do 2fa by SMS, or False to do 2fa by email.
TWO_FACTOR_IS_SMS = True

## Strategy related ##

# Create a list of symbols to trade
# To only trade one symbol:
# SYMBOLS = ['ETH']
# To trade multiple symbols:
# SYMBOLS = ['ETH', 'ETC', 'LTC']
SYMBOLS = ['BTC', 'ETH', 'ETC', 'LTC', 'DOGE']
 
# program will cancel all crypto orders older than this many seconds
DEFAULT_STALE_ORDER_AGE = 90
 
TAKE_PROFIT_PERCENT = 2.5
# TAKE_PROFIT_PERCENT = None
STOP_LOSS_PERCENT = 1
# STOP_LOSS_PERCENT = None

# Period for RSI calculation (number of data frames)
RSI_PERIOD = 21
# Size of data frame for calculating RSI
# Can be '15second', '5minute', '10minute', 'hour', 'day', or 'week'
# '15second' may result in an error if RSI_SPAN is longer than 'hour'
# RSI_WINDOW = '5minute'
RSI_WINDOW = '15second'
# The entire time frame to collect data points. Can be 'hour', 'day', 'week', 'month', '3month', 'year', or '5year'
# If the span is too small to fit RSI_PERIOD number of RSI_WINDOWs then there will be an error
# RSI_SPAN = "day"
RSI_SPAN = 'hour'

# Set RSI levels to buy/sell at
DEFAULT_RSI_BUY_AT = 30
DEFAULT_RSI_SELL_AT = 70

# If this is non-zero, the rsi buy or sell levels will be adjusted down or up after each buy or sell.
# Set higher to take advantage of longer price movements in one direction
RSI_STEP_BUY_WHEN_TRIGGERED = 8
RSI_STEP_SELL_WHEN_TRIGGERED = 16
# The rate (in RSI/second) to adjust the RSI cutoffs back towards the default levels.
# Set lower to take advantage of longer price movements in one direction
RSI_RESET_BUY_SPEED = 0.02
RSI_RESET_SELL_SPEED = 0.05


# per individual trade
MIN_DOLLARS_PER_TRADE = 2.00
MAX_DOLLARS_PER_TRADE = 5.50

## Misc settings ##

# Time to wait between loops (seconds)
MAIN_LOOP_SLEEP_TIME = 6.15

# If true, print extra information to console
DEBUG_INFO = False

# END CONSTANTS #
###~#~#~#~#~#~###


# Imports for builtin modules:
from collections import defaultdict
import datetime
from dateutil.parser import parse as datetime_parser
import time
import logging
FORMAT = '%(asctime)-15s%(message)s'
logging.basicConfig(filename='trades.log', encoding='utf-8', level=logging.INFO, format = FORMAT)

from io import StringIO 
import sys

class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self
    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio    # free up some memory
        sys.stdout = self._stdout
# Usage:
# with Capturing() as output:
    # do_something(my_object)

# Imports for installed modules:
import robin_stocks as rh
import pandas as pd
# from pprint import pprint # (mostly for debugging)
# import pickle # (not yet used)

# You may place the above constants in a file called "config.py" to override the defaults.
try:
    from config import *
except ModuleNotFoundError:
    pass


# Useful functions
def RSI(prices, current_only=False, period = RSI_PERIOD):
    """ Calculate RSI and return the values as a pandas DataFrame.
    
    prices -- should be a pandas DataFrame containing price info
    current_only -- set this to True to just return the most recent value of the RSI
    """
    
    # Get the price changes
    delta = prices.diff()
    
    # Get rid of the first entry, which is NaN
    delta = delta[1:] 
    
    up, down = delta.copy(), delta.copy()
    # List of only upward moves (replace downward moves with 0)
    up[up < 0] = 0
    # List of only downward moves (replace upward moves with 0)
    down[down > 0] = 0
    
    # Calculate EMA of upward and downward moves
    roll_up1 = up.ewm(span=period).mean()
    roll_down1 = down.abs().ewm(span=period).mean()
    
    # Relative Strength
    try:
        rs = roll_up1 / roll_down1
    except ZeroDivisionError:
        # RSI is 100 if all up moves...
        rs = float("Inf")
    # Finally...
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    if current_only:
        # Just return most recent RSI value
        # print(rsi)
        return rsi.iloc[-1]
    # Returns a list of RSI values
    return rsi
def float_to_ndigits(f):
    """ f must be a multiple of 10 """
    ndigits = 0
    while f < 1:
        ndigits += 1
        f *= 10
    return ndigits

class RobinTrader:
    """ Class to handle trading instance """
    def __init__(self, username, password, starting_rsi_buy_at = DEFAULT_RSI_BUY_AT, starting_rsi_sell_at = DEFAULT_RSI_SELL_AT):
        self.username = username
        self.password = password
        
        self.starting_total_value = None
        self.total_trades = 0
        self.up_since = time.time()
        
        self.increments = defaultdict(int) # {symbol : increment (int)}
        
        self.order_ids = dict() # {time : id} pairs
        
        self.starting_rsi_buy_at = starting_rsi_buy_at
        self.starting_rsi_sell_at = starting_rsi_sell_at
        
        self.rsi_buy_at = dict()
        self.rsi_sell_at = dict()
        
        # stores the last time the rsi_based_buy_sell function was run with [symbol]
        self.last_time = dict()
        
        rh.login(username=username, password=password, expiresIn=86400, by_sms=TWO_FACTOR_IS_SMS)
        
        self.load_active_crypto_orders()
        
    def check_cash_on_hand(self, symbol = "USD"):
        cash_on_hand = 0
        if symbol == "USD":
            info = rh.load_phoenix_account()
            cash_on_hand = float(info['uninvested_cash']['amount'])
            
            # TODO:
            # If we want to separate this bot from the rest of the acct, then we will need to 
            # do other calculations here based on orders placed and total amount willing to invest.
            # If we're fine using the entire account balancefor this bot, then we only need 
            # to return the uninvested_cash.
        else:
            crypto_on_hand = dict()
            crypto_positions = rh.get_crypto_positions()
            if symbol not in self.increments.keys():
                self.increments[symbol] = float_to_ndigits(float( list(filter(lambda x: x['currency']['code'] == symbol, crypto_positions))[0]['currency']['increment'] ))
            try:
                crypto_on_hand['cost'] = float( list(filter(lambda x: x['currency']['code'] == symbol, crypto_positions))[0]['cost_bases'][0]['direct_cost_basis'] )
            except IndexError:
                crypto_on_hand['cost'] = 0
            try:
                crypto_on_hand['quantity'] = float( list(filter(lambda x: x['currency']['code'] == symbol, crypto_positions))[0]['quantity'] )
            except IndexError:
                crypto_on_hand['quantity'] = 0
            crypto_on_hand['quote'] = float(rh.get_crypto_quote(symbol)['bid_price'])
            crypto_on_hand['value'] = crypto_on_hand['quote']*crypto_on_hand['quantity']
            
            cash_on_hand = crypto_on_hand
            
        return cash_on_hand
    def cancel_old_crypto_orders(self, age = DEFAULT_STALE_ORDER_AGE):
        to_remove = set()
        # if DEBUG_INFO: pprint(self.order_ids)
        for t in self.order_ids.keys():
            if time.time() - t > DEFAULT_STALE_ORDER_AGE:
                info = rh.cancel_crypto_order(self.order_ids[t])
                # if DEBUG_INFO: pprint(info)
                to_remove.add(t)
        for t in to_remove:
            del self.order_ids[t]      
    def load_active_crypto_orders(self):
        info = rh.get_all_open_crypto_orders() 
        for i in info:
            self.order_ids[datetime_parser(i['created_at']).timestamp()] = i['id']
        print(self.order_ids)
    def mainloop(self, sleep_time = MAIN_LOOP_SLEEP_TIME, loops = float("Inf")):
        """  """
        
        i = 0
        last_time=time.time()
        while i < loops:
            # if DEBUG_INFO: print(f"Loop {i+1}")
            try:
                outputs = []
                for symbol in SYMBOLS:
                    with Capturing() as output:
                        self.rsi_based_buy_sell(symbol = symbol)
                    outputs.append(output)
                
                print(f"{time.ctime()}\tUptime: {datetime.timedelta(seconds = int(time.time()-self.up_since))}\t\tTrades: {self.total_trades}")
                for o in outputs[0][0:2]:
                    print(o)
                total_value = float(outputs[0][1].split('\t')[3][1:])
                for out in outputs:
                    print(out[2])
                    total_value += float(out[2].split('\t')[3][1:])
                if self.starting_total_value is None:
                    self.starting_total_value = total_value
                diff = total_value - self.starting_total_value
                sign = "+" if diff >= 0 else "-"
                diff = abs(diff)
                print(f"Total Value:\t${total_value:.2f}\t\tChange:\t{sign}${diff:.2f}")

                for out in outputs:
                    for o in out[3:]:
                        print(o)
                for _ in range(max(0,3-len(out[3:]))):
                    print("")
                       
                self.cancel_old_crypto_orders()
            
            except (TypeError, KeyError, TimeoutError):
                # Probably 504 server error, and robin_stocks tried subscript NoneType object 
                # or KeyError
                print(f"Server busy. Waiting {MAIN_LOOP_SLEEP_TIME}s to retry.")
                time.sleep(MAIN_LOOP_SLEEP_TIME)
            

            print("")
            time.sleep(sleep_time)
            i += 1
    def rsi_based_buy_sell(self, symbol):
        """ Check the RSI and possibly place a buy or sell order """
        
        if symbol not in self.rsi_buy_at.keys():
            self.rsi_buy_at[symbol] = self.starting_rsi_buy_at
        if symbol not in self.rsi_sell_at.keys():
            self.rsi_sell_at[symbol] = self.starting_rsi_sell_at
        if symbol not in self.last_time.keys():
            self.last_time[symbol] = time.time()
        
        ## Check the RSI ##
        historical_data = rh.get_crypto_historicals(symbol=symbol, interval=RSI_WINDOW, span=RSI_SPAN, bounds="24_7", info=None)
        df = pd.DataFrame(historical_data)
        # convert prices to float since values are given as strings
        df["close_price"] = pd.to_numeric(df["close_price"], errors='coerce') 
        # Get the current RSI
        rsi = RSI(df["close_price"], current_only=True)
        
        cash = self.check_cash_on_hand()
        crypto_on_hand = self.check_cash_on_hand(symbol = symbol)
        quote = crypto_on_hand['quote']
        quantity = crypto_on_hand['quantity']
        cost_basis = crypto_on_hand['cost']
        try:
            pnl_percent = 100*(quote*quantity - cost_basis)/cost_basis
        except ZeroDivisionError:
            pnl_percent = 0
        sign = "+" if pnl_percent >= 0 else ""
        print("Sym\tQuote\tQty\tVal\tPnL\tCost\tRSI\tBuy@\tSell@")
        print(f"USD\t1\t{cash:.2f}\t${cash:.2f}\t\t\t")
        if symbol == 'BTC':
            print(f"{symbol}\t{quote:.0f}\t{quantity:.5f}\t${quote*quantity:.2f}\t{sign}{pnl_percent:.2f}%\t${cost_basis:.2f}\t{rsi:.2f}\t{self.rsi_buy_at[symbol]:.2f}\t{self.rsi_sell_at[symbol]:.2f}")
        elif symbol == 'DOGE':
            print(f"{symbol}\t{quote:.5f}\t{quantity:.0f}\t${quote*quantity:.2f}\t{sign}{pnl_percent:.2f}%\t${cost_basis:.2f}\t{rsi:.2f}\t{self.rsi_buy_at[symbol]:.2f}\t{self.rsi_sell_at[symbol]:.2f}")
        else:
            print(f"{symbol}\t{quote:.2f}\t{quantity:.5f}\t${quote*quantity:.2f}\t{sign}{pnl_percent:.2f}%\t${cost_basis:.2f}\t{rsi:.2f}\t{self.rsi_buy_at[symbol]:.2f}\t{self.rsi_sell_at[symbol]:.2f}")
        
        ## Adjust RSI buy/sell levels towards the defaults.
        self.adjust_rsi(symbol)
        ## Check for stop loss / take profits:
        if TAKE_PROFIT_PERCENT is not None and pnl_percent > TAKE_PROFIT_PERCENT:
            # info = rh.order_sell_crypto_limit(symbol, quantity, round(0.999*quote,2))
            info = self.trigger_tx(symbol, quantity, round(0.99*quote, 2), side="sell", quantity_on_hand = quantity)
            if info is not None: 
                print(f"Take profit triggered!  Selling {quantity} of {symbol}")
                self.total_trades += 1
            # if DEBUG_INFO: pprint(info)
            return # skip checking rsi this time around
        
        elif STOP_LOSS_PERCENT is not None and pnl_percent < -1*STOP_LOSS_PERCENT:
            info = self.trigger_tx(symbol, quantity, round(0.99*quote, 2), side="sell", quantity_on_hand = quantity)
            # info = rh.order_sell_crypto_limit(symbol, quantity, round(0.99*quote, 2))
            # rh.order_sell_crypto_by_quantity(symbol, round(quantity,6))
            if info is not None:
                print(f"Stop loss triggered! Selling {quantity} of {symbol}")
                self.total_trades += 1
                # Step RSI buy cutoff down so we don't buy again right away
                self.bump_rsi(symbol, 'buy', 5) # 5x normal adjustment
            # pprint(info)
            return # skip checking rsi
        
        # Check RSI to see if we should buy or sell
        if rsi <= self.rsi_buy_at[symbol]:
            info = self.trigger_tx(symbol, quantity = None, price = round(1.01*quote,2), side = 'buy', cash_on_hand = cash)
            if info is not None:
                try:
                    if not isinstance(info['quantity'], list):
                        print(f"Buying: {symbol}") # {info['quantity']:.6f} at {info['price']:.2f} ({info['quantity']*info['price']:.2f})")
                        self.bump_rsi(symbol, 'buy')
                        self.total_trades += 1
                except (ValueError, KeyError):
                    pass #logging.warn(f"Failed buying: {info}")
                    
        elif rsi >= self.rsi_sell_at[symbol]:
            info = self.trigger_tx(symbol, quantity = None, price = None, side = 'sell', quantity_on_hand = quantity)
            if info is not None:
                try:
                    if not isinstance(info['quantity'], list):
                        print(f"Selling: {symbol}") # {info['quantity']:0.6f} at {info['price']:.2f} ({info['quantity']*info['price']:.2f})")
                        self.bump_rsi(symbol, 'sell')
                        self.total_trades += 1                    
                except (ValueError, KeyError):
                    pass # logging.warn(f"Failed selling: {info}")
                
    def trigger_tx(self, symbol, quantity = None, price = None, side = None, cash_on_hand = None, quantity_on_hand = None):
        """ Attempts to make a trade. Returns None if no trade was made. """
        info = None
        if side not in {"buy", "sell"}:
            raise Exception("side should be 'buy' or 'sell'")
        if side == 'buy':
            cash_on_hand = self.check_cash_on_hand()
            if cash_on_hand < MIN_DOLLARS_PER_TRADE: return

            if price is None:
                raise Exception("Price cannot be None. Calcuate a price or change the code to calculate a default price.")
            elif quantity is None: 
                # price is not None and quantity is None
                # so calculate a quantity:
                if cash_on_hand is None:
                    cash_on_hand = self.check_cash_on_hand(symbol="USD")
                buy_amount = min(cash_on_hand, MAX_DOLLARS_PER_TRADE)
                if cash_on_hand - buy_amount < MIN_DOLLARS_PER_TRADE:
                    buy_amount = cash_on_hand
                info = rh.order_buy_crypto_by_price(symbol, buy_amount)
            else:
                info = rh.order_buy_crypto_limit(symbol, quantity, price)
            
        else: # side == 'sell'
            if price is None:
                price = float(rh.get_crypto_quote(symbol)['bid_price'])
            if quantity_on_hand is None:
                raise Exception("quantity_on_hand cannot be None. Calcuate a quantity or change the code to calculate a default price.")
            if quantity_on_hand*price < MIN_DOLLARS_PER_TRADE:
                return
            if quantity is None:
                
                quantity = round(MAX_DOLLARS_PER_TRADE/price, self.increments[symbol])
                if price*quantity_on_hand < MAX_DOLLARS_PER_TRADE or price*(quantity_on_hand - quantity) < MIN_DOLLARS_PER_TRADE:
                    quantity = quantity_on_hand
            else:
                pass
            info = rh.order_sell_crypto_by_quantity(symbol, quantity)
            
        if info is not None:
            with Capturing() as output:
                print(f"Trade executed: symbol = {symbol}, quantity = {quantity}, price = {price}, side = {side}\n")
                print(info)
            try:
                self.order_ids[time.time()] = info['id']                
                # self.bump_rsi(symbol,side)
            except KeyError:
                # Trade didn't complete, don't change rsi cutoffs
                pass
            logging.info(output)
        return info
    def adjust_rsi(self, symbol):
        # increase the buy level, but no more than the default
        self.rsi_buy_at[symbol] = min(DEFAULT_RSI_BUY_AT, self.rsi_buy_at[symbol] + (time.time() - self.last_time[symbol])*RSI_RESET_BUY_SPEED)
        # decrease the sell level, but no less than the default
        self.rsi_sell_at[symbol] = max(DEFAULT_RSI_SELL_AT, self.rsi_sell_at[symbol] - (time.time() - self.last_time[symbol])*RSI_RESET_SELL_SPEED)
        # Store the last time the rsi was adjusted
        self.last_time[symbol] = time.time()
    def bump_rsi(self, symbol, side, multiple=1):
        """ """
        assert side in {'buy','sell'}
        if side == 'buy':
            self.rsi_buy_at[symbol] = max(0, self.rsi_buy_at[symbol] - multiple*RSI_STEP_BUY_WHEN_TRIGGERED)
        elif side == 'sell':
            self.rsi_sell_at[symbol] = min(100, self.rsi_sell_at[symbol] + multiple*RSI_STEP_SELL_WHEN_TRIGGERED)

    
def main():
    
    client = RobinTrader(username = USERNAME, password = PASSWORD)
    client.mainloop(sleep_time = MAIN_LOOP_SLEEP_TIME)
    
if __name__ == "__main__":
    main()
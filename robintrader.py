#Use with Python3
#Installation:
#pip install robin_stocks pandas
# Imports for builtin modules:
from collections import defaultdict
import datetime
from dateutil.parser import parse as datetime_parser
import time
import logging
logging.basicConfig(filename='trades.log', encoding='utf-8', level=logging.INFO, format = '%(asctime)-15s%(message)s')
from io import StringIO 
import sys


# Imports for installed modules:
import robin_stocks as rh
import pandas as pd


###  SETTINGS  ###
#  ~~~~~~~~~~~~  #

# Modify these with your values

## Account related
USERNAME = "username"
PASSWORD = "password"

# Set True to do 2fa by SMS, or False to do 2fa by email.
TWO_FACTOR_IS_SMS = True

## Strategy related ##

# Create a list of symbols to trade
# To only trade one symbol:
# SYMBOLS = ['ETH']
# To trade multiple symbols:
# SYMBOLS = ['ETH', 'ETC', 'LTC']
SYMBOLS = ['BTC', 'ETH', 'LTC']
# SYMBOLS = ['BTC', 'ETH', 'ETC', 'LTC', 'DOGE']
# SYMBOLS = ['BTC']

# If true, allocate a maximum % of portfolio to each symbol
# if false, allocate a maximum dollar value of portfolio to each
MAX_ALLOCATION_IS_PERCENT = True
# The maximum allocation of funds towards any one particular symbol
# values default to DEFAULT_MAX_ALLOCATION when values are undefined
DEFAULT_MAX_ALLOCATION = 50
# Set the maximum for each symbol independently. Numbers need not add to 100%
MAX_ALLOCATION = defaultdict(lambda: DEFAULT_MAX_ALLOCATION)
# MAX_ALLOCATION['BTC'] = 60
# MAX_ALLOCATION['BTC'] = None # Equivalent to 100% / no limit
MAX_ALLOCATION['BTC'] = 61.8
MAX_ALLOCATION['ETH'] = 61.8
MAX_ALLOCATION['ETC'] = 30
MAX_ALLOCATION['LTC'] = 15
MAX_ALLOCATION['DOGE'] = 15

# Program will cancel all crypto orders older than this many seconds
DEFAULT_STALE_ORDER_AGE = 90
 
## Default take profits and stop loss variables can be numbers or None:
DEFAULT_TAKE_PROFIT_PERCENT = 5
# DEFAULT_TAKE_PROFIT_PERCENT = None
DEFAULT_STOP_LOSS_PERCENT = 1
# DEFAULT_STOP_LOSS_PERCENT = None
# Also create dictionaries for certain take profit percentages:
# Symbols not listed will take DEFAULT_TAKE_PROFIT_PERCENT
TAKE_PROFIT_PERCENT = defaultdict(lambda: DEFAULT_TAKE_PROFIT_PERCENT)
STOP_LOSS_PERCENT = defaultdict(lambda: DEFAULT_STOP_LOSS_PERCENT)
# Modify these or add more
# TAKE_PROFIT_PERCENT['BTC'] = None # can specificy not to take profits
TAKE_PROFIT_PERCENT['BTC'] = 10
TAKE_PROFIT_PERCENT['ETH'] = 10
TAKE_PROFIT_PERCENT['LTC'] = 2
STOP_LOSS_PERCENT['BTC'] = 5
STOP_LOSS_PERCENT['ETH'] = 5
STOP_LOSS_PERCENT['LTC'] = 1

# Period for RSI calculation (number of data frames)
DEFAULT_RSI_PERIOD = 9
RSI_PERIOD = defaultdict(lambda: DEFAULT_RSI_PERIOD)
# RSI_PERIOD['BTC'] = 14
RSI_PERIOD['LTC'] = 7

# Size of data frame for calculating RSI
# Can be '15second', '5minute', '10minute', 'hour', 'day', or 'week'
# '15second' may result in an error if RSI_SPAN is longer than 'hour'
DEFAULT_RSI_WINDOW = 'hour'
RSI_WINDOW = defaultdict(lambda: DEFAULT_RSI_WINDOW)
# RSI_WINDOW['BTC'] = 'hour'
RSI_WINDOW['LTC'] = '5minute'

# The entire time frame to collect data points. Can be 'hour', 'day', 'week', 'month', '3month', 'year', or '5year'
# If the span is too small to fit RSI_PERIOD number of RSI_WINDOWs then there will be an error
DEFAULT_RSI_SPAN = 'day'
RSI_SPAN = defaultdict(lambda: DEFAULT_RSI_SPAN)
# RSI_SPAN['BTC'] = 'day'
RSI_SPAN['LTC'] = 'hour'

# Set RSI levels to buy/sell at
DEFAULT_RSI_BUY_AT = 20
DEFAULT_RSI_SELL_AT = 80
RSI_BUY_AT = defaultdict(lambda: DEFAULT_RSI_BUY_AT)
RSI_SELL_AT = defaultdict(lambda: DEFAULT_RSI_SELL_AT)
RSI_BUY_AT['BTC'] = 30
RSI_SELL_AT['BTC'] = 80
RSI_BUY_AT['ETH'] = 30
RSI_SELL_AT['ETH'] = 80
RSI_BUY_AT['LTC'] = 10
RSI_SELL_AT['LTC'] = 70

# If step is non-zero, the rsi buy or sell levels will be adjusted down or up after each buy or sell.
# Set higher to take advantage of slower price movements in one direction
DEFAULT_RSI_STEP_BUY_WHEN_TRIGGERED = 1
DEFAULT_RSI_STEP_SELL_WHEN_TRIGGERED = 10
RSI_STEP_BUY_WHEN_TRIGGERED = defaultdict(lambda: DEFAULT_RSI_STEP_BUY_WHEN_TRIGGERED)
RSI_STEP_SELL_WHEN_TRIGGERED = defaultdict(lambda: DEFAULT_RSI_STEP_SELL_WHEN_TRIGGERED)
RSI_STEP_BUY_WHEN_TRIGGERED['BTC'] = 0.5
RSI_STEP_SELL_WHEN_TRIGGERED['BTC'] = 10
RSI_STEP_BUY_WHEN_TRIGGERED['ETH'] = 1
RSI_STEP_SELL_WHEN_TRIGGERED['ETH'] = 10
RSI_STEP_BUY_WHEN_TRIGGERED['LTC'] = 1
RSI_STEP_SELL_WHEN_TRIGGERED['LTC'] = 1

# The rate (in RSI/second) to adjust the RSI cutoffs back towards the default levels.
# Set lower to take advantage of longer price movements in one direction
DEFAULT_RSI_RESET_BUY_SPEED = 0.01
DEFAULT_RSI_RESET_SELL_SPEED = 0.01
RSI_RESET_BUY_SPEED = defaultdict(lambda: DEFAULT_RSI_RESET_BUY_SPEED)
RSI_RESET_SELL_SPEED = defaultdict(lambda: DEFAULT_RSI_RESET_SELL_SPEED)
RSI_RESET_BUY_SPEED['BTC'] = 0.01
RSI_RESET_SELL_SPEED['BTC'] = 0.01
RSI_RESET_BUY_SPEED['ETH'] = 0.01
RSI_RESET_SELL_SPEED['ETH'] = 0.01



# per individual trade
MIN_DOLLARS_PER_TRADE = 2.00 # no need to change 
DEFAULT_MAX_DOLLARS_PER_BUY = 5.50
DEFAULT_MAX_DOLLARS_PER_SELL = float('Inf')
MAX_DOLLARS_PER_BUY = defaultdict(lambda: DEFAULT_MAX_DOLLARS_PER_BUY)
MAX_DOLLARS_PER_SELL = defaultdict(lambda: DEFAULT_MAX_DOLLARS_PER_SELL)
MAX_DOLLARS_PER_BUY['BTC'] = 10
MAX_DOLLARS_PER_BUY['ETH'] = 7.50
MAX_DOLLARS_PER_BUY['LTC'] = float('Inf')
# MAX_DOLLARS_PER_SELL['BTC'] = 50
MAX_DOLLARS_PER_SELL['ETH'] = 50

## Misc settings ##

# Time to wait between loops (seconds)
MAIN_LOOP_SLEEP_TIME = 6.15

# If true, print extra information to console
DEBUG_INFO = False

# END SETTINGS #
###~#~#~#~#~#~###

# You may place the above constants in a file called "config.py" to override the defaults.
try:
    from config import *
except ModuleNotFoundError:
    pass


# Useful functions
def RSI(prices, period, current_only=False):
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
    """ f must be a power of 10 """
    ndigits = 0
    while f < 1:
        ndigits += 1
        f *= 10
    return ndigits
class Capturing(list):
    """Capture stdout and save it as a variable. Usage:
    with Capturing() as output:
        do_something(my_object)"""
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self
    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio    # free up some memory
        sys.stdout = self._stdout

class smartdict(dict):
    """ A dict d initialized with a dict D, whose default value for d[k] is D[k] """
    def __init__(self, defaults):
        self.defaults = defaults
    def __missing__(self, key):
        return self.defaults[key]


# r = smartdict(RSI_BUY_AT)
# print(r['BTC'])
# r['BTC'] = r['BTC'] + 1
# print(r['BTC'])
# quit()
class RobinTrader:
    """ Class to handle trading instance """
    def __init__(self, username, password):
        
        ## Data structure set up:
    
        # Account specific
        self.username = username
        self.password = password
        
        # Totals and instance specific
        self.up_since = time.time()
        self.starting_total_value = None
        self.total_value = 0
        self.total_trades = 0
        self.total_stops_losses = 0
        self.total_take_profits = 0

        # Symbol specific
        self.quote = defaultdict(float) # Current price
        self.quantity_on_hand = defaultdict(float) 
        self.cost_basis = defaultdict(float)
        self.symbol_value = defaultdict(float) # Value in usd
        self.symbol_pnl_percent = defaultdict(float)

        self.symbol_trades = defaultdict(int) 
        self.symbol_take_profits = defaultdict(int) # How many take profits triggered
        self.symbol_stop_losses = defaultdict(int) # How many stops triggered
        self.rsi = defaultdict(float)
        self.time_rsi_cutoff_last_adjusted = defaultdict(lambda: self.up_since)

        self.take_profit_percent = TAKE_PROFIT_PERCENT
        self.stop_loss_percent = STOP_LOSS_PERCENT
        
        # Can be a number or dict
        self.starting_rsi_buy_at = RSI_BUY_AT
        self.starting_rsi_sell_at = RSI_SELL_AT
        # for current levels
        self.rsi_buy_at = smartdict(self.starting_rsi_buy_at)
        self.rsi_sell_at = smartdict(self.starting_rsi_sell_at)
        self.rsi_step_buy_when_triggered = RSI_STEP_BUY_WHEN_TRIGGERED
        self.rsi_step_sell_when_triggered = RSI_STEP_SELL_WHEN_TRIGGERED
        self.rsi_reset_buy_speed = RSI_RESET_BUY_SPEED
        self.rsi_reset_sell_speed = RSI_RESET_SELL_SPEED
        
        # How many digit of precision allowed by robinhood
        self.ndigits = defaultdict(int) # {symbol : increment (int)}
        self.ndigits['ETC'] = 6
        self.ndigits['ETH'] = 6
        self.ndigits['BTC'] = 8
        self.ndigits['DOGE'] = 0
        self.ndigits['LTC'] = 8

        # holds active orders placed
        self.order_ids = dict() # {time : id} pairs
        
        
        
        ## Actions performed on start up:

        # Login
        rh.login(username=username, password=password, expiresIn=86400, by_sms=TWO_FACTOR_IS_SMS)
        rh.logout()
        rh.login(username=username, password=password, expiresIn=86400, by_sms=TWO_FACTOR_IS_SMS)
        # Load orders that may have been placed before this program started
        self.load_active_crypto_orders()
        self.quantity_on_hand['USD'] = self.check_cash_on_hand()
        
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
            if symbol not in self.ndigits.keys():
                self.ndigits[symbol] = float_to_ndigits(float( list(filter(lambda x: x['currency']['code'] == symbol, crypto_positions))[0]['currency']['increment'] ))
                self.ndigits[symbol] = min(8,self.ndigits[symbol])
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
        """Main loop which calls other functions/strategies. Also handles the display."""
        
        i = 0
        time_rsi_cutoff_last_adjusted=time.time()
        while i < loops:
            # if DEBUG_INFO: print(f"Loop {i+1}")
            try:
                # Work around since output is already coming from self.rsi_based_buy_sell
                # TODO: Make output less hackish by storing relevant data in instance variables and delgate output to specialized function
                outputs = []
                for symbol in SYMBOLS:
                    with Capturing() as output:
                        self.rsi_based_buy_sell(symbol = symbol)
                    outputs.append(output)
                
                print(f"\
{time.ctime()} \
| Uptime: {datetime.timedelta(seconds = int(time.time()-self.up_since))} \
| Total Trades: {self.total_trades} ({self.total_stops_losses} SL / {self.total_take_profits} TP)")
                
                for o in outputs[0][0:2]:
                    print(o)
                    
                
                self.total_value = float(outputs[0][1].split('\t')[3][1:])
                for out in outputs:
                    print(out[2])
                    self.total_value += float(out[2].split('\t')[3][1:])
                if self.starting_total_value is None:
                    self.starting_total_value = self.total_value
                diff = self.total_value - self.starting_total_value
                sign = "+" if diff >= 0 else "-"
                diff = abs(diff)
                print(f"Total Value:\t${self.total_value:.2f}\t\tChange:\t{sign}${diff:.2f}")
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
        
        # if symbol not in self.rsi_buy_at.keys():
        #     self.rsi_buy_at[symbol] = self.starting_rsi_buy_at
        # if symbol not in self.rsi_sell_at.keys():
        #     self.rsi_sell_at[symbol] = self.starting_rsi_sell_at
        if symbol not in self.time_rsi_cutoff_last_adjusted.keys():
            self.time_rsi_cutoff_last_adjusted[symbol] = time.time()
        
        ## Check the RSI ##
        historical_data = rh.get_crypto_historicals(symbol=symbol, interval=RSI_WINDOW[symbol], span=RSI_SPAN[symbol], bounds="24_7", info=None)
        df = pd.DataFrame(historical_data)
        # convert prices to float since values are given as strings
        df["close_price"] = pd.to_numeric(df["close_price"], errors='coerce') 
        # Get the current RSI
        rsi = RSI(df["close_price"], period = RSI_PERIOD[symbol], current_only=True)
        self.rsi[symbol] = rsi
        
        self.quantity_on_hand['USD'] = self.cash_on_hand = cash = self.check_cash_on_hand()
        crypto_on_hand = self.check_cash_on_hand(symbol = symbol)
        self.quote[symbol] = quote = crypto_on_hand['quote']
        self.quantity_on_hand[symbol] = quantity = crypto_on_hand['quantity']
        self.cost_basis[symbol] = cost_basis = crypto_on_hand['cost']
        self.symbol_value[symbol] = quote*quantity
        try:
            self.symbol_pnl_percent[symbol] = 100*(quote*quantity - cost_basis)/cost_basis
        except ZeroDivisionError:
            self.symbol_pnl_percent[symbol] = 0
        
        # TODO: Store data in instance variables and move output to mainloop or special output function:
        sign = "+" if self.symbol_pnl_percent[symbol] >= 0 else ""
        print("Sym\tQuote\tQty\tVal\tPnL\tCost\tRSI\tBuy@\tSell@\tTrades\tTP\tSL")
        print(f"USD\t1\t{self.quantity_on_hand['USD']:.2f}\t${self.quantity_on_hand['USD']:.2f}\t\t\t")
        if symbol == 'BTC':
            quote_prec = 0
            quant_prec = 5
        elif symbol == 'DOGE':
            quote_prec = 5
            quant_prec = 0
        else:
            quote_prec = 2
            quant_prec = 5
        print(f"{symbol}", end='\t')
        print(f"{self.quote[symbol]:.{quote_prec}f}", end='\t')
        print(f"{self.quantity_on_hand[symbol]:.{quant_prec}f}", end='\t')
        print(f"${self.symbol_value[symbol]:.2f}", end='\t')
        print(f"{sign}{self.symbol_pnl_percent[symbol]:.2f}%", end='\t')
        print(f"${self.cost_basis[symbol]:.2f}", end='\t')
        print(f"{self.rsi[symbol]:.2f}", end='\t')
        print(f"{self.rsi_buy_at[symbol]:.2f}", end='\t')
        print(f"{self.rsi_sell_at[symbol]:.2f}", end='\t')
        print(f"{self.symbol_trades[symbol]}",end='\t')
        print(f"{self.symbol_take_profits[symbol]}|{self.take_profit_percent[symbol]}%", end='\t')
        print(f"{self.symbol_stop_losses[symbol]}|{self.stop_loss_percent[symbol]}%", end='\t')
        
        
        ## Adjust RSI buy/sell levels towards the defaults.
        self.adjust_rsi(symbol)
        
        ## Check for stop loss / take profits:
        if self.take_profit_percent[symbol] is not None and self.symbol_pnl_percent[symbol] > self.take_profit_percent[symbol]:
            # info = rh.order_sell_crypto_limit(symbol, quantity, round(0.999*quote,2))
            info = self.trigger_tx(symbol, quantity, round(0.99*quote, 2), side="sell", quantity_on_hand = quantity)
            if info is not None: 
                print(f"Take profit triggered!  Selling {quantity} of {symbol}")
                self.total_trades += 1
                self.total_take_profits += 1
                self.symbol_take_profits[symbol] += 1
            return # skip checking rsi this time around
        
        elif self.stop_loss_percent[symbol] is not None and self.symbol_pnl_percent[symbol] < -1*self.stop_loss_percent[symbol]:
            info = self.trigger_tx(symbol, quantity, round(0.99*quote, 2), side="sell", quantity_on_hand = quantity)
            # info = rh.order_sell_crypto_limit(symbol, quantity, round(0.99*quote, 2))
            # rh.order_sell_crypto_by_quantity(symbol, round(quantity,6))
            if info is not None:
                print(f"Stop loss triggered! Selling {quantity} of {symbol}")
                self.total_trades += 1
                self.total_stops_losses += 1
                self.symbol_stop_losses[symbol] += 1
                # Step RSI buy cutoff down so we don't buy again right away
                self.bump_rsi(symbol, 'buy', 5) # 5x normal adjustment
            # pprint(info)
            return # skip checking rsi
        
        # Check RSI to see if we should buy or sell
        if rsi <= self.rsi_buy_at[symbol]:
            info = self.trigger_tx(symbol, quantity = None, price = 1.01*quote, side = 'buy', cash_on_hand = cash)
            if info is not None:
                try:
                    if not isinstance(info['quantity'], list):
                        print(f"Buying: {symbol}") # {info['quantity']:.6f} at {info['price']:.2f} ({info['quantity']*info['price']:.2f})")
                        self.bump_rsi(symbol, 'buy')
                        self.total_trades += 1
                        self.symbol_trades[symbol] += 1
                except (ValueError, KeyError):
                    logging.warning(f"Failed buying: {info}")
                    
        elif rsi >= self.rsi_sell_at[symbol]:
            info = self.trigger_tx(symbol, quantity = None, price = None, side = 'sell', quantity_on_hand = quantity)
            if info is not None:
                try:
                    if not isinstance(info['quantity'], list):
                        print(f"Selling: {symbol}") # {info['quantity']:0.6f} at {info['price']:.2f} ({info['quantity']*info['price']:.2f})")
                        self.bump_rsi(symbol, 'sell')
                        self.total_trades += 1     
                        self.symbol_trades[symbol] += 1
                except (ValueError, KeyError):
                    logging.warning(f"Failed selling: {info}")
    def trigger_tx(self, symbol, quantity = None, price = None, side = None, cash_on_hand = None, quantity_on_hand = None):
        """ Attempts to make a trade. Returns None if no trade was made. """
        info = None
        if side not in {"buy", "sell"}:
            raise Exception("side should be 'buy' or 'sell'")
        if side == 'buy':
            if cash_on_hand is None:
                cash_on_hand = self.check_cash_on_hand(symbol="USD")
            max_allocation = MAX_ALLOCATION[symbol]
            if MAX_ALLOCATION_IS_PERCENT:
                max_allocation = (max_allocation/100)*self.total_value
            if cash_on_hand < MIN_DOLLARS_PER_TRADE: return
            if price is None:
                raise Exception("Price cannot be None. Calcuate a price or change the code to calculate a default price.")
            if symbol == 'DOGE':
                price = round(price, 6)
            else:
                price = round(price, 2)
            
            if quantity is None: 
                # price is not None and quantity is None
                # so calculate a quantity:

                buy_amount = min(cash_on_hand, MAX_DOLLARS_PER_BUY[symbol], max_allocation - self.symbol_value[symbol])
                if cash_on_hand - buy_amount < MIN_DOLLARS_PER_TRADE:
                    # If buy would leave us with less cash than we can trade with, just use all of it.
                    buy_amount = cash_on_hand
                
                quantity = round(buy_amount/price , self.ndigits[symbol])
                info = rh.order_buy_crypto_limit(symbol, quantity, price)
                #if symbol == 'DOGE':
                #   info = rh.order_buy_crypto_by_price(symbol, round(buy_amount, self.ndigits[symbol]))
                #   quantity = round(buy_amount/price , self.ndigits[symbol])
                #   info = rh.order_buy_crypto_limit(symbol, quantity, price)
                #else:
                #    #info = rh.order_buy_crypto_by_price(symbol, buy_amount)
            else:
                info = rh.order_buy_crypto_limit(symbol, quantity, price)
            
        else: # side == 'sell'
            if price is None:
                price = float(rh.get_crypto_quote(symbol)['bid_price'])
            if symbol == 'DOGE':
                price = round(price, 8)
            else:
                price = round(price, 2)
            if quantity_on_hand is None:
                raise Exception("quantity_on_hand cannot be None. Calcuate a quantity or change the code to calculate a default price.")
            if quantity_on_hand*price < MIN_DOLLARS_PER_TRADE:
                return
            if quantity is None:
                quantity = round(MAX_DOLLARS_PER_SELL[symbol]/price, self.ndigits[symbol])
                if price*quantity_on_hand < MAX_DOLLARS_PER_SELL[symbol] or price*(quantity_on_hand - quantity) < MIN_DOLLARS_PER_TRADE:
                    quantity = quantity_on_hand
            else:
                pass
            info = rh.order_sell_crypto_by_quantity(symbol, quantity)

        retval = info
        if info is not None:
            with Capturing() as output:
                print(f"Trade executed: symbol = {symbol}, quantity = {quantity}, price = {price}, side = {side}\n")
                print(info)
            try:
                self.order_ids[time.time()] = info['id']                
                # self.bump_rsi(symbol,side)
            except KeyError:
                retval = None
            logging.info(output)
        return retval
    def adjust_rsi(self, symbol):
        # increase the buy level, but no more than the default
        self.rsi_buy_at[symbol] = min(self.starting_rsi_buy_at[symbol],
                                      self.rsi_buy_at[symbol] + (time.time() - self.time_rsi_cutoff_last_adjusted[symbol])*self.rsi_reset_buy_speed[symbol])
        # decrease the sell level, but no less than the default
        self.rsi_sell_at[symbol] = max(self.starting_rsi_sell_at[symbol], 
                                       self.rsi_sell_at[symbol] - (time.time() - self.time_rsi_cutoff_last_adjusted[symbol])*self.rsi_reset_sell_speed[symbol])
        # Store the last time the rsi was adjusted
        self.time_rsi_cutoff_last_adjusted[symbol] = time.time()
    def bump_rsi(self, symbol, side, multiple=1):
        """ """
        assert side in {'buy','sell'}
        if side == 'buy':
            self.rsi_buy_at[symbol] = max(0, self.rsi_buy_at[symbol] - multiple*self.rsi_step_buy_when_triggered[symbol])
        elif side == 'sell':
            self.rsi_sell_at[symbol] = min(100, self.rsi_sell_at[symbol] + multiple*self.rsi_step_sell_when_triggered[symbol])

def main():
   
    client = RobinTrader(username = USERNAME, password = PASSWORD)
    client.mainloop(sleep_time = MAIN_LOOP_SLEEP_TIME)
    
if __name__ == "__main__":
    main()
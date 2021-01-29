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

MAX_ALLOCATION_IS_PERCENT = True
# The maximum allocation of funds towards any one particular symbol
DEFAULT_MAX_ALLOCATION = 50
# If MAX_ALLOCATION is a dict, we can set maximum allocations for each symbol individually
MAX_ALLOCATION = defaultdict(lambda: DEFAULT_MAX_ALLOCATION)
# values default to DEFAULT_MAX_ALLOCATION when commented / undefined
MAX_ALLOCATION['BTC'] = 61.8
# MAX_ALLOCATION['BTC'] = None # Equivalent to 100% / no limit
MAX_ALLOCATION['ETH'] = 40
MAX_ALLOCATION['ETC'] = 30
MAX_ALLOCATION['LTC'] = 10
MAX_ALLOCATION['DOGE'] = 10

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

# If set to True, stops will tighten as price moves:
# If the position is in profit, the stop loss trail amount will decrease
# to match the difference in stop to quote price
MONOTONE_STOPLOSS = True

# per individual trade
MIN_DOLLARS_PER_TRADE = 2.00 # no need to change 
#DEFAULT_MAX_DOLLARS_PER_BUY = 5.50
DEFAULT_MAX_DOLLARS_PER_BUY = float('Inf')
DEFAULT_MAX_DOLLARS_PER_SELL = float('Inf')
MAX_DOLLARS_PER_BUY = defaultdict(lambda: DEFAULT_MAX_DOLLARS_PER_BUY)
MAX_DOLLARS_PER_SELL = defaultdict(lambda: DEFAULT_MAX_DOLLARS_PER_SELL)
MAX_DOLLARS_PER_BUY['BTC'] = 55
MAX_DOLLARS_PER_BUY['ETH'] = 55
# MAX_DOLLARS_PER_BUY['LTC'] = float('Inf')
# MAX_DOLLARS_PER_SELL['BTC'] = 50
# MAX_DOLLARS_PER_SELL['ETH'] = 50 

# Don't trigger a sell unless PNL % is at least this much: 
# (Except stop loss)
# Set to None to disable
DEFAULT_REQUIRE_PNL_TO_SELL = 1 
REQUIRE_PNL_TO_SELL = defaultdict(lambda: DEFAULT_REQUIRE_PNL_TO_SELL)
REQUIRE_PNL_TO_SELL['BTC'] = 2
REQUIRE_PNL_TO_SELL['ETH'] = 1
REQUIRE_PNL_TO_SELL['LTC'] = 1

# Period for RSI calculation (number of data frames)
DEFAULT_RSI_PERIOD = 9
RSI_PERIOD = defaultdict(lambda: DEFAULT_RSI_PERIOD)
RSI_PERIOD['BTC'] = 9
RSI_PERIOD['ETH'] = 9
RSI_PERIOD['LTC'] = 7

# Size of data frame for calculating RSI
# Can be '15second', '5minute', '10minute', 'hour', 'day', or 'week'
# '15second' may result in an error if RSI_SPAN is longer than 'hour'
DEFAULT_RSI_WINDOW = 'hour'
RSI_WINDOW = defaultdict(lambda: DEFAULT_RSI_WINDOW)
RSI_WINDOW['BTC'] = 'hour'
RSI_WINDOW['ETH'] = 'hour'
RSI_WINDOW['LTC'] = '5minute'

# The entire time frame to collect data points. Can be 'hour', 'day', 'week', 'month', '3month', 'year', or '5year'
# If the span is too small to fit RSI_PERIOD number of RSI_WINDOWs then there will be an error
DEFAULT_RSI_SPAN = 'day'
RSI_SPAN = defaultdict(lambda: DEFAULT_RSI_SPAN)
RSI_SPAN['BTC'] = 'day'
RSI_SPAN['ETH'] = 'day'
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
RSI_BUY_AT['LTC'] = 30
RSI_SELL_AT['LTC'] = 70

# If step is non-zero, the rsi buy or sell levels will be adjusted down or up after each buy or sell.
# Set higher to take advantage of slower price movements in one direction
DEFAULT_RSI_STEP_BUY_WHEN_TRIGGERED = 1
DEFAULT_RSI_STEP_SELL_WHEN_TRIGGERED = 10
RSI_STEP_BUY_WHEN_TRIGGERED = defaultdict(lambda: DEFAULT_RSI_STEP_BUY_WHEN_TRIGGERED)
RSI_STEP_SELL_WHEN_TRIGGERED = defaultdict(lambda: DEFAULT_RSI_STEP_SELL_WHEN_TRIGGERED)
RSI_STEP_BUY_WHEN_TRIGGERED['BTC'] = 20
RSI_STEP_SELL_WHEN_TRIGGERED['BTC'] = 15
RSI_STEP_BUY_WHEN_TRIGGERED['ETH'] = 20
RSI_STEP_SELL_WHEN_TRIGGERED['ETH'] = 15
RSI_STEP_BUY_WHEN_TRIGGERED['LTC'] = 35
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


## Misc settings ##

# Time to wait between loops (seconds)
MAIN_LOOP_SLEEP_TIME = 6.15

# If true, print extra information to console
DEBUG = DEBUG_INFO = False

# END CONSTANTS #
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
     sys.stdout.write("\x1b7\x1b[%d;%df%s\x1b8" % (x, y, text))
     sys.stdout.flush()
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
    """ A dict d initialized with a dict D or callable C,
        whose default value for d[k] is D[k] or C(k), respectively. """
    def __init__(self, defaults):
        self.defaults = defaults
    def __missing__(self, key):
        if callable(self.defaults):
            return self.defaults(key)
        else:
            return self.defaults[key]

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
        self.buy_price = smartdict(lambda k: self.cost_basis[k]/self.quantity_on_hand[k])
        self.symbol_value = defaultdict(float) # Value in usd
        self.symbol_pnl_percent = defaultdict(float)

        self.symbol_trades = defaultdict(int) 
        self.symbol_take_profits = defaultdict(int) # How many take profits triggered
        self.stop_losses_triggered = defaultdict(int) # How many stops triggered
        self.rsi = defaultdict(float)
        self.time_rsi_cutoff_last_adjusted = defaultdict(lambda: self.up_since)

        self.take_profit_percent = TAKE_PROFIT_PERCENT
        self.stop_loss_percent = STOP_LOSS_PERCENT
        self.stop_loss_quote = smartdict(lambda k: self.quote[k]*(100-self.stop_loss_percent[k])/100)
        self.stop_loss_delta = smartdict(lambda k: self.quote[k] - self.stop_loss_quote[k])

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

        ## Arguments
        global DEBUG
        if "--debug" in sys.argv:
            DEBUG = True
        
        ## Actions performed on start up:

        # Login
        rh.login(username=username, password=password, expiresIn=86400, by_sms=TWO_FACTOR_IS_SMS)
        rh.logout()
        rh.login(username=username, password=password, expiresIn=86400, by_sms=TWO_FACTOR_IS_SMS)
        # Load orders that may have been placed before this program started
        self.load_active_crypto_orders()
        self.quantity_on_hand['USD'] = self.check_cash_on_hand()


        # DEBUG
        if DEBUG:
            with open("account-data.log", 'w') as f:
                for line in pformat(rh.load_phoenix_account(), indent=4):
                    f.write(line)
        
    def check_cash_on_hand(self, symbol = "USD"):
        cash_on_hand = 0
        if symbol == "USD":
            info = rh.load_phoenix_account()

            cash_on_hand = float(info['account_buying_power']['amount'])
            
            # TODO:
            # If we want to separate this bot from the rest of the acct, then we will need to 
            # do other calculations here based on orders placed and total amount willing to invest.
            # If we're fine using the entire account balance for this bot, then we only need 
            # to return the account_buying_power.
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
        errors = (TimeoutError, ConnectionError) if DEBUG else (TypeError, KeyError, TimeoutError, ConnectionError)
        while i < loops:
            try:
                # Work around since output is already coming from self.rsi_based_buy_sell
                # TODO: Make output less hackish by storing relevant data in instance variables and delgate output to specialized function
                outputs = []
                for symbol in SYMBOLS:
                    with Capturing() as output:
                        self.rsi_based_buy_sell(symbol = symbol)
                    outputs.append(output)

                heading = f"┃{time.ctime()} │ " + \
                          f"Uptime: {datetime.timedelta(seconds = int(time.time()-self.up_since))}  │ " + \
                          f"Total Trades: {self.total_trades} ({self.total_stops_losses} SL / {self.total_take_profits} TP)" 
                print(f"┏{'━'*(12*8-5)}┓")
                line_size = 8*12
                print(f"{heading}{' '*(line_size-len(heading)-4)}┃")
                print(f"┗{'━'*(12*8-5)}┛")
                # for title and usd
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
            
            except errors: # (TypeError, KeyError, TimeoutError):
                # Probably 504 server error, and robin_stocks tried subscript NoneType object 
                # or KeyError
                print(f"Server busy. Waiting {int(2*MAIN_LOOP_SLEEP_TIME)}s to retry.")
                time.sleep(MAIN_LOOP_SLEEP_TIME)
            

            print("")
            time.sleep(sleep_time)
            i += 1
    def rsi_based_buy_sell(self, symbol):
        """ Check the RSI and possibly place a buy or sell order """
        
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

        ## If quantity is 0, reset trailing stops
        if self.quantity_on_hand[symbol] == 0: 
            if symbol in self.stop_loss_quote.keys(): 
                del self.stop_loss_quote[symbol]
                del self.stop_loss_delta[symbol]

        try:
            self.symbol_pnl_percent[symbol] = 100*(quote*quantity - cost_basis)/cost_basis
        except ZeroDivisionError:
            self.symbol_pnl_percent[symbol] = 0
        
        # TODO: Store data in instance variables and move output to mainloop or special output function:
        sign = "+" if self.symbol_pnl_percent[symbol] >= 0 else ""
        print("Sym\tQuote\tQty\tVal\tPnL\tCost\tStop\tRSI\tBuy@\tSell@\tT|P|L\tTP|SL")
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
        print(f"{self.stop_loss_quote[symbol]:.{quote_prec}f}", end='\t')
        print(f"{self.rsi[symbol]:.2f}", end='\t')
        print(f"{self.rsi_buy_at[symbol]:.2f}", end='\t')
        print(f"{self.rsi_sell_at[symbol]:.2f}", end='\t')
        print(f"{self.symbol_trades[symbol]}|{self.symbol_take_profits[symbol]}|{self.stop_losses_triggered[symbol]}",end='\t')
        print(f"{self.take_profit_percent[symbol]}%|{self.stop_loss_percent[symbol]}%", end='\t')
        print()

        ## Adjust stop if price is high, but only if quantity is > 0
        if self.quantity_on_hand[symbol] > 0 and self.quote[symbol] > self.stop_loss_quote[symbol] + self.stop_loss_delta[symbol]:
            self.stop_loss_quote[symbol] = self.quote[symbol] - self.stop_loss_delta[symbol]
        elif MONOTONE_STOPLOSS:
            if self.symbol_pnl_percent[symbol] > 0:
                self.stop_loss_delta[symbol] = max(0,self.quote[symbol] - self.stop_loss_quote[symbol])
                
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
                self.stop_losses_triggered[symbol] += 1
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
                    
        elif self.quantity_on_hand[symbol] > 0 and rsi >= self.rsi_sell_at[symbol]:
            if REQUIRE_PNL_TO_SELL[symbol] is not None and\
                self.symbol_pnl_percent[symbol] < REQUIRE_PNL_TO_SELL[symbol]:
                print(f"RSI sell level met for {symbol}, but PnL not high enough")
            else:
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
#Use with Python3
#Installation:
#pip install robin_stocks pandas

# Imports for builtin modules:
import time

# Imports for installed modules:
import robin_stocks as rh
import pandas as pd

### Constants ###
#  ~~~~~~~~~~~  #

## Account related ##
USERNAME = "replace this string with your username" 
PASSWORD = "replace with your password" 

# Set True to do 2fa by SMS, or False to do 2fa by email.
TWO_FACTOR_IS_SMS = False

## Strategy related ##

# Set RSI levels to buy/sell at
DEFAULT_RSI_BUY_AT = 30
DEFAULT_RSI_SELL_AT = 70

# per individual trade
MIN_DOLLARS_PER_TRADE = 2
MAX_DOLLARS_PER_TRADE = 7 

# How much money to allocate to the RSI strategy at first.
# Profits will be continue to be traded.
# DEFAULT_DOLLAR_ALLOCATION = 25.00 # in dollars. Not currently used. Assuming using whole account


## Misc settings ##

# Time to wait between loops (seconds)
MAIN_LOOP_SLEEP_TIME = 7

# If true, print extra information to console
DEBUG_INFO = False

###~#~#~#~#~#~###

def RSI(prices, current_only=False, period = 7):
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
    

class RobinTrader:
    """ Class to handle trading instance """
    def __init__(self, username, password):
        self.username = username
        self.password = password
        
        self.cash_on_hand = None
        
        rh.login(username=username, password=password, expiresIn=86400, by_sms=TWO_FACTOR_IS_SMS)
    def check_cash_on_hand(self):
        info = pd.DataFrame(rh.load_phoenix_account())
        if DEBUG_INFO: print(info)
        # keys are 
        """['account_buying_power', 'cash_available_from_instant_deposits',
       'cash_held_for_currency_orders', 'cash_held_for_dividends',
       'cash_held_for_equity_orders', 'cash_held_for_options_collateral',
       'cash_held_for_orders', 'crypto', 'crypto_buying_power', 'equities',
       'extended_hours_portfolio_equity', 'instant_allocated',
       'levered_amount', 'near_margin_call', 'options_buying_power',
       'portfolio_equity', 'portfolio_previous_close', 'previous_close',
       'regular_hours_portfolio_equity', 'total_equity',
       'total_extended_hours_equity', 'total_extended_hours_market_value',
       'total_market_value', 'total_regular_hours_equity',
       'total_regular_hours_market_value', 'uninvested_cash',
       'withdrawable_cash']"""
        uninvested_cash = float(info['uninvested_cash']['amount'])
        # print(f"uninvested_cash: ${uninvested_cash}")
        cash_on_hand = uninvested_cash
        
        # TODO:
        # If we want to separate this bot from the rest of the acct, then we will need to 
        # do other calculations here based on orders placed and total amount willing to invest.
        # If we're fine using the entire account balancefor this bot, then we only need 
        # to return the uninvested_cash.
        
        return cash_on_hand
        
    def mainloop(self, sleep_time = MAIN_LOOP_SLEEP_TIME, loops = float("Inf")):
        """  """
        
        i = 0
        while i < loops:
            if DEBUG_INFO:
                print(f"Loop {i+1}")
                
            self.rsi_based_buy_sell(symbol = "ETH")
            
            time.sleep(sleep_time)
            i += 1

    def rsi_based_buy_sell(self, symbol, rsi_buy_at = DEFAULT_RSI_BUY_AT, rsi_sell_at = DEFAULT_RSI_SELL_AT):
        """ Check the RSI and possibly place a buy or sell order """
        
        # Check the RSI
        historical_data = rh.get_crypto_historicals(symbol=symbol, interval="5minute", span="hour", bounds="24_7", info=None)
        df = pd.DataFrame(historical_data)
        
        # convert prices to float since values are given as strings
        df["close_price"] = pd.to_numeric(df["close_price"], errors='coerce') 
        # Get the current RSI
        rsi = RSI(df["close_price"], current_only=True)
        print(f"RSI: {rsi}")
        if rsi <= rsi_buy_at:
            self.trigger_buy(symbol=symbol)
        elif rsi >= rsi_sell_at:
            self.trigger_sell(symbol=symbol)
    def trigger_buy(self, symbol):
        print("Buy triggered!")
        cash_on_hand = self.check_cash_on_hand()
        print(f"Have ${cash_on_hand}")
        if cash_on_hand < MIN_DOLLARS_PER_TRADE:
            print(f"Not enough to cash buy {symbol}.")
            return
        
        info = rh.get_crypto_info(symbol)
        if DEBUG_INFO: print(info)

        buy_amount = min(cash_on_hand, MAX_DOLLARS_PER_TRADE)
        print(f"Buying ${buy_amount} worth of {symbol}")
        info = rh.order_buy_crypto_by_price("ETH", buy_amount)
        if DEBUG_INFO: print(info)
        ### info looks like e.g.
        # {'account_id': '0123456789-abcd-ef01-2345-67890abcdef', 'average_price': None, 'cancel_url': 'https://nummus.robinhood.com/orders/111111-2222-3333-4444-567890abcdef/cancel/', 'created_at': '2021-01-21T23:19:54.976785-05:00', 'cumulative_quantity': '0.000000000000000000', 'currency_pair_id': '76637d50-c702-4ed1-bcb5-5b0732a81f48', 'executions': [], 'id': '111111-2222-3333-4444-567890abcdef', 'last_transaction_at': None, 'price': '1177.840000000000000000', 'quantity': '0.021200000000000000', 'ref_id': '543210f-edcb-aaaa-bbbb-01234543210', 'rounded_executed_notional': '0.00', 'side': 'buy', 'state': 'unconfirmed', 'time_in_force': 'gtc', 'type': 'market', 'updated_at': '2021-01-21T23:19:55.196963-05:00'}
        # or 
        # {'non_field_errors': ['Insufficient holdings.']}
        # or 
        # {'quantity': ['Order quantity is too small.']}
        if 'non_field_errors' in info.keys():
            # Probably insufficient money...
            if DEBUG_INFO: print(f"Error(s): {info['non_field_errors']}")
        elif 'quantity' in info.keys() and isinstance(info['quantity'], list):
            # Probably order quantity is too small...
            if DEBUG_INFO: print(f"Error(s): {info['quantity']}")
        else:
            # Probably worked
            pass
        
        return
    def trigger_sell(self, symbol):
        print("Sell triggered!")
        
        info = rh.get_crypto_positions()
        quantity = float( list(filter(lambda x: x['currency']['code'] == symbol, info))[0]['quantity'] )
        print(f"Have {quantity} {symbol}")
        quote = float(rh.get_crypto_quote(symbol)['bid_price'])
        
        if quote*quantity >= MAX_DOLLARS_PER_TRADE:
            quantity = round(MAX_DOLLARS_PER_TRADE/quote, 6)
        
        print(f"Selling {quantity} of {symbol}")
        if quote*quantity < MIN_DOLLARS_PER_TRADE:
            print(f"Value of {symbol} below minimum. Skipping.")
            return
        
        info = rh.order_sell_crypto_by_quantity(symbol, quantity)
        
        if DEBUG_INFO: print(info)
        
        return 
    
def main():
    
    client = RobinTrader(username = USERNAME, password = PASSWORD)
    client.mainloop(sleep_time = MAIN_LOOP_SLEEP_TIME)
    
    rh.logout()

if __name__ == "__main__":
    main()
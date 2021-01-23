#Use with Python3
#Installation:
#pip install robin_stocks pandas pickle

# Imports for builtin modules:
import time

# Imports for installed modules:
import robin_stocks as rh
import pandas as pd
# from pprint import pprint
# import pickle # (not yet used)

try:
    # uncomment this line to force builtin constants in next block, or don't include config.py file
    # raise ModuleNotFoundError 
    from config import *
except ModuleNotFoundError:
    # If config.py is not available, this program uses the constants below.
    # Consider copying your settings to config.py so that it is easier
    # to save settings between updates.
    ### Constants ###
    #  ~~~~~~~~~~~  #

    ## Account related ## 
    # Replace with your Robinhood login info
    USERNAME = "YourUsernameHere"
    PASSWORD = "YourPasswordHere" 

    # Set True to do 2fa by SMS, or False to do 2fa by email.
    TWO_FACTOR_IS_SMS = False

    ## Strategy related ##

    # Percentage above cost basis before taking profit by selling position
    # Change to None to disable
    TAKE_PROFIT_PERCENT = 0.45
    # TAKE_PROFIT_PERCENT = None
    
    # Percentage below cost basis before stopping loss by selling position
    # Change to None to disable
    STOP_LOSS_PERCENT = 0.45
    # STOP_LOSS_PERCENT = None

    # Set RSI levels to buy/sell at
    DEFAULT_RSI_BUY_AT = 30
    DEFAULT_RSI_SELL_AT = 70    

    # If this is non-zero, the rsi buy or sell levels will be adjusted down or up after each buy or sell.
    # Set higher to take advantage of longer price movements in one direction
    RSI_STEP_WHEN_TRIGGERED = 5
    # The rate (in RSI/second) to adjust the RSI cutoffs back towards the default levels.
    # Set lower to take advantage of longer price movements in one direction
    RSI_ADJUST_RESET_SPEED = 0.01

    # per individual trade
    MIN_DOLLARS_PER_TRADE = 2.25
    MAX_DOLLARS_PER_TRADE = 5.50

    # How much money to allocate to the RSI strategy at first.
    # Profits will be continue to be traded.
    # DEFAULT_DOLLAR_ALLOCATION = 25.00 # in dollars. Not currently used. Assuming using whole account


    ## Misc settings ##

    # Time to wait between loops (seconds)
    MAIN_LOOP_SLEEP_TIME = 5

    # If true, print extra information to console
    DEBUG_INFO = False

    ###~#~#~#~#~#~###


# Useful functions
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
    def __init__(self, username, password, starting_rsi_buy_at = DEFAULT_RSI_BUY_AT, starting_rsi_sell_at = DEFAULT_RSI_SELL_AT):
        self.username = username
        self.password = password
        
        self.cash_on_hand = None
        
        self.rsi_buy_at = starting_rsi_buy_at
        self.rsi_sell_at = starting_rsi_sell_at
        
        
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
            # if DEBUG_INFO: print(f"Loop {i+1}")
            try:
                self.rsi_based_buy_sell(symbol = "ETH")
            except (TypeError, KeyError):
                # Probably 504 server error, and robin_stocks tried subscript NoneType object 
                # or KeyError
                print("Server busy. Waiting 10s to retry.")
                time.sleep(10)
            
            
            
            
            
            #Adjust RSI buy/sell levels towards the defaults.
            # increase the buy level, but no more than the default
            self.rsi_buy_at = min(DEFAULT_RSI_BUY_AT, self.rsi_buy_at + MAIN_LOOP_SLEEP_TIME*RSI_ADJUST_RESET_SPEED)
            # decrease the sell level, but no less than the default
            self.rsi_sell_at = max(DEFAULT_RSI_SELL_AT, self.rsi_sell_at - MAIN_LOOP_SLEEP_TIME*RSI_ADJUST_RESET_SPEED)
            print("")
            time.sleep(sleep_time)
            i += 1

    def rsi_based_buy_sell(self, symbol):
        """ Check the RSI and possibly place a buy or sell order """
        
        # Check the RSI
        historical_data = rh.get_crypto_historicals(symbol=symbol, interval="5minute", span="hour", bounds="24_7", info=None)
        df = pd.DataFrame(historical_data)
        
        # convert prices to float since values are given as strings
        df["close_price"] = pd.to_numeric(df["close_price"], errors='coerce') 
        # Get the current RSI
        rsi = RSI(df["close_price"], current_only=True)
        print(f"RSI: {round(rsi,4)}\tCutoffs: {round(self.rsi_buy_at,3)}/{round(self.rsi_sell_at,3)}")
        cash = self.check_cash_on_hand()
        quote = float(rh.get_crypto_quote(symbol)['bid_price'])
        crypto_positions = rh.get_crypto_positions()
        quantity = float( list(filter(lambda x: x['currency']['code'] == symbol, crypto_positions))[0]['quantity'] )
        cost_basis = float( list(filter(lambda x: x['currency']['code'] == symbol, crypto_positions))[0]['cost_bases'][0]['direct_cost_basis'] )
        try:
            pnl_percent = 100*(quote*quantity - cost_basis)/cost_basis
        except ZeroDivisionError:
            pnl_percent = 0
        sign = "+" if pnl_percent >= 0 else ""
        print(f"Assets:\n\t{round(cash,2)} USD\n\t{round(quantity,6)} {symbol}\t Value: ${round(quote*quantity,2)} ({sign}{round(pnl_percent,2)}%)\tCost: ${round(cost_basis,2)}")
        print(f"Total Value: ${round(cash + quote*quantity,2)}")
        #print(f"Cost Basis of {symbol}: {cost_basis}")
        #print(f"Value of {symbol}: {quote*quantity}")
        
        # Check for stop loss / take profits:
        if TAKE_PROFIT_PERCENT is not None and pnl_percent > TAKE_PROFIT_PERCENT:
            info = rh.order_sell_crypto_limit(symbol, quantity, round(0.999*quote,2))
            print(f"Take profit triggered!  Selling {symbol}")
            # pprint(info)
            return # skip checking rsi this time around
            
        elif STOP_LOSS_PERCENT is not None and pnl_percent < -1*STOP_LOSS_PERCENT:
            info = rh.order_sell_crypto_limit(symbol, quantity, round(0.995*quote, 2))
            # rh.order_sell_crypto_by_quantity(symbol, round(quantity,6))
            print(f"Stop loss triggered! Selling {symbol}")
            # Step RSI buy cutoff down so we don't buy again right away
            self.rsi_buy_at = max(0, self.rsi_buy_at - RSI_STEP_WHEN_TRIGGERED)
            # pprint(info)
            return # skip checking rsi
        
        # Check for RSI
        if rsi <= self.rsi_buy_at:
            self.trigger_buy(symbol=symbol)
        elif rsi >= self.rsi_sell_at:
            self.trigger_sell(symbol=symbol)
            
    def trigger_buy(self, symbol, buy_amount = None):
        print("Buy triggered!")
        cash_on_hand = self.check_cash_on_hand()
        if cash_on_hand < MIN_DOLLARS_PER_TRADE:
            print(f"Not enough to cash buy {symbol}.")
            return
        
        # info = rh.get_crypto_info(symbol)
        # if DEBUG_INFO: print(info)
        if buy_amount is None:
            buy_amount = min(cash_on_hand, MAX_DOLLARS_PER_TRADE)
        print(f"Buying ${buy_amount} worth of {symbol}")
        info = rh.order_buy_crypto_by_price("ETH", buy_amount)
        if DEBUG_INFO: print(info)

        if 'non_field_errors' in info.keys():
            # Probably insufficient money...
            if DEBUG_INFO: print(f"Error(s): {info['non_field_errors']}")
        elif 'quantity' in info.keys() and isinstance(info['quantity'], list):
            # Probably order quantity is too small...
            if DEBUG_INFO: print(f"Error(s): {info['quantity']}")
        else:
            # Probably worked
            pass
        # Decrease rsi buy level
        self.rsi_buy_at = max(0, self.rsi_buy_at - RSI_STEP_WHEN_TRIGGERED)
        return
    def trigger_sell(self, symbol, sell_amount = None):
        print("Sell triggered!")
        
        quantity = float( list(filter(lambda x: x['currency']['code'] == symbol, rh.get_crypto_positions()))[0]['quantity'] )
        # print(f"Have {quantity} {symbol}")
        quote = float(rh.get_crypto_quote(symbol)['bid_price'])
        
        if quote*quantity >= MAX_DOLLARS_PER_TRADE:
            quantity = round(MAX_DOLLARS_PER_TRADE/quote, 6)
        
        print(f"Selling {quantity} of {symbol}")
        if quote*quantity < MIN_DOLLARS_PER_TRADE:
            print(f"Amount of {symbol} below minimum. Skipping.")
            return
        
        info = rh.order_sell_crypto_by_quantity(symbol, quantity)
        
        if DEBUG_INFO: print(info)
        
        
        # Increase rsi sell level
        self.rsi_sell_at = min(100, self.rsi_sell_at + RSI_STEP_WHEN_TRIGGERED)
        return 
    
def main():
    
    client = RobinTrader(username = USERNAME, password = PASSWORD)
    client.mainloop(sleep_time = MAIN_LOOP_SLEEP_TIME)
    
    rh.logout()

if __name__ == "__main__":
    main()
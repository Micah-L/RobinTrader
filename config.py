### Constants ###
#  ~~~~~~~~~~~  #

## Account related ## 
# Replace with your Robinhood login info
USERNAME = "YourUsernameHere"
PASSWORD = "YourPasswordHere" 

# Set True to do 2fa by SMS, or False to do 2fa by email.
TWO_FACTOR_IS_SMS = False

## Strategy related ##

# 
TAKE_PROFIT_PERCENT = 0.45
# TAKE_PROFIT_PERCENT = None
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
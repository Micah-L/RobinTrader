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
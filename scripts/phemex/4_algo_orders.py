######### CodingAlgoOrders
#connect to an exchange

import ccxt
import key_file as k
import time
import schedule

phemex = ccxt.phemex({
    'enableRateLimit': True,
    'apiKey': k.key,
    'secret': k.secret
})

#set up
bal = phemex.fetch_balance()
symbol = 'uBTCUSD'
size = 1
bid = 30000
params = { 'timeInForce': 'PostOnly',}

#making a simple order
order = phemex.create_limit_buy_order(symbol, size, bid, params)
print(order)

#how to cancel order
phemex.cancel_all_orders(symbol)

#create an order
#sleep for 10 and then cancel
print('just made order')
time.sleep(10)
phemex.cancel_all_orders(symbol)

#looping through code
go = True
while go == True:
    phemex.create_limit_buy_order(symbol, size, bid, params)

    time.sleep(5)

    phemex.cancel_all_orders(symbol)

    

def bot():
    try:
        phemex.create_limit_buy_order(symbol, size, bid, params)

        time.sleep(5)

        phemex.cancel_all_orders(symbol)

schedule.every(9).minutes.do(bot)

while True:
    try: 
        schedule.run_pending()
        time.sleep(1)

    except Exception as e:
        print('oops')
        print(f"An error occurred: {str(e)}")

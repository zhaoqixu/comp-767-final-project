# Deep Reinforcement Learning in Financial Portfolio Management


## Set up

Python Version

+ ***3.6***

Modules needed

+ ***tensorflow***
+ ***numpy*** 
+ ***pandas*** 
+ ***matplotlib***

## Using the environment

The environment provides supports for easily testing different reinforcement learning in portfolio management.
+ pgmain.py -  the entrance to run our training and testing PG method
+ ddpgmain.py -  the entrance to run DDPG method
+ ppomain.py -  the entrance to run PPO method
+ ./agent- contains ddpg.py, ppo.py, pg.py and baseline method(winner and losser)
+ ./data- contains America.csv for USA stock data, China.csv for China stock data. download_data.py can download China stock data by Tushare. environment.py generates states data for trading agents.
+ three configuration file for training or testing settings: config.json, ddpg.json, ppo.json

Training/Testing
```
python pgmain.py --mode=train
```

```
python pgmain.py --mode=test
```


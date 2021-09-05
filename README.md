# BSc-Thesis
Deep RL agent for trading using buy and sell signals as input

# Info
This repository uses [TensorTrade](https://github.com/PSS1998/tensortrade) as the enviornment for training of RL agent.<br/>
For more details on RL environment use [TensorTrade](https://github.com/PSS1998/tensortrade/tree/master/docs/source) documentation.<br/>
For more details on building your RL model use [Stable Baselines3](https://stable-baselines3.readthedocs.io/en/master/) documentation.<br/>
For more details on the API available for financial markets data through Finnhub read their [API documentation](https://finnhub.io/docs/api/introduction) which is implemented in this repository.<br/>

## Usage
In order to train and use a RL agent for the input signals, you have to do the following steps.<br/>
You can modify each step depending on your needs.<br/>
1. git clone https://github.com/PSS1998/BSc-Thesis.git <br/>
2. cd BSc-Thesis <br/>
3. git clone https://github.com/PSS1998/tensortrade.git <br/>
4. cd tensortrade <br/>
5. python -m pip install . <br/>
6. cd .. <br/>
7. python -m pip install -r requirements.txt <br/>
8. mkdir data <br/>
9. create an account in [Finnhub.io](https://finnhub.io/) and add your API to config.py. also change any other config you need. <br/>
10. python download_data.py <br/>
11. python RL_Train.py <br/>


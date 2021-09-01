

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensortrade.env.default as default

from tensortrade.env.default.renderers import PlotlyTradingChart, MatplotlibTradingChart
from tensortrade.feed.core import Stream, DataFeed, NameSpace
from tensortrade.oms.instruments import USD, BTC, ETH, LTC, Instrument
from tensortrade.oms.wallets import Wallet, Portfolio
from tensortrade.oms.exchanges import Exchange
from tensortrade.oms.services.execution.simulated import execute_order
# from tensortrade.agents import DQNAgent, A2CAgent

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3 import PPO, DQN, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, BaseCallback, EvalCallback
from stable_baselines3.common.env_checker import check_env

import torch as th


def analyze_profit(env, eval_name=""):

    def sharpe_ratio(returns):
        risk_free_rate = 0
        return (np.mean(returns) - risk_free_rate + 1e-9) / (np.std(returns) + 1e-9)
    
    def sortino_ratio(returns):
        target_returns = 0
        risk_free_rate = 0
        
        downside_returns = returns.copy()
        downside_returns[returns < target_returns] = returns ** 2

        expected_return = np.mean(returns)
        downside_std = np.sqrt(np.std(downside_returns))

        return (expected_return - risk_free_rate + 1e-9) / (downside_std + 1e-9)

    x = list(env.action_scheme.portfolio.performance.keys())
    duration = len(x)
    net_worths = [nw['net_worth'] for nw in env.action_scheme.portfolio.performance.values()]
    returns = pd.Series(net_worths).pct_change().dropna()
    risk_adjusted_return_sortino = sortino_ratio(returns)
    risk_adjusted_return_sharpe = sharpe_ratio(returns)
    print("risk_adjusted_return_sortino", risk_adjusted_return_sortino)
    print("risk_adjusted_return_sharpe", risk_adjusted_return_sharpe)

    total_num_trades = len(env.action_scheme.broker.trades)
    print("total_num_trades", total_num_trades)

    trades = [trade for sublist in env.action_scheme.broker.trades.values() for trade in sublist]
    diff = 0
    for trade in trades:
        diff += trade.step - diff
    average_time_between_trades = (diff/total_num_trades)*5
    print("average_time_between_trades", average_time_between_trades, "min")

    performance = pd.DataFrame.from_dict(env.action_scheme.portfolio.performance, orient='index')
    net_worths = performance.net_worth

    highwatermarks = net_worths.cummax()
    drawdowns = 1 - (1 + net_worths) / (1 + highwatermarks)
    max_drawdown = max(drawdowns)

    from itertools import accumulate
    drawdown_times = (drawdowns > 0).astype(np.int64)
    max_drawdown_time = (max(accumulate(drawdown_times, lambda x,y: (x+y)*y))*5)/1440

    total_drawdown_time = drawdown_times.groupby((drawdown_times != drawdown_times.shift()).cumsum()).cumsum().max()

    print("max_drawdown", max_drawdown)
    print("max_drawdown_time", max_drawdown_time, "day")
    print("total_drawdown_time", total_drawdown_time)

    current_net_worth = round(net_worths[len(net_worths)-1], 1)
    initial_net_worth = round(net_worths[0], 1)
    profit_percent = round((current_net_worth - initial_net_worth) / initial_net_worth * 100, 2)
    profit_percent_yearly = profit_percent/((duration/(8640*12)))

    print("profit_percent", profit_percent, "%")
    print("profit_percent_yearly", profit_percent_yearly, "%")

    f = open( eval_name+'_eval_result.txt', 'w' )
    f.write( eval_name + '\n' )
    f.write( 'risk_adjusted_return_sortino = ' + str(risk_adjusted_return_sortino) + '\n' )
    f.write( 'risk_adjusted_return_sharpe = ' + str(risk_adjusted_return_sharpe) + '\n' )
    f.write( 'total_num_trades = ' + str(total_num_trades) + '\n' )
    f.write( 'average_time_between_trades = ' + str(average_time_between_trades) + '\n' )
    f.write( 'max_drawdown = ' + str(max_drawdown) + '\n' )
    f.write( 'max_drawdown_time = ' + str(max_drawdown_time) + '\n' )
    f.write( 'total_drawdown_time = ' + str(total_drawdown_time) + '\n' )
    f.write( 'profit_percent = ' + str(profit_percent) + '\n' )
    f.write( 'profit_percent_yearly = ' + str(profit_percent_yearly) + '\n' )
    f.close()


def environment(env_type, data_file_name, action_scheme, reward_scheme, window_size):
    df = pd.read_csv('data/'+data_file_name+'.csv')
    df.rename(columns = {'time':'date'}, inplace = True)
    binance = Exchange("binance", service=execute_order)(Stream.source(list(df['close']), dtype="float").rename("USDT-ETH"))
    price_history = df[['date', 'open', 'high', 'low', 'close', 'volume']] 
    renderer_feed = DataFeed([
        Stream.source(price_history[c].tolist(), dtype="float").rename(c) for c in price_history]
    )
    df = df.drop(df.columns[0], axis=1)
    dataset = df.drop(columns=['date', 'open', 'high', 'low', 'close', 'volume', 'close_noise'], inplace=False)
    with NameSpace("binance"):
        binance_streams = [
                Stream.source(list(dataset[c]), dtype="float").rename(c) for c in dataset.columns
            ]
    feed = DataFeed(binance_streams)
    ETH = Instrument('ETH', 8, 'Ethereum')
    USDT = Instrument('USDT', 8, 'Thether')
    portfolio = Portfolio(USDT, [
        Wallet(binance, 10000 * USDT),
        Wallet(binance, 0 * ETH),
    ])
    if env_type == 'train':
        random_start = True
    elif env_type == 'eval':
        random_start = False
    chart_renderer1 = PlotlyTradingChart(
        # height=800,  # affects both displayed and saved file height. None for 100% height.
        save_format="jpeg",  # save the chart to an HTML file
    )
    chart_renderer2 = MatplotlibTradingChart(
        display=False,
        # height=800,  # affects both displayed and saved file height. None for 100% height.
        save_format="jpeg",  # save the chart to an HTML file
    )
    env = default.create(
        portfolio=portfolio,
        action_scheme=action_scheme,
        reward_scheme=reward_scheme,
        feed=feed,
        window_size=window_size,
        renderer_feed=renderer_feed,
        renderer=[
            # chart_renderer1, 
            chart_renderer2
            ],
        enable_logger=False,
        random_start=random_start
    )
    return env

def evaluate_env(agent, eval_env):
    env = eval_env
    count = 0
    # reward = 0
    obs = env.reset()
    dones = 0
    while dones == 0:
    # while True:
        count += 1
        if (count % 10000) == 0:
            print(count)
        # if count < 20000:
        #     obs, rewards, dones, info = env.step(0)
        #     continue
        action, _states = agent.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        # if(count == 1):
        #     obs, rewards, dones, info = env.step(1)
        # else:
        #     obs, rewards, dones, info = env.step(0)
        # reward += rewards
        if (count % 10000) == 0:
            performance = pd.DataFrame.from_dict(env.action_scheme.portfolio.performance, orient='index')
            print(performance.tail(1)['net_worth'].values[0])
            # print(dones)
            # env.render()
            # env.save()
    print(count)
    performance = pd.DataFrame.from_dict(env.action_scheme.portfolio.performance, orient='index')
    print(performance.tail(1)['net_worth'])
    return env

def save_render_env(env, eval_name="compared-to-BuyandHold_simple-three-type_4-million-step"):
    env.render()
    env.save()

    x = list(env.action_scheme.portfolio.performance.keys())
    y = [x['net_worth'] for x in env.action_scheme.portfolio.performance.values()]

    np.savez(eval_name+'.npz', step=x, net_worth=y)

    npzfile = np.load(eval_name+'.npz')
    print(npzfile.files)
    print(npzfile[npzfile.files[0]])
    print(npzfile[npzfile.files[1]])

    plt.xlabel('Step')
    plt.ylabel('Net Worth')
    plt.title('Performance')
    plt.plot(x, y)
    # plt.show()
    plt.savefig(eval_name+'-eval_result.png')



class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.net_worth = 0

    def _on_rollout_end(self) -> None:
        performance = pd.DataFrame.from_dict(self.model.env.get_attr('action_scheme')[0].portfolio.performance, orient='index')
        value = performance.tail(1)['net_worth'].values[0]
        self.net_worth = value
        self.logger.record("rollout/net_worth", self.net_worth)

    def _on_step(self) -> bool:
        return True

checkpoint_callback = CheckpointCallback(save_freq=1000000, save_path='./logs/', name_prefix='rl_model')
tensorboard_callback = TensorboardCallback()

eval_env = environment('eval', "BINANCE_BTCUSDT_5_signal2", "simple", "compared-to-BuyandHold", 15)

eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/eval/best_model', log_path='./logs/eval', n_eval_episodes=1, eval_freq=1000000, deterministic=True, render=False)

callback = CallbackList([
            checkpoint_callback, 
            tensorboard_callback, 
            eval_callback
            ])

# agent = PPO.load("trader")
# agent.set_env(env)

env = environment('train', "BINANCE_ETHUSDT_5_signal2", "simple", "compared-to-BuyandHold", 15)
# check_env(env)

def train_agent():

    # Custom actor (pi) and value function (vf) networks of two layers of size 32 and 16 each with Relu activation function
    policy_kwargs = dict(activation_fn=th.nn.Tanh, net_arch=[dict(pi=[64, 64], vf=[64, 64])])
    agent = PPO(ActorCriticPolicy, env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log="./tensorboard/", learning_rate=0.01)

    agent.learn(total_timesteps=14000000, tb_log_name="first_run", callback=callback)

    agent.save("trader")
    
def test_agent():

    agent = PPO.load("logs/rl_model_2000000_steps")

    # a = agent.policy.state_dict()
    # b = agent.get_parameters()

    env = evaluate_env(agent, eval_env)

    save_render_env(env, "simple_compared-to-BuyandHold_2-million-step_15_noise2")
    analyze_profit(env, eval_name="simple_compared-to-BuyandHold_2-million-step_15_noise2")
    
    
train_agent()
test_agent()

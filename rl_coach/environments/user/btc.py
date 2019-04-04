"""BTC trading environment. Trains on BTC price history to learn to buy/sell/hold.

This is an environment tailored towards TensorForce, not OpenAI Gym. Gym environments are
a standard used by many projects (Baselines, Coach, etc) and so would make sense to use; and TForce is compatible with
Gym envs. It's just that there's hoops to go through converting a Gym env to TForce, and it was ugly code. I actually
had it that way, you can search through Git if you want the Gym env; but one day I decided "I'm not having success with
any of these other projects, TForce is the best - I'm just gonna stick to that" and this approach was cleaner.

I actually do want to try NervanaSystems/Coach, that one's new since I started developing. Will require converting this
env back to Gym format. Anyone wanna give it a go?
"""

from box import Box
import copy
import json
import os

import gym
from gym import spaces
from rl_coach.environments.user.data.data import Data, Exchange, EXCHANGE
from gym.utils import seeding
import numpy as np
import pandas as pd


# See 6fc4ed2 for Scaling states/rewards

class BitcoinEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        self.hypers = Box(json.load(open(os.path.dirname(__file__) + '/config/btc.json')))
        # [sfan] 是否是保证金交易
        # 如果是保证金交易，则state和reward与价格差值有关；反之，与价格的比例有关
        self.leverage = self.hypers.EPISODE.leverage

        if self.leverage:
            self.start_cash, self.start_value = 0.0, .0  # .4, .4
            # self.hypers.EPISODE.stop_loss_dots_per_trade: 每次交易时的最大止损点数，如5个点，10个点
            self.stop_loss = (-1) * self.hypers.EPISODE.stop_loss_dots_per_trade
        else:
            # cash/val start @ about $3.5k each. You should increase/decrease depending on how much you'll put into your
            # exchange accounts to trade with. Presumably the agent will learn to work with what you've got (cash/value
            # are state inputs); but starting capital does effect the learning process.
            self.start_cash, self.start_value = 1.0, .0  # .4, .4
            # [sfan] stop loss value: minimal cash remained
            self.stop_loss = self.start_cash * self.hypers.EPISODE.stop_loss_fraction


        # [sfan] default: 'train'; can be set by 'set_mode' method
        self.mode = self.hypers.EPISODE.mode

        # We have these "accumulator" objects, which collect values over steps, over episodes, etc. Easier to keep
        # same-named variables separate this way.
        acc = dict(
            ep=dict(
                i=-1,  # +1 in reset, makes 0
                returns=[],
                uniques=[],
            ),
            step=dict(),  # setup in reset()
        )
        self.acc = Box(train=copy.deepcopy(acc), test=copy.deepcopy(acc))
        self.acc.train.step.hold_value = self.start_value + self.start_cash
        self.acc.test.step.hold_value = self.start_value + self.start_cash
        self.data = Data(window=self.hypers.STATE.step_window, indicators={}, mode=self.mode, leverage=self.leverage)

        # gdax min order size = .01btc; kraken = .002btc
        self.min_trade = {Exchange.GDAX: .01, Exchange.KRAKEN: .002}[EXCHANGE]
        # self.update_btc_price()

        # Action space
        # see {last_good_commit_ for action_types other than 'single_discrete'
        # In single_discrete, we allow buy2%, sell2%, hold (and nothing else)
        # [sfan] 0: empty; 1: long position; 2: short position
        num_actions = len(self.hypers.ACTION.pct_map)
        self.actions_ = dict(type='int', shape=(), num_actions=num_actions)

        # Observation space
        # width = step-window (150 time-steps)
        # height = nothing (1)
        # channels = features/inputs (price actions, OHCLV, etc).
        self.cols_ = self.data.df.shape[1]
        shape = (self.hypers.STATE.step_window, 1, self.cols_)
        self.states_ = dict(type='float', shape=shape)

        self.action_space = spaces.Discrete(num_actions)
        self.observation_space = spaces.Box(low=-2, high=2, shape=(self.hypers.STATE.step_window, 1, self.cols_))

        self.seed()

        self.is_stop_loss = False
        self.terminal = False
        self.fee = 0

        # for render
        self.raw_states_ = None
        self.raw_next_states_ = None
        self.ohlc_history = None
        self.action_history = None
        self.fig = None

    @property
    def states(self): return self.states_

    @property
    def actions(self): return self.actions_

    # [sfan] mode: 'train' or 'test'
    def set_mode(self, mode):
        if self.mode != mode:
            self.mode = mode
            self.data = Data(window=self.hypers.STATE.step_window, indicators={}, mode=self.mode, leverage=self.leverage)

    # We don't want random-seeding for reproducibilityy! We _want_ two runs to give different results, because we only
    # trust the hyper combo which consistently gives positive results.
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.is_stop_loss = False
        self.terminal = False
        self.fee = 0
        acc = self.acc[self.mode]
        if self.mode == 'test':
            if hasattr(acc.step, "i"):
                last_step = acc.step.i
            else:
                last_step = 0
        acc.step.i = 0
        acc.step.cash, acc.step.value = self.start_cash, self.start_value
        # acc.step.hold_value = self.start_value + self.start_cash
        acc.step.totals = Box(
            trade=[self.start_cash + self.start_value],
            hold=[self.start_cash + self.start_value]
        )
        acc.step.signals = []
        if self.mode == 'test':
            # [sfan] TODO: read testset start index and end index from config
            acc.ep.i += 1
            acc.ep.i += last_step
        elif self.mode == 'train':
            # [sfan] randomly chose episode start point
            acc.ep.i = self.np_random.randint(low=0, high=self.data.df.shape[0] - self.hypers.STATE.step_window - 1)

        # self.data.reset_cash_val()
        # self.data.set_cash_val(acc.ep.i, acc.step.i, 0., 0.)
        self.states_, self.raw_states_ = self.get_next_state()
        self.raw_next_states_ = None if self.raw_states_ is None else self.raw_states_.copy()

        return self.states_

    def step(self, action):
        """
        Episode terminating(e.g., terminal=True) conditions:
            1. Finish one trading, e.g., bug once and sell once(according to config)
            2. Reach the stop-loss line
            3. Reach the maximum episode length
            4. When testing, all testing data has been consumed
        """
        acc = self.acc[self.mode]
        totals = acc.step.totals
        act_pct = self.hypers.ACTION.pct_map[str(action)]
        acc.step.signals.append(act_pct)

        fee_rate = self.hypers.REWARD.fee_rate
        """
        fee = {
            Exchange.GDAX: 0.0025,  # https://support.gdax.com/customer/en/portal/articles/2425097-what-are-the-fees-on-gdax-
            Exchange.KRAKEN: 0.0026  # https://www.kraken.com/en-us/help/fees
        }[EXCHANGE]
        """

        if not self.leverage:
            # Perform the trade.
            # "act_pct == 0": take a short position(空仓)
            # "act_pct > 0": buying lone(做多，若已经做多了则持仓)
            # "act_pct < 0": selling short(做空，若已经做空了则持仓)
            # "acc.step.value < 0": means you have already sold short(已经做空)
            # "acc.step.value = 0": means you have taken a short position(空仓)
            # "acc.step.value > 0": means you have already bought long(已经做多)
            if act_pct > 0:
                if acc.step.value == 0 and acc.step.cash >= self.stop_loss:
                    # 直接买入多单
                    act_value = act_pct * acc.step.cash
                    self.fee = abs(act_value) * fee_rate
                    acc.step.value += act_value - self.fee
                    acc.step.cash -= act_value
                elif acc.step.value < 0:
                    # 第一步：平掉空单
                    fee = abs(acc.step.value) * fee_rate
                    acc.step.cash += acc.step.value - fee
                    acc.step.value = 0
                    # 第二步：买入多单
                    if acc.step.cash >= self.stop_loss:
                        act_value = act_pct * acc.step.cash
                        fee = abs(act_value) * fee_rate
                        acc.step.value += act_value - fee
                        acc.step.cash -= act_value
            elif act_pct == 0:
                # self.fee = acc.step.value * fee_rate
                # acc.step.cash += acc.step.value - self.fee
                acc.step.cash += acc.step.value
                acc.step.value = 0
                # [sfan] Episode terminating condition 1:
                #   Trade once per episode. When shorting the trade, the episode is terminated.
                if self.hypers.EPISODE.trade_once:
                    self.terminal = True
            else:
                if acc.step.value == 0:
                    # 直接空单
                    # 注意：act_value小于0
                    act_value = act_pct * acc.step.cash
                    acc.step.value += act_value
                    acc.step.cash -= act_value
                elif acc.step.value > 0:
                    # 第一步：平掉多单
                    acc.step.cash += acc.step.value
                    acc.step.value = 0
                    # 第二步：空单
                    act_value = act_pct * acc.step.cash
                    acc.step.value += act_value
                    acc.step.cash -= act_value

        # next delta. [1,2,2].pct_change() == [NaN, 1, 0]
        # pct_change = self.prices_diff[acc.step.i + 1]
        _, y, _ = self.data.get_data(acc.ep.i, acc.step.i)  # TODO verify
        pct_change = y[self.data.target]

        # [sfan]
        hold_before = totals.trade[-1]
        if self.leverage:
            acc.step.hold_value = pct_change + hold_before
            acc.step.value += act_pct * pct_change
        else:
            acc.step.hold_value = pct_change * hold_before
            acc.step.value = pct_change * acc.step.value

        totals.hold.append(acc.step.hold_value)
        total_now = acc.step.value + acc.step.cash
        totals.trade.append(total_now)

        acc.step.i += 1

        """
        self.data.set_cash_val(
            acc.ep.i, acc.step.i,
            acc.step.cash/self.start_cash,
            acc.step.value/self.start_value
        )
        """
        self.raw_states_ = None if self.raw_next_states_ is None else self.raw_next_states_.copy()
        next_state, raw_state = self.get_next_state()
        self.states_ = next_state.copy()
        self.raw_next_states_ = raw_state.copy()

        # [sfan] Episode terminating condition 4:
        if next_state is None:
            self.terminal = True

        # [sfan] Avoid None value of the next state of next state, which causes crash of coach
        _, next_next_state, _ = self.data.get_data(acc.ep.i, acc.step.i + 1)  # TODO verify
        if next_next_state is None:
            self.terminal = True

        # [sfan] Episode terminating condition 2:
        # If reaching the stop loss level, the episode is terminated.
        if self.hypers.EPISODE.force_stop_loss:
            if self.leverage:
                if pct_change * self.data.max_value < self.stop_loss:
                    self.terminal = True
                    self.is_stop_loss = True
            else:
                if total_now < self.stop_loss:
                    """
                    print("**************************")
                    print("Profit is {}".format(totals.trade[-1] * 1.0 / self.start_cash - 1))
                    print("Profit of last time-step is {}".format(totals.trade[-2] * 1.0 / self.start_cash -1))
                    """
                    self.terminal = True
                    self.is_stop_loss = True

        # [sfan] Episode terminating condition 3:
        max_episode_len = self.hypers.EPISODE.max_len
        if self.mode == 'train' and acc.step.i >= max_episode_len:
            self.terminal = True

        """
        if terminal and self.mode in ('train', 'test'):
            # We're done.
            acc.step.signals.append(0)  # Add one last signal (to match length)
        """

        if self.terminal and self.mode in ('live', 'test_live'):
            raise NotImplementedError

        reward = self.get_return()

        # if acc.step.value <= 0 or acc.step.cash <= 0: terminal = 1
        return next_state, reward, self.terminal, {}

    def close(self):
        pass

    def update_btc_price(self):
        self.btc_price = 8000
        # try:
        #     self.btc_price = int(requests.get(f"https://api.cryptowat.ch/markets/{EXCHANGE.value}/btcusd/price").json()['result']['price'])
        # except:
        #     self.btc_price = self.btc_price or 8000

    def xform_data(self, df):
        # TODO here was autoencoder, talib indicators, price-anchoring
        raise NotImplementedError

    def get_next_state(self):
        acc = self.acc[self.mode]
        X, _, raw = self.data.get_data(acc.ep.i, acc.step.i)
        if X is not None:
            return X.values[:, np.newaxis, :], raw  # height, width(nothing), depth
        else:
            return None, None

    def get_return(self):
        acc = self.acc[self.mode]
        totals = acc.step.totals
        action = acc.step.signals[-1]

        if self.hypers.EPISODE.force_stop_loss and self.is_stop_loss:
            if self.leverage:
                reward = max(totals.trade[-1] - totals.trade[-2], self.hypers.EPISODE.stop_loss_fraction * (-1))
            else:
                reward = self.hypers.EPISODE.stop_loss_fraction - 1
        else:
            reward = totals.trade[-1] - totals.trade[-2]

        if self.hypers.EPISODE.trade_once:
            reward += self.hypers.REWARD.extra_reward

        if self.leverage:
            reward *= self.data.max_value

        # reward -= self.fee

        """
        if self.hypers.EPISODE.force_stop_loss and self.is_stop_loss:
            reward = self.hypers.EPISODE.stop_loss_fraction - 1
        else:
            if action:
                reward = (totals.trade[-1] / totals.trade[-2]) - 1
            else:
                reward = (totals.hold[-1] / totals.trade[-2] - 1) * (-1)
                reward -= self.fee

        if self.hypers.EPISODE.trade_once:
            reward += self.hypers.REWARD.extra_reward
        """

        """
        if self.terminal:
            if self.hypers.EPISODE.force_stop_loss and self.is_stop_loss:
                reward = self.hypers.EPISODE.stop_loss_fraction - 1
            else:
                reward = (totals.trade[-1] / totals.trade[0]) - 1

        else:
            if self.hypers.EPISODE.trade_once:
                reward += self.hypers.REWARD.extra_reward
                # Encourage Explore
                if len(totals.trade) > 1:
                    reward += (totals.trade[-1] / totals.trade[-2] - 1) * action
                else:
                    reward += (totals.trade[-1] / (self.start_cash + self.start_value) - 1) * action
            else:
                reward = 0
        """
        """
        # [sfan] if action is empty position(=0), the reward is calculated over holding
        if len(totals.trade) > 1:
            reward = (totals.hold[-1] / totals.trade[-2] - 1) * (-1)
        else:
            reward = (totals.hold[-1] / (self.start_cash + self.start_value) - 1) * (-1)
        """

        return reward
    
    def get_episode_stats(self):
        """
        [sfan] Calculate the episode stats, including:
        * episode profit
        * action stats
        """
        acc = self.acc[self.mode]
        totals = acc.step.totals
        signals = np.array(acc.step.signals)
        profit = totals.trade[-1] / totals.trade[0] - 1
        if self.is_stop_loss and self.hypers.EPISODE.force_stop_loss:
            profit = self.hypers.EPISODE.stop_loss_fraction - 1

        eq_0 = (signals == 0).sum()
        gt_0 = (signals > 0).sum()
        lt_0 = (signals < 0).sum()

        stats = {
            "profit": profit,
            "stop_loss": self.is_stop_loss,
            "episode_len": acc.step.i,
            "action": {
                "0": eq_0,
                "1": gt_0,
                "2": lt_0
            }
        }

        return stats

    def render(self, mode='human'):
        import matplotlib.pyplot as plt
        from matplotlib.finance import candlestick_ohlc
        import matplotlib.dates as mdates

        if self.raw_next_states_ is None:
            return None

        df_history = self.raw_states_.copy()
        columns = ['timestamp', 'open', 'high', 'low', 'close']

        df_history['timestamp'] = df_history['timestamp'].map(lambda x: mdates.date2num(x))
        df_history['ii'] = range(len(df_history))
        ohlc = df_history['ii'].map(lambda x: tuple(df_history.iloc[x][columns])).tolist()
        weekday_ohlc = [tuple([i]+list(item[1:])) for i, item in enumerate(ohlc)]

        if not self.ohlc_history:
            self.ohlc_history = weekday_ohlc
            self.action_history = [0] * len(weekday_ohlc)
        else:
            self.ohlc_history.pop(0)
            self.ohlc_history.append(weekday_ohlc[-1])
            signals = self.acc[self.mode].step.signals
            if signals:
                self.action_history.pop(0)
                self.action_history.append(signals[-1])

        action_ohlc = []
        for i, item in enumerate(self.action_history):
            if item > 0:
                tuple_item = (i, 0, 1, 0, 1)
            elif item == 0:
                tuple_item = (i, 0, 0, 0, 0)
            else:
                tuple_item = (i, 0, -1, 0, -1)

            action_ohlc.append(tuple_item)
        # action_ohlc = [tuple([i] + []) for i, item in enumerate(self.action_history)]

        plt.ion()
        if self.fig is None:
            self.fig = plt.figure(figsize=(len(weekday_ohlc)*10.0/46, 7))
        plt.clf()
        # plt.cla()
        ax = plt.subplot(2, 1, 1)
        candlestick_ohlc(ax, weekday_ohlc, width=0.6, colorup='r', colordown='g')
        ax.set_xticks(range(0, len(weekday_ohlc), 1))  # 每小时标一个日期
        ax.set_xticklabels([mdates.num2date(ohlc[index][0]).strftime('%Y-%m-%d') for index in ax.get_xticks()])
        # ax.legend(['price'], frameon=False)
        plt.xlim(0, len(weekday_ohlc) - 1)  # 设置一下x轴的范围
        plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')  # 将x轴的label转一下，好看一点

        ax = plt.subplot(2, 1, 2)
        candlestick_ohlc(ax, action_ohlc, width=0.6, colorup='r', colordown='g')
        ax.set_xticks(range(0, len(action_ohlc), 1))  # 每小时标一个日期
        # ax.set_xticklabels([mdates.num2date(ohlc[index][0]).strftime('%Y-%m-%d') for index in ax.get_xticks()])
        # ax.legend(['aciton'], frameon=False)
        plt.xlim(0, len(action_ohlc) - 1)  # 设置一下x轴的范围
        plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')  # 将x轴的label转一下，好看一点

        plt.pause(0.0001)

        return None



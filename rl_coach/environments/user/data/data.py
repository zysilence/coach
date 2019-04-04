import json
from os import path
from enum import Enum

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

# From connecting source file, `import engine` and run `engine.connect()`. Need each connection to be separate
# (see https://stackoverflow.com/questions/3724900/python-ssl-problem-with-multiprocessing)
config_json = json.load(open(path.dirname(__file__) + '/../config/btc.json'))
DB = config_json['DB_HISTORY'].split('/')[-1]
engine_runs = create_engine(config_json['DB_RUNS'])

# Decide which exchange you want to trade on (significant even in training). Pros & cons; Kraken's API provides more
# details than GDAX (bid/ask spread, VWAP, etc) which means predicting its next price-action is easier for RL. It
# also has a lower minimum trade (.002 BTC vs GDAX's .01 BTC), which gives it more wiggle room. However, its API is
# very unstable and slow, so when you actually go live you'r bot will be suffering. GDAX's API is rock-solid. Look
# into the API stability, it may change by the time you're using this. If Kraken is solid, use it instead.
class Exchange(Enum):
    GDAX = 'gdax'
    KRAKEN = 'kraken'
EXCHANGE = Exchange.KRAKEN

# see {last_good_commit} for imputes (ffill, bfill, zero),
# alex database


class Data(object):
    def __init__(self, window=300, indicators={}, mode='train', leverage=False):
        self.window = window
        self.indicators = indicators
        self.mode = mode
        self.leverage = leverage

        # self.ep_stride = ep_len  # disjoint
        # self.ep_stride = 100  # overlap; shift each episode by x seconds.
        # TODO overlapping stride would cause test/train overlap. Tweak it so train can overlap data, but test gets silo'd

        col_renames = {
            'Timestamp': 'timestamp',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
        }
        self.target = "close"

        filename = config_json['DATA']['file_name']

        self.df = pd.read_csv(path.join(path.dirname(__file__), 'populate', 'bitcoin-historical-data', filename))
        self.df = self.df.rename(columns=col_renames)
        ts = "timestamp"
        self.df[ts] = pd.to_datetime(self.df[ts], unit='s')
        self.raw_data = self.df.copy()
        # self.raw_data = self.raw_data.rename(columns={ts: 'date'})
        self.df = self.df.set_index(ts)
        self.raw_data = self.raw_data.set_index(ts, drop=False)

        # [sfan] Select features
        features = config_json['DATA']['features']
        self.df = self.df[features]
        features.append('timestamp')
        self.raw_data = self.raw_data[features]

        # too quiet before 2015, time waste. copy() to avoid pandas errors
        # [sfan] start year is read from the config file
        # df = df.loc['2015':].copy()
        if 'train' == self.mode:
            start_date = config_json['DATA']['train_start_date']
            end_date = config_json['DATA']['train_end_date']
        else:
            start_date = config_json['DATA']['test_start_date']
            end_date = config_json['DATA']['test_end_date']
        self.df = self.df.loc[start_date:end_date].copy()
        self.raw_data = self.raw_data.loc[start_date:end_date].copy()

        # [sfan] fill nan
        self.df = self.df.replace([np.inf, -np.inf], np.nan).ffill()  # .bfill()?
        self.raw_data = self.raw_data.replace([np.inf, -np.inf], np.nan).ffill()  # .bfill()?

        # [sfan] Use scale or not?
        self.max_value = self.df.max().max()
        self.min_value = self.df.min().min()
        if self.leverage:
            # self.df = (self.df - min_value) / (max_value - min_value)
            self.df = self.df / self.max_value
            print("=============== Data Info ================")
            print("Max value: {}; Min value: {}".format(self.max_value, self.min_value))
        """
        df = pd.DataFrame(
            robust_scale(df.values, quantile_range=(.1, 100-.1)),
            columns=df.columns, index=df.index
        )
        """

        """
        df['month'] = df.index.month
        df['day'] = df.index.day
        df['hour'] = df.index.hour
        """

        # TODO drop null rows? (inner join?)
        # TODO arbitrage
        # TODO indicators

        """
        diff_cols = [
            f"{table}_{k}" for k in
            'open high low close volume_btc volume vwap'.split(' ')
            for table in filenames.keys()
        ]
        df[diff_cols] = df[diff_cols].pct_change()\
            .replace([np.inf, -np.inf], np.nan)\
            .ffill()  # .bfill()?
        df = df.iloc[1:]
        target = df[self.target]  # don't scale price changes; we use that in raw form later
        df = pd.DataFrame(
            robust_scale(df.values, quantile_range=(.1, 100-.1)),
            columns=df.columns, index=df.index
        )
        df[self.target] = target

        # [sfan] 'cash' and 'value' features are filled in every timestep with default value 0
        df['cash'], df['value'] = 0., 0.
        """

    def offset(self, ep_start, step):
        return ep_start + step

    def has_more(self, ep_start, step):
        return self.offset(ep_start, step) + self.window < self.df.shape[0]
        # return (ep + 1) * self.ep_stride + self.window < self.df.shape[0]

    def get_data(self, ep_start, step):
        offset = self.offset(ep_start, step)
        try:
            X = self.df.iloc[offset:offset+self.window]
            y = self.df.iloc[offset+self.window]
            raw = self.raw_data[offset:offset+self.window]
            base = X.iloc[-1]['close']
            # [sfan] normalized by close price of the last timestep in the window
            if not self.leverage:
                X = X / base
                y = y / base
            else:
                y -= base
        except IndexError:
            X = None
            y = None
            raw = None

        return X, y, raw


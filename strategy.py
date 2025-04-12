import pandas as pd
import numpy as np
from scipy.stats import norm

class StatisticalArbitrageStrategy:
    def __init__(self, data1, data2, lookback_window=30, z_score_threshold=2.0):
        if data1 is None or data2 is None or data1.empty or data2.empty:
            raise ValueError("Input data cannot be None or empty")
            
        self.data1 = data1
        self.data2 = data2
        self.lookback_window = lookback_window
        self.z_score_threshold = z_score_threshold
        
        common_index = data1.index.intersection(data2.index)
        self.price_df = pd.DataFrame({
            'Asset_A': data1.loc[common_index, 'Close'],
            'Asset_B': data2.loc[common_index, 'Close']
        })
        
        self.calculate_log_metrics()
        self.generate_signals()
        self.calculate_probabilities()

    def calculate_log_metrics(self):
        self.price_df['Log_Return_A'] = np.log(self.price_df['Asset_A'] / self.price_df['Asset_A'].shift(1))
        self.price_df['Log_Return_B'] = np.log(self.price_df['Asset_B'] / self.price_df['Asset_B'].shift(1))
        self.price_df['Log_Spread'] = self.price_df['Log_Return_A'] - self.price_df['Log_Return_B']
        self.price_df['Spread_Mean'] = self.price_df['Log_Spread'].rolling(window=self.lookback_window).mean()
        self.price_df['Spread_Std'] = self.price_df['Log_Spread'].rolling(window=self.lookback_window).std()
        self.price_df['Z-Score'] = (self.price_df['Log_Spread'] - self.price_df['Spread_Mean']) / self.price_df['Spread_Std']
        self.price_df = self.price_df.dropna()

    def generate_signals(self):
        self.price_df['Signal'] = np.where(
            self.price_df['Z-Score'] > self.z_score_threshold, 
            'Short_A_Long_B',
            np.where(
                self.price_df['Z-Score'] < -self.z_score_threshold,
                'Long_A_Short_B', 
                'Hold'
            )
        )
        self.price_df['Signal_Change'] = self.price_df['Signal'] != self.price_df['Signal'].shift(1)

    def calculate_probabilities(self):
        self.price_df['Probability'] = np.where(
            self.price_df['Signal'] == 'Long_A_Short_B',
            norm.cdf(self.price_df['Z-Score']),
            np.where(
                self.price_df['Signal'] == 'Short_A_Long_B',
                1 - norm.cdf(self.price_df['Z-Score']),
                0
            )
        )
        self.price_df['Prob_Mean_Reversion'] = np.where(
            self.price_df['Z-Score'] > 0,
            1 - norm.cdf(self.price_df['Z-Score']),
            norm.cdf(self.price_df['Z-Score'])
        )

    def get_trade_recommendations(self):
        current_data = self.price_df.iloc[-1]
        
        if current_data['Signal'] == 'Hold':
            return {
                'recommendation': 'No trade',
                'probability': 0,
                'z_score': current_data['Z-Score']
            }
        
        return {
            'recommendation': current_data['Signal'],
            'probability': current_data['Probability'],
            'z_score': current_data['Z-Score'],
            'prob_mean_reversion': current_data['Prob_Mean_Reversion']
        }

    def get_strategy_metrics(self):
        signals = self.price_df[self.price_df['Signal_Change'] & (self.price_df['Signal'] != 'Hold')]
        
        if len(signals) == 0:
            return {
                'total_trades': 0,
                'success_rate': 0,
                'avg_probability': 0
            }
        
        signals['Success'] = np.where(
            ((signals['Signal'] == 'Long_A_Short_B') & (signals['Z-Score'].shift(-1) > signals['Z-Score'])) |
            ((signals['Signal'] == 'Short_A_Long_B') & (signals['Z-Score'].shift(-1) < signals['Z-Score'])),
            1, 0
        )
        
        success_rate = signals['Success'].mean()
        avg_probability = signals['Probability'].mean()
        
        return {
            'total_trades': len(signals),
            'success_rate': success_rate,
            'avg_probability': avg_probability
        }
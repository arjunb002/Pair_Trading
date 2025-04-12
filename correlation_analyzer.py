import pandas as pd

class CorrelationAnalyzer:
    
    def calculate_correlation(self, stock1_data, stock2_data, price_column='Close'):
        if price_column not in stock1_data.columns or price_column not in stock2_data.columns:
            raise ValueError(f"Price column '{price_column}' not found in both dataframes")
            
        aligned_data = pd.merge(
            stock1_data[[price_column]], 
            stock2_data[[price_column]], 
            left_index=True, 
            right_index=True,
            how='inner'
        )
        
        if len(aligned_data) < 2:
            raise ValueError("Not enough overlapping data points to calculate correlation. Choose stocks from same index.")
            
        correlation = aligned_data.corr().iloc[0, 1]
        return correlation
    
    def assess_trading_readiness(self, correlation):
        abs_corr = abs(correlation)
        
        if abs_corr >= 0.8:
            return ("High", "Ready to trade - Strong correlation")
        elif abs_corr >= 0.5:
            return ("Medium", "Can find better trade - Moderate correlation")
        else:
            return ("Low", "Search for better pairs - Weak correlation")
    
    def analyze_pair(self, stock1_data, stock2_data):
        try:
            correlation = self.calculate_correlation(stock1_data, stock2_data)
            assessment, recommendation = self.assess_trading_readiness(correlation)
            
            return {
                'correlation': correlation,
                'assessment': assessment,
                'recommendation': recommendation,
                'status': 'success'
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }
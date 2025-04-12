import matplotlib
matplotlib.use('Agg')  # Must come first
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
from data_fetcher import DataFetcher
from strategy import StatisticalArbitrageStrategy
from correlation_analyzer import CorrelationAnalyzer

st.set_page_config(
    page_title="Pair Trading Dashboard",
    layout="wide",
    page_icon="📊"
)

def display_correlation_analysis(corr_result):
    st.subheader("🔗 Correlation Analysis")
    
    if corr_result['status'] == 'error':
        st.error(corr_result['message'])
        return
    
    cols = st.columns(3)
    with cols[0]:
        st.metric("Correlation", f"{corr_result['correlation']:.2f}")
    with cols[1]:
        if corr_result['assessment'] == "High":
            st.metric("Assessment", "High", delta="Strong", delta_color="normal")
        elif corr_result['assessment'] == "Medium":
            st.metric("Assessment", "Medium", delta="Moderate", delta_color="off")
        else:
            st.metric("Assessment", "Low", delta="Weak", delta_color="inverse")
    with cols[2]:
        st.metric("Recommendation", corr_result['recommendation'])

def plot_asset_prices(strategy, stock1_name, stock2_name, raw_data1, raw_data2):
    st.subheader("💰 Asset Prices")
    
    
    tab1, tab2, tab3, tab4 = st.tabs([
        f"{stock1_name} Raw", 
        f"{stock2_name} Raw", 
        "Normalized View",
        "Trading Signals"
    ])
    
    
    with tab1:
        fig1, ax1 = plt.subplots(figsize=(10, 4))
        ax1.plot(raw_data1.index, raw_data1['Close'],label=f"{stock1_name} Price", color='blue')
        ax1.set_title(f"{stock1_name} Raw Price")
        ax1.set_ylabel("Price")
        ax1.legend()
        ax1.grid(True)
        st.pyplot(fig1)
        plt.close(fig1)
    
    
    with tab2:
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.plot(raw_data2.index, raw_data2['Close'],label=f"{stock2_name} Price", color='orange')
        ax2.set_title(f"{stock2_name} Raw Price")
        ax2.set_ylabel("Price")
        ax2.legend()
        ax2.grid(True)
        st.pyplot(fig2)
        plt.close(fig2)
    
    
    with tab3:
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        
        
        norm_A = strategy.price_df['Asset_A'] / strategy.price_df['Asset_A'].iloc[0] * 100
        norm_B = strategy.price_df['Asset_B'] / strategy.price_df['Asset_B'].iloc[0] * 100
        
        ax3.plot(strategy.price_df.index, norm_A, 
                label=f"{stock1_name} (Norm)", color='blue')
        ax3.plot(strategy.price_df.index, norm_B, 
                label=f"{stock2_name} (Norm)", color='orange')
        
       
        signals = strategy.price_df[strategy.price_df['Signal'] != 'Hold']
     
        if not signals.empty:
            buy_signals = signals[signals['Signal'] == 'Long_A_Short_B']
            sell_signals = signals[signals['Signal'] == 'Short_A_Long_B']
            
            ax3.scatter(buy_signals.index,norm_A[buy_signals.index], label='Buy Signal',color='green', marker='^', s=100)
            ax3.scatter(sell_signals.index,norm_A[sell_signals.index], label='Sell Signal',color='red', marker='v', s=100)
        
        ax3.set_title(f"Normalized Prices (Base=100) with Trading Signals")
        ax3.set_ylabel("Normalized Price")
        ax3.legend()
        ax3.grid(True)
        st.pyplot(fig3)
        plt.close(fig3)
    
    
    with tab4:
        fig4, ax4 = plt.subplots(figsize=(10, 2))
        signals = strategy.price_df[strategy.price_df['Signal'] != 'Hold']
        
        if not signals.empty:
            buy_signals = signals[signals['Signal'] == 'Long_A_Short_B']
            sell_signals = signals[signals['Signal'] == 'Short_A_Long_B']
            
            ax4.scatter(buy_signals.index, [1]*len(buy_signals), label='Buy (Long A, Short B)', color='green', marker='^', s=100)
            ax4.scatter(sell_signals.index, [-1]*len(sell_signals), label='Sell (Short A, Long B)', color='red', marker='v', s=100)
        
        ax4.set_title("Trading Signals")
        ax4.set_yticks([-1, 0, 1])
        ax4.set_yticklabels(["Sell", "", "Buy"])
        ax4.set_ylim(-1.5, 1.5)
        ax4.legend(loc='upper right')
        ax4.grid(False)
        st.pyplot(fig4)
        plt.close(fig4)

def plot_spread_analysis(strategy):
    st.subheader("📈 Spread Analysis")
    
    fig, ax = plt.subplots(figsize=(10, 4))
    
    ax.plot(strategy.price_df.index, strategy.price_df['Log_Spread'], label='Log Spread', color='purple', alpha=0.8)
    ax.plot(strategy.price_df.index, strategy.price_df['Spread_Mean'], label='Rolling Mean', color='green', linestyle='--', alpha=0.6)
    
    upper_thresh = strategy.z_score_threshold * strategy.price_df['Spread_Std'].mean()
    lower_thresh = -upper_thresh
    
    ax.axhline(upper_thresh, color='red', linestyle=':', label=f'Upper Threshold (Z={strategy.z_score_threshold})')
    ax.axhline(lower_thresh, color='red', linestyle=':', label=f'Lower Threshold (Z=-{strategy.z_score_threshold})')
    
    ax.set_title("Log Spread with Mean Reversion Thresholds")
    ax.set_ylabel("Log Spread Value")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

def display_strategy_metrics(strategy):
    st.subheader("📊 Strategy Performance")
    metrics = strategy.get_strategy_metrics()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Trades", metrics['total_trades'])
    with col2:
        st.metric("Success Rate", f"{metrics['success_rate']:.1%}")
    with col3:
        st.metric("Z-Score Threshold", f"{strategy.z_score_threshold}")

def main():
    st.title("Pair Trading Dashboard")
    
    with st.sidebar:
        st.header("Parameters")
        stock1 = st.text_input("Stock 1 (e.g., AAPL)", "AAPL").upper()
        stock2 = st.text_input("Stock 2 (e.g., MSFT)", "MSFT").upper()
        lookback = st.slider("Lookback Window (days)", 10, 100, 30)
        z_threshold = st.slider("Z-Score Threshold", 1.0, 3.0, 2.0, step=0.1)
        period = st.selectbox("Data Period", ['1y', '2y', '5y', '10y', 'max'], index=0)
        
        if st.button("Run Analysis", type="primary"):
            st.session_state.run_analysis = True
    
    if 'run_analysis' in st.session_state and st.session_state.run_analysis:
        with st.spinner("Analyzing pair..."):
            try:
                fetcher = DataFetcher()
                analyzer = CorrelationAnalyzer()
                
                
                raw_data1 = fetcher.get_stock_data(stock1, period)
                raw_data2 = fetcher.get_stock_data(stock2, period)
                
               
                aligned_data1 = raw_data1.copy()
                aligned_data2 = raw_data2.copy()
                
                corr_result = analyzer.analyze_pair(aligned_data1, aligned_data2)
                display_correlation_analysis(corr_result)
                
                strategy = StatisticalArbitrageStrategy(
                    aligned_data1, aligned_data2, 
                    lookback_window=lookback,
                    z_score_threshold=z_threshold
                )
                
                tab1, tab2 = st.tabs(["Price Charts", "Spread Analysis"])
                
                with tab1:
                    plot_asset_prices(strategy, stock1, stock2, raw_data1, raw_data2)
                
                with tab2:
                    plot_spread_analysis(strategy)
                
                display_strategy_metrics(strategy)
                
            except Exception as e:
                st.error(f"Error: {str(e)}. There is index problem so choose Stocks from same index.")

st.caption("Made by Arjun Balakrishnan | 2024DMF07 | Madras School Of Economics")
st.caption("This is only for education purposes.")

if __name__ == "__main__":
    main()
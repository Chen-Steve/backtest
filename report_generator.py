import pandas as pd
import io

def generate_report(stats, metrics, data, trades):
    report = f"""
    Backtest Report
    ==================

    Performance Metrics
    -------------------
    Total Return: {metrics['Total Return']}%
    Annualized Return: {metrics['Annualized Return']}%
    Sharpe Ratio: {metrics['Sharpe Ratio']}
    Max Drawdown: {metrics['Max Drawdown']}%

    Optimization Results
    --------------------
    {stats}

    Trade Details
    -------------
    {trades}

    """
    return report

def download_report(report):
    return report.encode('utf-8')

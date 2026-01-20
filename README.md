# Why the Moving Average Strategy Fails and How to Improve It

**Research Question:** Can a simple moving average crossover strategy be made 
profitable through systematic refinement, or is the underlying concept 
fundamentally flawed?

## Summary
Started with basic MA crossover (Sharpe: ...) and systematically improved 
to Sharpe: 0.8 through volatility filtering and regime detection.

## Key Findings


## Repository Structure
[Explain the folders]

## How to Reproduce
```bash
pip install -r requirements.txt
jupyter notebook notebooks/06_final_analysis.ipynb
```

## Technologies
Python 3.10, pandas, backtesting.py, yfinance
```

## Pro Tips

1. **Version control**: Commit after each refinement with clear messages
   - "Add baseline MA crossover strategy"
   - "Refinement v1: Add ATR volatility filter, Sharpe improved from X to Y"

2. **Make it visual**: Every notebook should have charts showing what changed and why

3. **Document failures**: If a refinement makes things worse, keep it! Shows scientific thinking

4. **Comparison tables**: Create a summary table comparing all versions:
```
   Strategy    | Sharpe | Max DD | Win Rate | Annual Return
   Baseline    | -0.3   | 35%    | 45%      | -2%
   V1 (ATR)    |  0.4   | 28%    | 48%      | 5%
   V2 (Regime) |  0.8   | 22%    | 52%      | 8%
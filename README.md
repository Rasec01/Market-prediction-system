# Market Prediction System

Does a random forest find a usable short-term signal in simple price features? This project answers that honestly: forecasts are made walk-forward, the model only ever trains on labels that were known at the time, and the backtest charges transaction costs.

## How it works

```
prices ──► features (returns, moving-average gaps, volatility ratios)
       ──► label: return from today's close to the close 5 days later
       ──► walk-forward random forest (retrained every quarter)
       ──► long/flat signal: hold the asset when the forecast is positive
       ──► backtest on the next day's return, minus costs
```

| Choice | Why |
|---|---|
| Returns and ratios as features, no raw prices | Price levels drift out of the range the trees were trained on; ratios don't |
| Walk-forward, retrained every 63 days | Each forecast comes from a model that never saw that period |
| Five-day embargo | A five-day label needs prices five days ahead, so the last five days before each block are left out of its training set |
| Trade on the next day | The signal is decided at today's close and earns tomorrow's return |
| Costs per trade (default 5 bps) | A daily signal trades often; results before costs mislead |
| Long/flat, not long/short | Easier to compare with buy-and-hold and closer to how a long-only investor would use it |

## Quick start

```bash
pip install -r requirements.txt
python main.py                                    # SPY from 2005
python main.py --ticker QQQ --cost-bps 10 --chart outputs/qqq.png
python main.py --csv data/prices.csv              # your own Date,Close file
```

The script prints the out-of-sample information coefficient (rank correlation between forecast and realised return), the hit rate, and a table comparing the strategy with buy-and-hold: annual return, volatility, Sharpe ratio (risk-free rate set to zero), maximum drawdown, time in the market and trades per year.

## Results

Run `python main.py --chart outputs/spy.png` and paste the table and chart here. Expect a weak signal: on liquid index ETFs, simple price features rarely beat buy-and-hold after costs. Reporting that clearly is part of the point.

## Tests

```bash
pytest -q
```

The key test rewrites all prices after a given date and checks that no forecast made before that date changes, which is a direct check for look-ahead bias. Other tests check the label, that the features don't depend on the price level, and that the backtest trades on the next day and charges costs.

## Project structure

```
├── main.py
├── src/
│   ├── data_loader.py   # yfinance download or CSV
│   ├── features.py      # scale-free features and the forward-return label
│   ├── model.py         # walk-forward random forest, forecast quality
│   └── backtest.py      # long/flat backtest with costs, performance table, chart
└── tests/test_no_lookahead.py
```

## Disclaimer

For research and education only. Not investment advice.

## License

MIT

# ALPHX Quantitative Backend Engine
### High-Performance Cryptocurrency Trading, ML Ensemble & Signal Generation Engine

[![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-009688.svg?style=flat&logo=FastAPI&logoColor=white)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg?style=flat&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C.svg?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-EB5424.svg?style=flat)](https://xgboost.readthedocs.io)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.3%2B-2E8B57.svg?style=flat)](https://lightgbm.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

ALPHX Quantitative Engine is an institutional-grade algorithmic trading backend built in Python with **FastAPI**. It computes multi-timeframe technical indicators, detects algorithmic price action patterns (Support/Resistance, Fair Value Gaps), executes machine learning ensemble predictions (XGBoost, LightGBM, Deep LSTM), manages mathematical risk and dynamic ATR-based trade setups, and delivers high-frequency trade execution on **CoinDCX** (Futures, Margin, Spot) with native dual-currency (**USDT $ & INR ₹**) precision.

---

## 📑 Table of Contents
1. [Architecture Overview](#-architecture-overview)
2. [Mathematical Formulas & Technical Indicators](#-mathematical-formulas--technical-indicators)
3. [Quantitative Signal & ML Ensemble Engine](#-quantitative-signal--ml-ensemble-engine)
4. [Risk Management & Dynamic Position Sizing](#-risk-management--dynamic-position-sizing)
5. [Dual-Currency (USD $ & INR ₹) Engine](#-dual-currency-usd---inr--engine)
6. [Exchange Interoperability (CoinDCX & Paper Trading)](#-exchange-interoperability-coindcx--paper-trading)
7. [Automated Telegram Broadcasting Sentinel](#-automated-telegram-broadcasting-sentinel)
8. [Project Structure](#-project-structure)
9. [Installation & Setup Guide](#-installation--setup-guide)
10. [API Reference](#-api-reference)
11. [License](#-license)

---

## 🏛 Architecture Overview

```
                                    ┌────────────────────────┐
                                    │    Market Data Feeds   │
                                    │ (CoinDCX / WebSockets) │
                                    └───────────┬────────────┘
                                                │ Candles & OrderBook
                                                ▼
                                    ┌────────────────────────┐
                                    │    Indicator Engine    │
                                    │ (EMAs, RSI, MACD, ATR) │
                                    └───────────┬────────────┘
                                                │
                 ┌──────────────────────────────┴──────────────────────────────┐
                 ▼                                                             ▼
    ┌─────────────────────────┐                                   ┌─────────────────────────┐
    │   Price Action Engine   │                                   │   Feature Transformer   │
    │ (S&R, FVG, Pinbar/Doji) │                                   │ (30+ Normalized Feats)  │
    └────────────┬────────────┘                                   └────────────┬────────────┘
                 │                                                             │
                 ▼                                                             ▼
    ┌─────────────────────────┐                                   ┌─────────────────────────┐
    │   6-Factor Quant Scorer │                                   │   ML Ensemble Predictor │
    │ Trend, Mom, Vol, OB, PA │                                   │ XGBoost, LightGBM, LSTM │
    └────────────┬────────────┘                                   └────────────┬────────────┘
                 │                                                             │
                 └──────────────────────────────┬──────────────────────────────┘
                                                │ Consensus & Confidence
                                                ▼
                                    ┌────────────────────────┐
                                    │   Risk Management &    │
                                    │ Dynamic Position Sizer │
                                    └───────────┬────────────┘
                                                │ Dual-Currency Sized Orders
                         ┌──────────────────────┴──────────────────────┐
                         ▼                                             ▼
            ┌─────────────────────────┐                   ┌─────────────────────────┐
            │   CoinDCX Execution     │                   │ Telegram Signal Bot     │
            │ (Futures, Margin, Spot) │                   │ (Broadcasts to Channel) │
            └─────────────────────────┘                   └─────────────────────────┘
```

---

## 📐 Mathematical Formulas & Technical Indicators

The engine computes technical features over candle streams with zero data leakage using vectorized calculations:

### 1. Exponential Moving Averages (EMA)
$$EMA_t = \alpha \cdot P_t + (1 - \alpha) \cdot EMA_{t-1}$$
$$\alpha = \frac{2}{N + 1}$$
* Computed spans: $N \in \{9, 21, 50, 100, 200\}$.
* Bullish Alignment: $EMA_9 > EMA_{21} > EMA_{50} > EMA_{200}$.

### 2. Moving Average Convergence Divergence (MACD)
$$MACD_{\text{line}} = EMA_{12}(Close) - EMA_{26}(Close)$$
$$Signal_{\text{line}} = EMA_9(MACD_{\text{line}})$$
$$Histogram = MACD_{\text{line}} - Signal_{\text{line}}$$

### 3. Relative Strength Index (RSI - Wilder's Smoothing)
$$RS = \frac{EMA_{14}(\text{Upward Price Changes})}{EMA_{14}(\text{Downward Price Changes})}$$
$$RSI = 100 - \left( \frac{100}{1 + RS} \right)$$
* Oversold bounce trigger: $RSI < 32$. Overbought exhaustion trigger: $RSI > 68$.

### 4. Average True Range (ATR) & Volatility
$$\text{True Range (TR)} = \max(High - Low, |High - Close_{t-1}|, |Low - Close_{t-1}|)$$
$$ATR_{14} = \frac{1}{14} \sum_{i=1}^{14} TR_i$$

### 5. Supertrend (10, 3.0)
$$\text{Basic Upper Band} = \frac{High + Low}{2} + 3.0 \cdot ATR_{10}$$
$$\text{Basic Lower Band} = \frac{High + Low}{2} - 3.0 \cdot ATR_{10}$$
Maintains trailing state to filter false breaks in high-volatility regimes.

### 6. Bollinger Bands & Squeeze Momentum
$$\text{Middle Band} = SMA_{20}(Close)$$
$$\text{Upper Band} = SMA_{20}(Close) + 2.0 \cdot \sigma_{20}$$
$$\text{Lower Band} = SMA_{20}(Close) - 2.0 \cdot \sigma_{20}$$
$$\%B = \frac{Close - \text{Lower Band}}{\text{Upper Band} - \text{Lower Band}}$$
* Squeeze Detection: Occurs when Bollinger Bands contract entirely inside Keltner Channels ($KC = SMA_{20} \pm 1.5 \cdot ATR_{14}$), indicating explosive directional expansion is imminent.

### 7. Volume-Weighted Average Price (VWAP) & Flow
$$VWAP = \frac{\sum (Price_{\text{typical}} \cdot Volume)}{\sum Volume}, \quad \text{where } Price_{\text{typical}} = \frac{High + Low + Close}{3}$$
$$CMF_{20} = \frac{\sum_{i=1}^{20} \left[ \frac{(Close_i - Low_i) - (High_i - Close_i)}{High_i - Low_i} \cdot Volume_i \right]}{\sum_{i=1}^{20} Volume_i}$$

### 8. Algorithmic Price Action & Fair Value Gaps (FVG)
* **Bullish FVG**: Identified when $Low_{t} > High_{t-2}$, creating an unfilled liquidity imbalance gap.
* **Bearish FVG**: Identified when $High_{t} < Low_{t-2}$.
* **Support & Resistance**: Pivot detection via relative extrema ($\text{order}=15$), clustered dynamically within a $0.5\%$ price variance tolerance.

---

## 🤖 Quantitative Signal & ML Ensemble Engine

### 1. Composite Multi-Factor Score ($S_{\text{composite}}$)
The rule engine aggregates 6 uncorrelated analytical dimensions into a unified score $S \in [-1.0, 1.0]$:

$$S_{\text{composite}} = 0.30 \cdot S_{\text{trend}} + 0.25 \cdot S_{\text{momentum}} + 0.15 \cdot S_{\text{volume}} + 0.15 \cdot S_{\text{orderbook}} + 0.05 \cdot S_{\text{futures}} + 0.10 \cdot S_{\text{price\_action}}$$

* **Long Directive (BUY)**: $S_{\text{composite}} \ge +0.18$
* **Short Directive (SELL)**: $S_{\text{composite}} \le -0.18$
* **Neutral (HOLD)**: $-0.18 < S_{\text{composite}} < +0.18$

$$\text{Confidence} = \min\left(0.96, \max\left(0.68, 0.62 + 0.38 \cdot |S_{\text{composite}}|\right)\right)$$

### 2. Machine Learning Multi-Model Consensus
The backend deploys an ensemble of 3 distinct architectures:

1. **XGBoost Classifier** ($w_1 = 0.40$): Extreme gradient boosted decision trees capturing non-linear tabular interactions across technical oscillators.
2. **LightGBM Classifier** ($w_2 = 0.30$): Leaf-wise gradient boosting optimized for rapid inference across statistical price distributions.
3. **Deep LSTM Network** ($w_3 = 0.30$): Multi-layer Long Short-Term Memory recurrent neural network processing sequential 30-candle window embeddings.

$$\hat{P}(\text{class}) = 0.40 \cdot P_{\text{xgb}}(\text{class}) + 0.30 \cdot P_{\text{lgb}}(\text{class}) + 0.30 \cdot P_{\text{lstm}}(\text{class})$$

* **UNANIMOUS Agreement**: All 3 models agree on the directional verdict.
* **MAJORITY Agreement**: 2 of 3 models agree.
* **SPLIT**: Conflicting signals trigger automatic hold/risk mitigation.

---

## 🛡 Risk Management & Dynamic Position Sizing

### 1. Fixed Fractional Risk Formula
Position sizing limits capital exposure to a strict percentage (default $2\%$) of total portfolio equity:

$$\text{Risk Capital} = \text{Portfolio Equity} \times 0.02$$
$$\text{Raw Quantity} = \frac{\text{Risk Capital}}{|\text{Entry Price} - \text{Stop Loss}|}$$

### 2. Dynamic ATR Level Architecture
Stop loss and target levels adapt dynamically to real-time market volatility:

| Directive | Stop Loss (SL) | Target 1 (TP1) | Target 2 (TP2) | Target 3 (TP3) |
| :--- | :--- | :--- | :--- | :--- |
| **BUY / LONG** | $\text{Entry} - (1.5 \times ATR)$ | $\text{Entry} + (2.0 \times ATR)$ | $\text{Entry} + (3.0 \times ATR)$ | $\text{Entry} + (4.5 \times ATR)$ |
| **SELL / SHORT**| $\text{Entry} + (1.5 \times ATR)$ | $\text{Entry} - (2.0 \times ATR)$ | $\text{Entry} - (3.0 \times ATR)$ | $\text{Entry} - (4.5 \times ATR)$ |
| **Risk:Reward**| **1.0x (Baseline)** | **1:1.33** | **1:2.0** | **1:3.0** |

### 3. Exchange Lot Quantization
To prevent `"Invalid quantity"` rejections on live exchanges, quantities are strictly step-quantized:

$$Q = \max\left(Q_{\min}, \text{round}\left(\frac{Q_{\text{raw}}}{Step}\right) \times Step\right)$$

* **BTC**: Min $0.001$, Step $0.001$ (3 decimal places)
* **ETH**: Min $0.01$, Step $0.01$ (2 decimal places)
* **SOL / BNB**: Min $0.1$, Step $0.1$ (1 decimal place)
* **DOGE / XRP / ADA / TRX / MATIC**: Min $1.0$, Step $1.0$ (0 decimal places)
* **PEPE / SHIB / BONK**: Min $10000.0$, Step $1000.0$ (0 decimal places)

---

## 💱 Dual-Currency (USD $ & INR ₹) Engine

To support Indian cryptocurrency traders seamlessly, the engine natively handles cross-currency conversions:

* **Live Exchange Rate Provider**: Continuously synchronizes USD/INR rates (fallback $99.95$).
* **Unified Capital Aggregation**:
  $$Capital_{\text{USDT}} = Balance_{\text{USDT}} + \left(\frac{Balance_{\text{INR}}}{Rate_{\text{USD/INR}}}\right)$$
  $$Capital_{\text{INR}} = \left(Balance_{\text{USDT}} \times Rate_{\text{USD/INR}}\right) + Balance_{\text{INR}}$$
* **Zero-Balance Safety**: Automatically provisions paper test capital ($1,000 USDT / ₹1,00,000 INR) so unverified or test accounts never trigger execution crashes.

---

## ⚡ Exchange Interoperability (CoinDCX & Paper Trading)

The adapter layer standardizes all interactions across spot, margin, and derivatives:
* **CoinDCX Futures**: Authenticated HMAC-SHA256 signature signing with millisecond epoch timestamps, targeting `/exchange/v1/derivatives/futures/orders/create`.
* **CoinDCX Spot & Margin**: Direct market creation (`BTCUSDT`), automated leverage application, and position tracking.
* **Paper Trading Sandbox**: High-fidelity local order matching with real-time balance deductions, PnL tracking, and liquidation price estimation.

---

## 📢 Automated Telegram Broadcasting Sentinel

The integrated Telegram service ([@alphx_signal_bot](https://t.me/alphx_signal_bot)) provides automated and manual signal delivery:

1. **Auto-Broadcast Engine**: Automatically pushes institutional trade signals when confidence exceeds the threshold ($\ge 70\%$).
2. **Anti-Flood Deduplication**: Enforces a 3-minute cooldown window per asset to prevent spamming channels during consolidation.
3. **Interactive Telegram Buttons**: Includes deep-link inline action buttons for 1-click execution.

---

## 📁 Project Structure

```text
AI_singal_Backend/
├── app/
│   ├── api/
│   │   ├── routes/
│   │   │   ├── analyze.py         # Technical & ML analysis route
│   │   │   ├── trade.py           # Order execution & dual-currency risk sizing
│   │   │   ├── markets.py         # CoinDCX screener & market details
│   │   │   ├── telegram.py        # Telegram bot auto-detection & dispatch
│   │   │   ├── webhook.py         # TradingView & CoinDCX inbound webhook listener
│   │   │   └── ws.py              # WebSocket streaming gateway
│   │   └── schemas.py             # Pydantic data schemas & request validators
│   ├── core/
│   │   ├── indicator_engine.py    # Vectorized technical indicator library
│   │   ├── signal_generator.py    # 6-dimension quantitative scoring engine
│   │   ├── risk_manager.py        # Position sizing & dynamic ATR levels
│   │   ├── price_action.py        # Support/Resistance, FVG, & candlestick patterns
│   │   ├── currency.py            # USD/INR real-time currency conversion
│   │   └── telegram_service.py    # Async Telegram Bot broadcast service
│   ├── exchanges/
│   │   ├── base.py                # Abstract exchange interface & data classes
│   │   └── coindcx/
│   │       ├── adapter.py         # Main CoinDCX unified exchange adapter
│   │       ├── auth.py            # HMAC-SHA256 request signer
│   │       ├── constants.py       # CoinDCX API endpoint mapping
│   │       ├── futures.py         # Derivatives futures order executor
│   │       ├── spot.py            # Spot market trading operations
│   │       ├── margin.py          # Margin trading operations
│   │       ├── market_data.py     # Ticker, candles, orderbook, & trade cache
│   │       └── websocket.py       # Live stream client
│   ├── ml/
│   │   ├── ensemble.py            # XGBoost + LightGBM + LSTM consensus
│   │   ├── feature_builder.py     # 30+ numerical feature transformer
│   │   ├── models/                # Model wrappers
│   │   └── pretrained/            # Model weights and pipeline configurations
│   ├── config.py                  # Global settings via Pydantic BaseSettings
│   └── main.py                    # FastAPI application entrypoint & middleware
├── .env.example                   # Environment variable template
├── requirements.txt               # Production Python dependencies
└── README.md
```

---

## 🚀 Installation & Setup Guide

### 1. Prerequisites
* **Python 3.12+**
* Virtual environment tool (`venv` or `uv`)
* CoinDCX API Key & Secret (optional for Paper Trading mode)
* Telegram Bot Token from [@BotFather](https://t.me/BotFather) (for Telegram broadcasting)

### 2. Clone Repository
```bash
git clone https://github.com/your-username/AI_singal_Backend.git
cd AI_singal_Backend
```

### 3. Setup Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 4. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Configure Environment Variables
Copy the template and fill in your details:
```bash
cp .env.example .env
```

Edit `.env`:
```env
DEBUG=true
PORT=8000

# CoinDCX API Credentials (Leave as 'demo' for Paper Trading)
COINDCX_API_KEY=your_coindcx_api_key_here
COINDCX_API_SECRET=your_coindcx_api_secret_here

# Risk Parameters
DEFAULT_LEVERAGE=3.0
MAX_RISK_PER_TRADE=0.02
MIN_CONFIDENCE=0.70

# Telegram Bot Integration
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
TELEGRAM_CHAT_ID=your_channel_or_group_id_here
TELEGRAM_AUTO_SEND=false
TELEGRAM_MIN_CONFIDENCE=0.70

# Inbound Webhook Security
WEBHOOK_SECRET_KEY=your_secure_webhook_secret_key_here
```

### 6. Start the Backend Server
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Once started, interactive API documentation is available at:
* **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
* **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## 🔌 API Reference

### Market Analysis
* `GET /api/analyze?symbol=B-BTC_USDT&timeframe=15m`
  * Computes technical indicators, price action setups, ML ensemble predictions, and ATR levels.

### Order Execution
* `POST /api/trade/execute`
  * Executes a trade with automated risk management or custom capital in USD or INR.
* `POST /api/trade/close`
  * Closes an active position on CoinDCX or Paper Trading.

### Exchange Telemetry
* `GET /api/exchange/balances?exchange=coindcx`
  * Retrieves live balances across USDT, INR, and crypto assets.
* `GET /api/exchange/positions?exchange=coindcx`
  * Returns active positions with mark price, liquidation price, and unrealized PnL.

### Telegram Automation
* `GET /api/telegram/detect`
  * Discovers recently joined Telegram channels and groups.
* `POST /api/telegram/send`
  * Manually triggers a broadcast of the current signal.
* `POST /api/telegram/test`
  * Sends an instant connectivity ping to the configured channel.

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

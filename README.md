# 🧠 AI-Powered Natural Language Backtester

> Type a trading strategy in plain English — get instant AI-powered backtested results with a professional research note.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-Interactive_Charts-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-00d4aa?style=for-the-badge)

---

## ✨ What Is This?

A **Streamlit web app** that lets you describe trading strategies in **plain English** and instantly runs backtests on historical stock data — powered by **AI**.

### Two Levels of AI

| Level | What It Does |
|-------|-------------|
| **🤖 Level 1 — AI Interpreter** | Understands your natural language strategy (even vague ones like *"buy when the stock looks oversold"*) and generates executable Python/vectorbt backtest code |
| **🧪 Level 2 — AI Analyst** | After the backtest runs, analyzes the results and writes a professional research note with risk analysis, market context, and improvement suggestions |

### Dual Mode
- **AI Mode** (default) — AI interprets any strategy, no matter how complex or ambiguous
- **Rule-based Fallback** — If AI fails, falls back to a regex-based parser for standard strategies

**Example input:**
```
Buy when stock looks oversold and volume is spiking, sell when RSI goes above 70
```

**What you get:**
- 📊 Key metrics (Total Return, Sharpe Ratio, Max Drawdown, Win Rate, Alpha)
- 📈 Interactive Plotly chart with indicators + buy/sell markers
- 📝 Detailed trade log with entry/exit prices
- 💻 Auto-generated Python backtest code you can copy & learn from
- 🧪 AI-written research note analyzing your strategy's performance

---

## 🖥️ Screenshots

<p align="center">
  <img src="assets/home.png" alt="Home Screen" width="90%"/>
  <br><em>Premium dark UI with strategy input and sidebar settings</em>
</p>

<p align="center">
  <img src="assets/results.png" alt="Backtest Results" width="90%"/>
  <br><em>Interactive chart with indicators and buy/sell signals</em>
</p>

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/ai-backtester.git
cd ai-backtester
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up your API key
Create a `.env` file in the project root:
```env
AI_API_KEY=your_api_key_here
```
Get a free API key from [Google AI Studio](https://aistudio.google.com).

### 4. Run the app
```bash
python -m streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

---

## 💬 Supported Strategies

### AI Mode (handles anything)
Just type naturally — the AI can interpret complex, ambiguous strategies:

| Strategy Type | Example Input |
|--------------|---------------|
| **Momentum** | `Buy when stock shows strong upward momentum with increasing volume` |
| **Mean Reversion** | `Buy when stock looks oversold and RSI divergence appears` |
| **Trend Following** | `Use a moving average crossover strategy with 50 and 200 day periods` |
| **Volatility** | `Trade when the stock is 2 standard deviations from the 20-day mean` |
| **Multi-indicator** | `Buy when MACD crosses up, RSI is below 40, and price is above EMA 200` |

### Rule-based Fallback (standard patterns)

| Strategy | Input |
|----------|-------|
| **Golden Cross** | `Buy when SMA 50 crosses above SMA 200, sell when SMA 50 crosses below SMA 200` |
| **RSI Mean Reversion** | `Buy when RSI drops below 30, sell when RSI goes above 70` |
| **EMA Crossover** | `Buy when EMA 20 crosses above EMA 50, sell when EMA 20 crosses below EMA 50` |
| **MACD Signal** | `Buy when MACD line crosses above MACD signal, sell when MACD crosses below signal` |
| **Bollinger Breakout** | `Buy when price crosses above upper Bollinger Band, sell when price drops below lower band` |

### Supported Indicators
- **SMA** (Simple Moving Average) — any period
- **EMA** (Exponential Moving Average) — any period
- **RSI** (Relative Strength Index) — default 14-period
- **MACD** (Line, Signal, Histogram) — default 12/26/9
- **Bollinger Bands** (Upper, Lower, Middle) — default 20-period, 2σ
- **Price** (close/closing price)
- **Volume** (via AI mode)
- **Custom combinations** (via AI mode)

---

## 🏗️ Architecture

```
User Input                     "Buy when stock looks oversold"
    │
    ▼
┌──────────────────┐
│  ai_interpreter  │           Natural language → AI → Python/vectorbt code
│  (Level 1 — AI)  │           Handles ambiguous & complex strategies
├──────────────────┤
│  parser.py       │           Fallback: regex-based NLP parser
│  code_generator  │           Template-based code generation
└──────┬───────────┘
       │
       ▼
┌──────────────┐
│ backtester.py│               exec() in sandboxed namespace
│  Execution   │               Returns metrics + chart data
└──────┬───────┘
       │
       ▼
┌──────────────────┐
│   ai_analyst     │           Level 2: Metrics → AI → Research Note
│  (Level 2 — AI)  │           Performance analysis + suggestions
├──────────────────┤
│   app.py         │           Streamlit dashboard
│  Dashboard       │           Metric cards + Plotly charts + Trade log
└──────────────────┘
```

### File Structure
```
ai-backtester/
├── app.py                  # Main Streamlit app (entry point)
├── ai_client.py            # AI API wrapper (key management, error handling)
├── ai_interpreter.py       # Level 1: NL strategy → executable code
├── ai_analyst.py           # Level 2: Backtest results → research note
├── parser.py               # Fallback NL strategy parser (rule-based)
├── code_generator.py       # Fallback vectorbt code generator
├── backtester.py           # Executes generated code safely
├── .env                    # API key (git-ignored, you create this)
├── requirements.txt        # Python dependencies
├── .gitignore
├── .streamlit/
│   └── config.toml         # Streamlit theme configuration
├── assets/                 # Screenshots for README
├── LICENSE
└── README.md
```

---

## ⚙️ Configuration

### Sidebar Settings (in the app)
| Setting | Default | Description |
|---------|---------|-------------|
| **Ticker** | `^NSEI` | Any Yahoo Finance ticker (RELIANCE.NS, AAPL, SPY, etc.) |
| **Start Date** | 2019-01-01 | Backtest start date |
| **End Date** | 2024-01-01 | Backtest end date |
| **Initial Capital** | ₹10,00,000 | Starting portfolio value |
| **Transaction Fees** | 0.10% | Per-trade cost (brokerage + STT) |

### Environment Variables
| Variable | Required | Description |
|----------|----------|-------------|
| `AI_API_KEY` | Yes | Your API key (set in `.env` file) |

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Frontend** | Streamlit + Custom CSS |
| **Charts** | Plotly (interactive, dark theme) |
| **AI Engine** | Google Generative AI (2.5 Flash) |
| **Backtesting** | vectorbt |
| **Data** | yfinance (Yahoo Finance API) |
| **NLP Fallback** | Custom rule-based (regex + tokenizer) |
| **Language** | Python 3.10+ |

---

## 📈 Portfolio Context

This project was built as part of a **quantitative finance portfolio** that includes:

1. **Dual Moving Average Crossover** — SMA 50/200 golden cross strategy
2. **RSI Mean Reversion** — RSI(14) oversold/overbought signals
3. **Pairs Trading** — HDFCBANK.NS vs ICICIBANK.NS statistical arbitrage
4. **AI NL Backtester** (this project) — AI-powered natural language strategy engine

---

## 🤝 Contributing

Contributions are welcome! Here's how:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-indicator`)
3. Commit your changes (`git commit -m 'Add Stochastic Oscillator support'`)
4. Push to the branch (`git push origin feature/new-indicator`)
5. Open a Pull Request

### Ideas for Contribution
- [ ] Add more indicators (Stochastic, ADX, ATR, VWAP)
- [ ] Add stop-loss / take-profit parsing
- [ ] Multi-asset portfolio backtesting
- [ ] Strategy comparison mode
- [ ] Deploy to Streamlit Cloud

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file.

---

<p align="center">
  Built with ❤️ by <b>Vismay</b> · 2025
  <br>
  <sub>Powered by AI · Python · Streamlit · vectorbt · Plotly</sub>
</p>

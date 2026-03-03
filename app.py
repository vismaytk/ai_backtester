"""
═══════════════════════════════════════════════════════════════════════════════
  🧠 AI-Powered Natural Language Backtester
  ─────────────────────────────────────────────────────────────────────────
  Type a trading strategy in plain English. Get instant backtests.

  Run:  streamlit run app.py
═══════════════════════════════════════════════════════════════════════════════
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import sys
import os

# Add parent dir to path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from parser import parse_strategy, IndicatorType
from code_generator import generate_backtest_code
from backtester import run_backtest, BacktestError

# AI modules (Level 1 + Level 2)
from ai_client import init_ai, AIError
from ai_interpreter import generate_strategy_code, retry_with_error
from ai_analyst import generate_research_note


# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG & STYLING
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="AI Backtester — NL Strategy Engine",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for premium dark theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Global */
    .stApp {
        background: linear-gradient(135deg, #0a0a1a 0%, #0d1117 50%, #0a0a1a 100%);
        font-family: 'Inter', sans-serif;
    }

    /* Header */
    .main-header {
        background: linear-gradient(135deg, rgba(0, 212, 170, 0.08) 0%, rgba(139, 139, 255, 0.08) 100%);
        border: 1px solid rgba(0, 212, 170, 0.2);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .main-header h1 {
        color: #ffffff;
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
        background: linear-gradient(135deg, #00d4aa, #8b8bff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .main-header p {
        color: #8b949e;
        font-size: 1rem;
        margin: 0.5rem 0 0 0;
    }

    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(22, 27, 44, 0.9) 0%, rgba(13, 17, 23, 0.95) 100%);
        border: 1px solid rgba(0, 212, 170, 0.15);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        border-color: rgba(0, 212, 170, 0.4);
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 212, 170, 0.1);
    }
    .metric-label {
        color: #8b949e;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.3rem;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }
    .metric-good { color: #00d4aa; }
    .metric-bad { color: #ff6b6b; }
    .metric-neutral { color: #8b8bff; }

    /* Strategy Input */
    .stTextArea textarea {
        background: rgba(22, 27, 44, 0.8) !important;
        border: 1px solid rgba(0, 212, 170, 0.2) !important;
        border-radius: 12px !important;
        color: #e6edf3 !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 1rem !important;
        padding: 1rem !important;
    }
    .stTextArea textarea:focus {
        border-color: rgba(0, 212, 170, 0.5) !important;
        box-shadow: 0 0 20px rgba(0, 212, 170, 0.1) !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #00d4aa 0%, #00b894 100%) !important;
        color: #0a0a1a !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        padding: 0.75rem 2rem !important;
        letter-spacing: 0.5px !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(0, 212, 170, 0.3) !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #0a0a1a 100%);
        border-right: 1px solid rgba(0, 212, 170, 0.1);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(22, 27, 44, 0.5);
        border-radius: 8px;
        border: 1px solid rgba(0, 212, 170, 0.1);
        color: #8b949e;
        font-weight: 500;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: rgba(0, 212, 170, 0.1);
        border-color: rgba(0, 212, 170, 0.3);
        color: #00d4aa;
    }

    /* Code blocks */
    .stCodeBlock {
        border-radius: 12px !important;
    }

    /* Parsed strategy box */
    .parse-box {
        background: rgba(22, 27, 44, 0.7);
        border: 1px solid rgba(139, 139, 255, 0.2);
        border-radius: 10px;
        padding: 1rem 1.5rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        color: #e6edf3;
        margin: 0.5rem 0;
    }

    /* Example cards */
    .example-card {
        background: rgba(22, 27, 44, 0.6);
        border: 1px solid rgba(139, 139, 255, 0.15);
        border-radius: 10px;
        padding: 0.8rem 1.2rem;
        margin: 0.3rem 0;
        cursor: pointer;
        transition: all 0.2s ease;
        font-size: 0.85rem;
        color: #c9d1d9;
    }
    .example-card:hover {
        border-color: rgba(0, 212, 170, 0.3);
        background: rgba(0, 212, 170, 0.05);
    }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0a0a1a; }
    ::-webkit-scrollbar-thumb { background: #1f2937; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #374151; }

    /* Hide default Streamlit elements */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# EXAMPLE STRATEGIES
# ══════════════════════════════════════════════════════════════════════════════

EXAMPLE_STRATEGIES = {
    "🔀 Golden Cross (SMA 50/200)": "Buy when SMA 50 crosses above SMA 200, sell when SMA 50 crosses below SMA 200",
    "📊 RSI Mean Reversion": "Buy when RSI drops below 30, sell when RSI goes above 70",
    "📈 EMA Crossover (20/50)": "Buy when EMA 20 crosses above EMA 50, sell when EMA 20 crosses below EMA 50",
    "📉 MACD Signal Crossover": "Buy when MACD line crosses above MACD signal, sell when MACD line crosses below MACD signal",
    "🎯 Bollinger Band Breakout": "Buy when price crosses above upper Bollinger Band, sell when price crosses below lower Bollinger Band",
    "⚡ Fast SMA Crossover (10/30)": "Buy when SMA 10 crosses above SMA 30, sell when SMA 10 crosses below SMA 30",
    "🎲 RSI Extreme Reversal": "Buy when RSI drops below 20, sell when RSI goes above 80",
    "📐 Triple Moving Average": "Buy when SMA 20 crosses above SMA 50, sell when SMA 20 crosses below SMA 100",
}


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### ⚙️ Backtest Settings")
    st.markdown("---")

    ticker = st.text_input(
        "📌 Ticker Symbol",
        value="^NSEI",
        help="Yahoo Finance ticker. E.g.: ^NSEI, RELIANCE.NS, AAPL, SPY"
    )

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("📅 Start Date", value=pd.Timestamp("2019-01-01"))
    with col2:
        end_date = st.date_input("📅 End Date", value=pd.Timestamp("2024-01-01"))

    init_cash = st.number_input(
        "💰 Initial Capital (₹)",
        value=1_000_000,
        min_value=10_000,
        step=100_000,
        format="%d",
    )

    fees = st.slider(
        "💸 Transaction Fees (%)",
        min_value=0.0, max_value=1.0, value=0.1, step=0.05,
        format="%.2f%%",
    )
    fees_decimal = fees / 100

    st.markdown("---")
    st.markdown("### 📚 Example Strategies")
    st.markdown("<p style='color:#8b949e;font-size:0.8rem;'>Click to load an example:</p>", unsafe_allow_html=True)

    for label, strat_text in EXAMPLE_STRATEGIES.items():
        if st.button(label, key=f"example_{label}"):
            st.session_state["strategy_input"] = strat_text

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center;color:#58606a;font-size:0.75rem;'>"
        "Built by <b>Vismay</b> · 2025<br>"
        "Powered by AI + vectorbt + Plotly"
        "</div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ══════════════════════════════════════════════════════════════════════════════

# Header
st.markdown("""
<div class="main-header">
    <h1>🧠 AI-Powered Natural Language Backtester</h1>
    <p>Type a trading strategy in plain English — get instant backtested results</p>
</div>
""", unsafe_allow_html=True)

# Strategy Input
default_strategy = st.session_state.get("strategy_input",
    "Buy when SMA 50 crosses above SMA 200, sell when SMA 50 crosses below SMA 200")

strategy_text = st.text_area(
    "💬 Describe your trading strategy in plain English:",
    value=default_strategy,
    height=100,
    placeholder="e.g., Buy when RSI drops below 30, sell when RSI goes above 70",
)

# Parse button & status
col_btn, col_status = st.columns([1, 3])
with col_btn:
    run_clicked = st.button("🚀 Run Backtest", type="primary")


# ══════════════════════════════════════════════════════════════════════════════
# BACKTEST EXECUTION
# ══════════════════════════════════════════════════════════════════════════════

if run_clicked and strategy_text.strip():
    code = None
    using_ai = True

    # ── AI MODE: AI generates code directly ─────────────────────────────
    try:
        init_ai()

        with st.spinner("🤖 AI is interpreting your strategy..."):
            code = generate_strategy_code(
                strategy=strategy_text,
                ticker=ticker,
                start_date=str(start_date),
                end_date=str(end_date),
                init_cash=init_cash,
                fees=fees_decimal,
            )

        with col_status:
            st.markdown(
                '<div class="parse-box">'
                '🤖 <span class="metric-good">AI interpreted your strategy</span>'
                '<br>Code generated — executing backtest...'
                '</div>',
                unsafe_allow_html=True,
            )

    except AIError as e:
        st.error(f"❌ **AI Error:** {str(e)}")
        st.info("💡 Falling back to rule-based parser...")
        using_ai = False  # fall through to regex mode below

    if not using_ai:
        # ── RULE-BASED FALLBACK ─────────────────────────────────────────
        with st.spinner("🧠 Parsing your strategy..."):
            parsed = parse_strategy(strategy_text)

        if not parsed.is_valid:
            st.error(
                "❌ **Could not parse your strategy.** Please try rephrasing.\n\n"
                "**Supported patterns:**\n"
                "- `Buy when SMA 50 crosses above SMA 200`\n"
                "- `Sell when RSI goes above 70`\n"
                "- `Buy when price crosses above upper Bollinger Band`\n"
                "- `Buy when MACD line crosses above MACD signal`"
            )
            if parsed.warnings:
                for w in parsed.warnings:
                    st.warning(f"⚠️ {w}")
            st.stop()

        with col_status:
            confidence_color = "metric-good" if parsed.parse_confidence >= 0.8 else "metric-neutral"
            st.markdown(
                f'<div class="parse-box">'
                f'📐 Parsed (rule-based) · Confidence: <span class="{confidence_color}">'
                f'{parsed.parse_confidence:.0%}</span>'
                f'<br>{parsed.summary().replace(chr(10), "<br>")}</div>',
                unsafe_allow_html=True,
            )

        if parsed.warnings:
            for w in parsed.warnings:
                st.info(f"ℹ️ {w}")

        with st.spinner("⚙️ Generating backtest code..."):
            code = generate_backtest_code(
                parsed,
                ticker=ticker,
                start_date=str(start_date),
                end_date=str(end_date),
                init_cash=init_cash,
                fees=fees_decimal,
            )

    # ── Execute Backtest (shared by both modes) ─────────────────────────
    with st.spinner("🚀 Running backtest..."):
        try:
            results = run_backtest(code)
        except (BacktestError, Exception) as first_error:
            # If AI mode, auto-retry with error feedback
            if using_ai:
                with st.spinner("🔄 Code had an issue — AI is fixing it..."):
                    try:
                        code = retry_with_error(
                            strategy=strategy_text,
                            previous_code=code,
                            error_message=str(first_error),
                            ticker=ticker,
                            start_date=str(start_date),
                            end_date=str(end_date),
                            init_cash=init_cash,
                            fees=fees_decimal,
                        )
                        results = run_backtest(code)
                        st.success("✅ AI fixed the code on retry!")
                    except Exception as retry_error:
                        st.error(
                            f"❌ **Backtest failed after retry:**\n\n"
                            f"**Original error:** {str(first_error)}\n\n"
                            f"**Retry error:** {str(retry_error)}"
                        )
                        st.code(code, language="python")
                        st.stop()
            else:
                st.error(f"❌ **Backtest failed:**\n\n```\n{str(first_error)}\n```")
                st.code(code, language="python")
                st.stop()

    # ══════════════════════════════════════════════════════════════════════════
    # RESULTS DASHBOARD
    # ══════════════════════════════════════════════════════════════════════════

    st.markdown("---")
    st.markdown("## 📊 Backtest Results")

    # ── Metric Cards ──────────────────────────────────────────────────────────
    m1, m2, m3, m4, m5, m6 = st.columns(6)

    def metric_html(label, value, fmt, good_threshold=None, bad_threshold=None, inverse=False):
        if isinstance(value, (int, np.integer)):
            formatted = str(value)
            css_class = "metric-neutral"
        else:
            formatted = fmt.format(value) if not pd.isna(value) else "N/A"
            if good_threshold is not None and not pd.isna(value):
                if inverse:
                    css_class = "metric-good" if value < good_threshold else ("metric-bad" if value > bad_threshold else "metric-neutral")
                else:
                    css_class = "metric-good" if value > good_threshold else ("metric-bad" if value < bad_threshold else "metric-neutral")
            else:
                css_class = "metric-neutral"
        return f"""<div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value {css_class}">{formatted}</div>
        </div>"""

    with m1:
        st.markdown(metric_html("Total Return", results["total_return"], "{:.2%}", 0, 0), unsafe_allow_html=True)
    with m2:
        st.markdown(metric_html("Sharpe Ratio", results["sharpe_ratio"], "{:.3f}", 1.0, 0), unsafe_allow_html=True)
    with m3:
        st.markdown(metric_html("Max Drawdown", results["max_drawdown"], "{:.2%}", 0.1, 0.3, inverse=True), unsafe_allow_html=True)
    with m4:
        st.markdown(metric_html("Win Rate", results["win_rate"], "{:.1%}", 0.5, 0.3), unsafe_allow_html=True)
    with m5:
        st.markdown(metric_html("Total Trades", results["total_trades"], "{}", None, None), unsafe_allow_html=True)
    with m6:
        st.markdown(metric_html("Final Value", results["final_value"], "₹{:,.0f}", init_cash, init_cash * 0.9), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Benchmark Comparison ──────────────────────────────────────────────────
    bcol1, bcol2, bcol3 = st.columns(3)
    with bcol1:
        st.markdown(metric_html("Buy & Hold Return", results["bh_return"], "{:.2%}", 0, 0), unsafe_allow_html=True)
    with bcol2:
        st.markdown(metric_html("Buy & Hold Sharpe", results["bh_sharpe"], "{:.3f}", 1.0, 0), unsafe_allow_html=True)
    with bcol3:
        alpha = results["total_return"] - results["bh_return"]
        st.markdown(metric_html("Alpha (vs B&H)", alpha, "{:+.2%}", 0, 0), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabbed Results ────────────────────────────────────────────────────────
    if using_ai:
        tab_chart, tab_trades, tab_code, tab_research = st.tabs(
            ["📈 Interactive Chart", "📝 Trade Log", "💻 Generated Code", "🧪 AI Research Note"]
        )
    else:
        tab_chart, tab_trades, tab_code = st.tabs(
            ["📈 Interactive Chart", "📝 Trade Log", "💻 Generated Code"]
        )

    with tab_chart:
        # Build the chart
        close = results["close"]
        buy_signals = results["buy_signals"]
        sell_signals = results["sell_signals"]
        equity = results["equity"]
        bh_equity = results["bh_equity"]

        # Check if we have extra indicators to plot
        extra_indicators = {k: v for k, v in results.items()
                           if k not in {"total_return", "sharpe_ratio", "max_drawdown",
                                       "win_rate", "total_trades", "profit_factor",
                                       "final_value", "equity", "bh_return", "bh_sharpe",
                                       "bh_drawdown", "bh_equity", "close", "buy_signals",
                                       "sell_signals", "trades_readable", "ticker"}
                           and hasattr(v, 'index')}

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.65, 0.35],
            subplot_titles=(
                f"{results['ticker']} — Strategy Performance",
                "Equity Curve Comparison",
            ),
        )

        # Price line
        fig.add_trace(
            go.Scatter(
                x=close.index, y=close.values,
                name="Close Price",
                line=dict(color="#8B8BFF", width=1.5),
                opacity=0.8,
            ),
            row=1, col=1,
        )

        # Extra indicators
        indicator_colors = ["#00D4AA", "#FF6B6B", "#FFD700", "#AB82FF", "#FF8C00", "#00CED1"]
        for i, (name, series) in enumerate(extra_indicators.items()):
            color = indicator_colors[i % len(indicator_colors)]
            fig.add_trace(
                go.Scatter(
                    x=series.index, y=series.values,
                    name=name,
                    line=dict(color=color, width=2, dash="dot"),
                ),
                row=1, col=1,
            )

        # Buy signals
        if len(buy_signals) > 0:
            fig.add_trace(
                go.Scatter(
                    x=buy_signals.index, y=buy_signals.values,
                    mode="markers",
                    name="🟢 BUY",
                    marker=dict(symbol="triangle-up", size=14, color="#00FF88",
                                line=dict(color="white", width=2)),
                ),
                row=1, col=1,
            )

        # Sell signals
        if len(sell_signals) > 0:
            fig.add_trace(
                go.Scatter(
                    x=sell_signals.index, y=sell_signals.values,
                    mode="markers",
                    name="🔴 SELL",
                    marker=dict(symbol="triangle-down", size=14, color="#FF4444",
                                line=dict(color="white", width=2)),
                ),
                row=1, col=1,
            )

        # Equity curves
        fig.add_trace(
            go.Scatter(
                x=equity.index, y=equity.values,
                name="Strategy Equity",
                line=dict(color="#00D4AA", width=2),
                fill="tozeroy", fillcolor="rgba(0, 212, 170, 0.08)",
            ),
            row=2, col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=bh_equity.index, y=bh_equity.values,
                name="Buy & Hold Equity",
                line=dict(color="#FF6B6B", width=2, dash="dash"),
            ),
            row=2, col=1,
        )

        # Layout
        fig.update_layout(
            template="plotly_dark",
            height=750,
            title=dict(
                text=f"<b>Backtest Results</b>"
                     f"<br><sup>{strategy_text[:80]}{'...' if len(strategy_text) > 80 else ''}</sup>",
                font=dict(size=18, family="Inter"),
            ),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                font=dict(size=10),
            ),
            hovermode="x unified",
            yaxis_title="Price",
            yaxis2_title="Portfolio Value (₹)",
            xaxis2_title="Date",
            margin=dict(t=100, b=40, l=60, r=20),
            paper_bgcolor="rgba(10,10,26,0.8)",
            plot_bgcolor="rgba(10,10,26,0.8)",
        )

        st.plotly_chart(fig)

    with tab_trades:
        trades_df = results["trades_readable"]
        if len(trades_df) > 0:
            st.dataframe(
                trades_df,
                height=400,
            )
            st.markdown(f"**Total trades:** {len(trades_df)}")
        else:
            st.info("No trades were executed with this strategy.")

    with tab_code:
        mode_label = "🤖 AI" if using_ai else "📐 Rule-based parser"
        st.markdown(
            f"💡 *This code was generated by **{mode_label}** from your natural language input. "
            f"You can copy and run it independently.*"
        )
        st.code(code, language="python", line_numbers=True)

    # ── AI Research Note Tab (Level 2) ────────────────────────────────────
    if using_ai:
        with tab_research:
            st.markdown(
                '<div style="background:rgba(74,14,143,0.12);border:1px solid rgba(139,139,255,0.2);'
                'border-radius:10px;padding:0.8rem 1.2rem;margin-bottom:1rem;font-size:0.85rem;color:#c9b1ff;">'
                '🧠 <b>AI is analyzing your backtest results...</b> This may take a few seconds.'
                '</div>',
                unsafe_allow_html=True,
            )
            try:
                with st.spinner("🧠 AI is writing a research note..."):
                    research_note = generate_research_note(
                        strategy_description=strategy_text,
                        results=results,
                        ticker=ticker,
                        start_date=str(start_date),
                        end_date=str(end_date),
                    )
                st.markdown(research_note)
            except AIError as e:
                st.warning(f"⚠️ Could not generate research note: {str(e)}")
            except Exception as e:
                st.warning(f"⚠️ Research note generation failed: {str(e)}")


elif run_clicked:
    st.warning("⚠️ Please enter a strategy description first.")


# ══════════════════════════════════════════════════════════════════════════════
# EMPTY STATE — Show when no backtest has been run yet
# ══════════════════════════════════════════════════════════════════════════════

if not run_clicked:
    st.markdown("---")

    st.markdown("""
    <div style="text-align: center; padding: 3rem 1rem;">
        <h3 style="color: #8b949e; font-weight: 400;">
            👆 Type a strategy above and click <span style="color: #00d4aa; font-weight: 700;">Run Backtest</span>
        </h3>
        <p style="color: #58606a; font-size: 0.9rem; max-width: 600px; margin: 1rem auto;">
            Describe your trading strategy in plain English. The AI engine will parse it,
            generate backtest code, execute it, and show you interactive results — all instantly.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Feature cards
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value metric-good" style="font-size:2rem;">📊</div>
            <div class="metric-label" style="margin-top:0.5rem;">6 Indicators</div>
            <p style="color:#8b949e;font-size:0.75rem;margin-top:0.3rem;">SMA, EMA, RSI, MACD, Bollinger Bands, Price</p>
        </div>
        """, unsafe_allow_html=True)
    with f2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value metric-neutral" style="font-size:2rem;">⚡</div>
            <div class="metric-label" style="margin-top:0.5rem;">Instant Backtest</div>
            <p style="color:#8b949e;font-size:0.75rem;margin-top:0.3rem;">Powered by vectorbt for fast execution</p>
        </div>
        """, unsafe_allow_html=True)
    with f3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value" style="font-size:2rem;color:#ffd700;">📈</div>
            <div class="metric-label" style="margin-top:0.5rem;">Interactive Charts</div>
            <p style="color:#8b949e;font-size:0.75rem;margin-top:0.3rem;">Plotly charts with buy/sell markers</p>
        </div>
        """, unsafe_allow_html=True)
    with f4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value" style="font-size:2rem;color:#ff6b6b;">💻</div>
            <div class="metric-label" style="margin-top:0.5rem;">View Generated Code</div>
            <p style="color:#8b949e;font-size:0.75rem;margin-top:0.3rem;">Learn from the auto-generated Python</p>
        </div>
        """, unsafe_allow_html=True)

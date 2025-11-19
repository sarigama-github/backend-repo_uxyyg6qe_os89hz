"""
Deriv Matches Prediction Tool (Async, WebSocket-based)

This script connects to Deriv's WebSocket API, auto-scans multiple volatility indices,
continuously fetches ticks, analyzes recent last-digit patterns, and produces
non-trading prediction signals when an entry condition is met.

Key features:
- Connects with APP ID + API Token
- Concurrently scans multiple markets (e.g., V75, V100, V10, V25, V50)
- Maintains a rolling window (10–20 ticks) per market
- Computes strongest repeating last-digit transition pattern A -> B
  • Entry Point Digit = A (wait for this digit to appear)
  • Predicted Digit   = B (digit expected to match next)
  • Confidence        = P(next=B | current=A) estimates from recent history
- Displays real-time status: current market, entry, predicted, confidence, recent ticks
- Emits a signal when entry digit appears: "ENTRY FOUND → Prediction: <digit> (Confidence: <percentage>%)"
- Clean, modular, and production-ready structure
- Does NOT place trades automatically

Usage:
  export DERIV_APP_ID=YOUR_APP_ID
  export DERIV_API_TOKEN=YOUR_API_TOKEN
  python deriv_matches_predictor.py --markets V75 V100 V50 --window 20

Notes on logic:
- We model digits as a Markov-like transition table over the recent window.
- For each consecutive pair (d_i -> d_{i+1}) we count occurrences.
- We select the transition with highest conditional probability and adequate support.
- If insufficient support, we fallback to the globally most frequent digit within the
  window and set a simple pattern (entry = that digit, predicted = that digit) with
  confidence equal to its frequency ratio.

This script is for research and signaling only. No trades are placed.
"""

import asyncio
import json
import os
import signal
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import AsyncGenerator, Deque, Dict, List, Optional, Tuple

import websockets

WS_URL_TEMPLATE = "wss://ws.derivws.com/websockets/v3?app_id={app_id}"


@dataclass
class Prediction:
    symbol: str
    most_frequent_digit: int
    most_frequent_conf: float
    entry_digit: int
    predicted_digit: int
    transition_conf: float
    window_size: int
    recent_digits: List[int]


async def connect(app_id: str, api_token: str) -> websockets.WebSocketClientProtocol:
    """Connect to Deriv WebSocket and authorize with API token.

    Returns an authorized WebSocket connection ready for subscriptions.
    """
    url = WS_URL_TEMPLATE.format(app_id=app_id)
    ws = await websockets.connect(url, max_queue=32, ping_interval=20, ping_timeout=20)

    # Authorize
    await ws.send(json.dumps({"authorize": api_token}))
    auth_msg = json.loads(await ws.recv())
    if "error" in auth_msg:
        raise RuntimeError(f"Authorization error: {auth_msg['error']}")
    if auth_msg.get("msg_type") != "authorize":
        raise RuntimeError(f"Unexpected authorize response: {auth_msg}")
    return ws


async def subscribe_to_ticks(ws: websockets.WebSocketClientProtocol, symbol: str) -> AsyncGenerator[Dict, None]:
    """Subscribe to tick stream for a symbol and yield tick messages.

    Yields tick dicts with keys: 'symbol', 'epoch', 'quote', 'digit'.
    """
    await ws.send(json.dumps({"ticks": symbol, "subscribe": 1}))
    while True:
        raw = await ws.recv()
        msg = json.loads(raw)
        # Ignore non-tick messages
        if msg.get("msg_type") == "tick" and "tick" in msg:
            tick = msg["tick"]
            price_str = str(tick.get("quote"))
            # Extract last digit from price string (ignore decimal point)
            last_char = next((c for c in reversed(price_str) if c.isdigit()), None)
            if last_char is None:
                continue
            digit = int(last_char)
            yield {
                "symbol": tick.get("symbol", symbol),
                "epoch": tick.get("epoch"),
                "quote": tick.get("quote"),
                "digit": digit,
            }
        elif msg.get("msg_type") == "ping":
            # No action needed; server keepalive
            continue
        elif msg.get("error"):
            raise RuntimeError(f"WebSocket error for {symbol}: {msg['error']}")
        # else: ignore other messages


def analyze_digits(digits: List[int]) -> Tuple[int, float, int, int, float]:
    """Analyze recent digits and return:
    (most_frequent_digit, most_frequent_conf, entry_digit, predicted_digit, transition_conf)

    - most_frequent_digit/conf: global frequency leader and its ratio over the window
    - entry/predicted/transition_conf: strongest transition A->B with confidence P(B|A)
      If insufficient support, fallback to (A=B=most_frequent_digit) and conf=its ratio.
    """
    if not digits:
        return 0, 0.0, 0, 0, 0.0

    n = len(digits)
    freq = Counter(digits)
    mf_digit, mf_count = max(freq.items(), key=lambda kv: kv[1])
    mf_conf = mf_count / n

    # Build transition counts A->B for consecutive pairs
    trans_counts: Dict[int, Counter] = defaultdict(Counter)
    starts: Counter = Counter()
    for a, b in zip(digits[:-1], digits[1:]):
        trans_counts[a][b] += 1
        starts[a] += 1

    # Find best transition by confidence, tiebreak by support then by digit order
    best_a = mf_digit
    best_b = mf_digit
    best_conf = mf_conf

    for a, counts in trans_counts.items():
        total = starts[a]
        if total <= 0:
            continue
        b, c = max(counts.items(), key=lambda kv: kv[1])
        conf = c / total
        # Require minimal support to avoid noise (at least 2 observations)
        if c >= 2:
            if conf > best_conf or (abs(conf - best_conf) < 1e-9 and c > freq.get(best_b, 0)):
                best_a, best_b, best_conf = a, b, conf

    return mf_digit, mf_conf, best_a, best_b, best_conf


def format_digits(digs: List[int], max_len: int = 20) -> str:
    s = ''.join(str(d) for d in digs[-max_len:])
    return s.rjust(max_len)


async def display_prediction(pred: Prediction) -> None:
    """Display real-time prediction status for a market."""
    print(
        f"[ {pred.symbol} ] | Entry: {pred.entry_digit} | Predict: {pred.predicted_digit} "
        f"| Confidence: {pred.transition_conf*100:.1f}% | Window: {pred.window_size} | Recent: {format_digits(pred.recent_digits)}"
    )


async def run_market_scan(app_id: str, api_token: str, symbol: str, window: int = 20, event_signal: asyncio.Queue = None):
    """Run a scanner for a single market symbol.

    Maintains a rolling window of digits and updates predictions. When an
    entry condition is met (last digit equals entry_digit), emits a signal
    via the queue with details.
    """
    ws = await connect(app_id, api_token)
    digits: Deque[int] = deque(maxlen=window)
    try:
        async for tick in subscribe_to_ticks(ws, symbol):
            digits.append(tick["digit"]) 
            if len(digits) < max(10, window // 2):
                # warm-up period to accumulate enough context
                continue
            mf_digit, mf_conf, entry_d, pred_d, trans_conf = analyze_digits(list(digits))
            pred = Prediction(
                symbol=symbol,
                most_frequent_digit=mf_digit,
                most_frequent_conf=mf_conf,
                entry_digit=entry_d,
                predicted_digit=pred_d,
                transition_conf=trans_conf,
                window_size=len(digits),
                recent_digits=list(digits),
            )
            await display_prediction(pred)

            # Entry condition: current digit equals entry digit
            if digits[-1] == entry_d and event_signal is not None:
                await event_signal.put({
                    "symbol": symbol,
                    "message": f"ENTRY FOUND → Prediction: {pred_d} (Confidence: {trans_conf*100:.1f}%)",
                    "confidence": trans_conf,
                    "entry_digit": entry_d,
                    "predicted_digit": pred_d,
                    "recent_digits": list(digits),
                })
    except asyncio.CancelledError:
        pass
    finally:
        try:
            await ws.close()
        except Exception:
            pass


async def scan_markets(app_id: str, api_token: str, markets: List[str], window: int = 20, runtime: Optional[int] = None):
    """Scan multiple markets concurrently.

    - markets: list of symbols (e.g., ["R_75", "R_100", "R_10", "R_25", "R_50"]).
      Deriv symbols for volatility indices typically are: R_10, R_25, R_50, R_75, R_100, etc.
      For synthetic indices with continuous trading, these are common.
    - runtime: optional seconds to run before stopping; None runs indefinitely.
    """
    q: asyncio.Queue = asyncio.Queue()

    tasks = [
        asyncio.create_task(run_market_scan(app_id, api_token, sym, window, q))
        for sym in markets
    ]

    async def _stop_after(delay: int):
        await asyncio.sleep(delay)
        for t in tasks:
            t.cancel()

    stopper = None
    if runtime is not None and runtime > 0:
        stopper = asyncio.create_task(_stop_after(runtime))

    print("\nScanning markets:", ", ".join(markets))
    print("Waiting for entry signals... Press Ctrl+C to stop.\n")

    try:
        while True:
            signal_msg = await q.get()
            print(
                f"[ {signal_msg['symbol']} ] {signal_msg['message']} | Last digits: {''.join(str(d) for d in signal_msg['recent_digits'][-20:])}"
            )
    except asyncio.CancelledError:
        pass
    except KeyboardInterrupt:
        print("\nStopping...\n")
    finally:
        for t in tasks:
            t.cancel()
        if stopper:
            stopper.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def _derive_symbols_from_args(raw_markets: List[str]) -> List[str]:
    """Map human-friendly inputs like V75 to Deriv symbols like R_75."""
    mapping = {
        "V10": "R_10",
        "V25": "R_25",
        "V50": "R_50",
        "V75": "R_75",
        "V100": "R_100",
        # You can extend with other variants like 1s, 2s etc., if desired
    }
    symbols = []
    for m in raw_markets:
        m_up = m.upper().strip()
        symbols.append(mapping.get(m_up, m_up))
    return symbols


def _read_env_or_args(app_id: Optional[str], token: Optional[str]) -> Tuple[str, str]:
    app_id = app_id or os.getenv("DERIV_APP_ID")
    token = token or os.getenv("DERIV_API_TOKEN")
    if not app_id or not token:
        raise SystemExit("Please provide APP ID and API Token via args or environment variables DERIV_APP_ID/DERIV_API_TOKEN.")
    return app_id, token


async def _amain(app_id: Optional[str], token: Optional[str], markets: List[str], window: int, runtime: Optional[int]):
    app_id, token = _read_env_or_args(app_id, token)
    symbols = _derive_symbols_from_args(markets or ["V75", "V100", "V10", "V25", "V50"])
    await scan_markets(app_id, token, symbols, window=window, runtime=runtime)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Deriv Matches Prediction Tool (Signals Only)")
    parser.add_argument("--app-id", type=str, default=None, help="Deriv APP ID (or set DERIV_APP_ID)")
    parser.add_argument("--api-token", type=str, default=None, help="Deriv API Token (or set DERIV_API_TOKEN)")
    parser.add_argument("--markets", nargs="*", default=["V75", "V100", "V10", "V25", "V50"], help="Markets to scan (e.g., V75 V100 V50)")
    parser.add_argument("--window", type=int, default=20, help="Digits analysis window size (10–20 recommended)")
    parser.add_argument("--runtime", type=int, default=None, help="Optional runtime in seconds (default: run until Ctrl+C)")

    args = parser.parse_args()

    # Graceful shutdown on SIGINT/SIGTERM for asyncio on some platforms
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, loop.stop)
        except NotImplementedError:
            # add_signal_handler not available on some platforms (e.g., Windows)
            pass

    try:
        loop.run_until_complete(_amain(args.app_id, args.api_token, args.markets, args.window, args.runtime))
    finally:
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
        try:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        except Exception:
            pass
        loop.close()


if __name__ == "__main__":
    main()

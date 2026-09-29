# -*- coding: utf-8 -*-
"""
BTC 三均线区间策略（4小时线）—— 用户提供的原始代码，逻辑未改动（只保留了回测需要的部分，
去掉了长注释和实时信号打印）。审查和变体测试在 audit.py 里。

    黄线(MA7) 和 粉线(MA25) 都在 紫线(MA99) 上方  ->  只做多
    黄线 和 粉线 都在 紫线 下方                    ->  只做空
    一条在上一条在下                               ->  空仓
    多单不止盈；空单 1R 止盈；浮盈 1R 移保本；
    R = 2×ATR14/价格，R≥8% 不开；名义 = min(账户×0.5%/R, 账户×3)
"""
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

MA_FAST = 7
MA_MID = 25
MA_SLOW = 99
ATR_N = 14
ATR_MULT = 2.0
RISK = 0.005
GATE = 0.08
MAX_NOTIONAL_X = 3.0
LEVERAGE = 10
BE_AT_R = 1.0
TP_SHORT_R = 1.0
TP_LONG_R = None
FEE_ONE_WAY = 0.0005
MMR = 0.005
WARMUP = 300


def build_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    c = d["close"].astype(float)
    d["ma_fast"] = c.rolling(MA_FAST).mean()
    d["ma_mid"] = c.rolling(MA_MID).mean()
    d["ma_slow"] = c.rolling(MA_SLOW).mean()
    h = d["high"].astype(float)
    lo = d["low"].astype(float)
    pc = c.shift(1)
    tr = pd.concat([h - lo, (h - pc).abs(), (lo - pc).abs()], axis=1).max(axis=1)
    d["atr"] = tr.ewm(alpha=1.0 / ATR_N, min_periods=ATR_N, adjust=False).mean()
    ok = np.isfinite(d["ma_slow"]) & np.isfinite(d["atr"])
    d["ok"] = ok
    d["reg_long"] = ok & (d["ma_fast"] > d["ma_slow"]) & (d["ma_mid"] > d["ma_slow"])
    d["reg_short"] = ok & (d["ma_fast"] < d["ma_slow"]) & (d["ma_mid"] < d["ma_slow"])
    want = np.zeros(len(d), dtype=np.int8)
    want[d["reg_long"].to_numpy()] = 1
    want[d["reg_short"].to_numpy()] = -1
    d["want"] = want
    return d


def prepare(df: pd.DataFrame, warmup: int = WARMUP) -> pd.DataFrame:
    d = build_indicators(df)
    if warmup and warmup < len(d):
        d = d.iloc[warmup:].reset_index(drop=True)
    return d


@dataclass
class Sizing:
    ok: bool
    reason: str = ""
    rd: float = 0.0
    qty: float = 0.0
    notional: float = 0.0
    tp: Optional[float] = None


def size_position(direction: int, price: float, atr_prev: float, equity: float,
                  step_size: float = 0.001, min_qty: float = 0.001,
                  min_notional: float = 5.0) -> Sizing:
    if not (np.isfinite(atr_prev) and atr_prev > 0):
        return Sizing(False, "ATR 无效")
    if not (np.isfinite(price) and price > 0):
        return Sizing(False, "价格无效")
    if equity <= 0:
        return Sizing(False, "账户净值 <= 0")
    rd = ATR_MULT * atr_prev / price
    if rd <= 0:
        return Sizing(False, "风险距离 <= 0")
    if rd >= GATE:
        return Sizing(False, "波动过大", rd=rd)
    want_notional = min(RISK * equity / rd, MAX_NOTIONAL_X * equity)
    qty = np.floor(want_notional / price / step_size + 1e-9) * step_size
    dec = max(0, -int(round(np.log10(step_size)))) if step_size > 0 else 8
    qty = round(qty, dec)
    if qty < min_qty:
        return Sizing(False, "数量低于最小下单量", rd=rd)
    if qty * price < min_notional:
        return Sizing(False, "名义低于最小名义", rd=rd)
    tpr = TP_SHORT_R if direction == -1 else TP_LONG_R
    tp = price * (1 + direction * tpr * rd) if tpr else None
    return Sizing(True, "", rd=rd, qty=qty, notional=qty * price, tp=tp)


@dataclass
class Position:
    dir: int
    entry: float
    qty: float
    notional: float
    rd: float
    be_done: bool = False
    stop: Optional[float] = None
    tp: Optional[float] = None

    def r_of(self, px: float) -> float:
        if self.rd <= 0:
            return 0.0
        return self.dir * (px - self.entry) / self.entry / self.rd

    def liq_price(self, equity: float) -> Optional[float]:
        if self.qty <= 0 or equity <= 0:
            return None
        q = self.qty
        if self.dir == 1:
            return (q * self.entry - equity) / (q * (1 - MMR))
        return (equity + q * self.entry) / (q * (1 + MMR))


@dataclass
class ExitSignal:
    hit: bool
    reason: str = ""
    price: float = 0.0


def check_protective(pos: Position, high: float, low: float) -> ExitSignal:
    d = pos.dir
    if pos.stop is not None:
        if (d == 1 and low <= pos.stop) or (d == -1 and high >= pos.stop):
            return ExitSignal(True, "stop", pos.stop)
    if pos.tp is not None:
        if (d == 1 and high >= pos.tp) or (d == -1 and low <= pos.tp):
            return ExitSignal(True, "take_profit", pos.tp)
    return ExitSignal(False)


def check_stop_gap(pos: Position, open_px: float) -> ExitSignal:
    if pos.stop is None:
        return ExitSignal(False)
    d = pos.dir
    if (d == 1 and open_px <= pos.stop) or (d == -1 and open_px >= pos.stop):
        return ExitSignal(True, "stop_gap", open_px)
    return ExitSignal(False)


def arm_breakeven(pos: Position, bar_close: float) -> bool:
    if pos.be_done:
        return False
    if pos.dir * (bar_close - pos.entry) / pos.entry >= BE_AT_R * pos.rd:
        e = pos.entry
        if pos.stop is None:
            pos.stop = e
        else:
            pos.stop = max(pos.stop, e) if pos.dir == 1 else min(pos.stop, e)
        pos.be_done = True
        return True
    return False


@dataclass
class Decision:
    action: str
    target_dir: int
    note: str = ""


def decide(want: int, pos: Optional[Position], blocked_dir: int):
    bd = blocked_dir
    if bd != 0 and want != bd:
        bd = 0
    cur = 0 if pos is None else pos.dir
    if want == cur:
        return Decision("hold", cur), bd
    if pos is not None:
        if want == 0:
            return Decision("close", 0), bd
        return Decision("reverse", want), bd
    if want == bd:
        return Decision("hold", 0), bd
    return Decision("open", want), bd


@dataclass
class Trade:
    dir: int
    entry_ts: object
    entry: float
    exit_ts: object
    exit: float
    qty: float
    reason: str
    pnl: float
    r_multiple: float


def backtest(df: pd.DataFrame, equity0: float = 100_000.0,
             step_size: float = 0.001, warmup: int = WARMUP) -> dict:
    d = prepare(df, warmup).reset_index(drop=True)
    equity = equity0
    pos: Optional[Position] = None
    blocked_dir = 0
    trades: List[Trade] = []
    liquidations = 0
    curve = []
    for i in range(1, len(d)):
        bar = d.iloc[i]
        prev = d.iloc[i - 1]
        if not bool(bar["ok"]):
            curve.append(equity)
            continue
        o, h, lo, c = (float(bar["open"]), float(bar["high"]),
                       float(bar["low"]), float(bar["close"]))
        t_prev, t_bar = prev["time"], bar["time"]
        if pos is not None:
            arm_breakeven(pos, float(prev["close"]))
        if pos is not None:
            g = check_stop_gap(pos, o)
            if g.hit:
                pnl = pos.dir * (g.price - pos.entry) * pos.qty
                fee = g.price * pos.qty * FEE_ONE_WAY
                equity += pnl - fee
                trades.append(Trade(pos.dir, t_prev, pos.entry, t_bar, g.price,
                                    pos.qty, "stop_gap", pnl - fee, pos.r_of(g.price)))
                blocked_dir = pos.dir
                pos = None
        want = int(prev["want"])
        dec, blocked_dir = decide(want, pos, blocked_dir)
        if dec.action in ("close", "reverse") and pos is not None:
            pnl = pos.dir * (o - pos.entry) * pos.qty
            fee = o * pos.qty * FEE_ONE_WAY
            equity += pnl - fee
            trades.append(Trade(pos.dir, t_prev, pos.entry, t_bar, o, pos.qty,
                                "signal_exit", pnl - fee, pos.r_of(o)))
            pos = None
        if dec.action in ("open", "reverse") and dec.target_dir != 0:
            sz = size_position(dec.target_dir, o, float(prev["atr"]), equity, step_size=step_size)
            if sz.ok:
                entry_fee = o * sz.qty * FEE_ONE_WAY
                equity -= entry_fee
                pos = Position(dir=dec.target_dir, entry=o, qty=sz.qty,
                               notional=sz.notional, rd=sz.rd, tp=sz.tp)
        if pos is not None:
            lp = pos.liq_price(equity)
            liq_hit = lp is not None and (
                (pos.dir == 1 and lo <= lp) or (pos.dir == -1 and h >= lp))
            if liq_hit:
                px = lp
                pnl = pos.dir * (px - pos.entry) * pos.qty
                equity += pnl
                liquidations += 1
                trades.append(Trade(pos.dir, t_prev, pos.entry, t_bar, px,
                                    pos.qty, "liquidation", pnl, pos.r_of(px)))
                blocked_dir = pos.dir
                pos = None
            else:
                sig = check_protective(pos, h, lo)
                if sig.hit:
                    pnl = pos.dir * (sig.price - pos.entry) * pos.qty
                    fee = sig.price * pos.qty * FEE_ONE_WAY
                    equity += pnl - fee
                    trades.append(Trade(pos.dir, t_prev, pos.entry, t_bar,
                                        sig.price, pos.qty, sig.reason, pnl - fee,
                                        pos.r_of(sig.price)))
                    blocked_dir = pos.dir
                    pos = None
        curve.append(equity)
    if pos is not None:
        lastrow = d.iloc[-1]
        c = float(lastrow["close"])
        pnl = pos.dir * (c - pos.entry) * pos.qty
        fee = c * pos.qty * FEE_ONE_WAY
        equity += pnl - fee
        trades.append(Trade(pos.dir, lastrow["time"], pos.entry, lastrow["time"], c,
                            pos.qty, "end_of_data", pnl - fee, pos.r_of(c)))
    return _summarize(trades, curve, equity0, liquidations)


def _summarize(trades: List[Trade], curve: List[float], equity0: float, liquidations: int) -> dict:
    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    gross_win = sum(t.pnl for t in wins)
    gross_loss = -sum(t.pnl for t in losses)
    pf = (gross_win / gross_loss) if gross_loss > 0 else float("inf")
    eq = np.array([equity0] + list(curve), dtype=float)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    longs = [t for t in trades if t.dir == 1]
    shorts = [t for t in trades if t.dir == -1]

    def _pf(ts):
        w = sum(t.pnl for t in ts if t.pnl > 0)
        l = -sum(t.pnl for t in ts if t.pnl <= 0)
        return (w / l) if l > 0 else float("inf")

    def _wr(ts):
        return (len([t for t in ts if t.pnl > 0]) / len(ts) * 100) if ts else 0.0

    return {
        "trades": trades, "equity0": equity0, "equity_final": eq[-1],
        "return_pct": (eq[-1] / equity0 - 1) * 100, "pf": pf, "win_rate": _wr(trades),
        "max_dd_pct": float(dd.min() * 100), "n": len(trades), "liquidations": liquidations,
        "longs": len(longs), "long_pf": _pf(longs), "long_wr": _wr(longs),
        "shorts": len(shorts), "short_pf": _pf(shorts), "short_wr": _wr(shorts), "curve": eq,
    }


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    out = pd.DataFrame({k: df[k].astype(float) for k in ("open", "high", "low", "close")})
    out["time"] = pd.to_datetime(df["time"])
    return out

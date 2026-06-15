package ats.domain;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * 파라미터 1조합을 과거 데이터에 적용해 성과를 계산하는 백테스트 엔진.
 * GAOptimizer 적합도 평가와 WFO 검증에서 공용으로 사용.
 * 수수료(taker 0.04%/side) 반영, 종가 기준 체결 가정.
 */
public final class Backtester {
    public static final double FEE = 0.0004;

    public record Result(double roi, double mdd, int trades, double winRate,
                         double sharpe, List<TradeRecord> records, double[] equityCurve) { }

    private Backtester() { }

    public static Result run(Map<String, Object> p, List<MarketData> data,
                             double initialEquity, double riskPct, int leverage) {
        int n = data.size();
        int rp = pi(p, "rsiPeriod", 14), ms = pi(p, "maShort", 7), ml = pi(p, "maLong", 25);
        double os = pd(p, "rsiOversold", 35), ob = pd(p, "rsiOverbought", 65);
        double sl = pd(p, "stopLoss", 0.01), tp = pd(p, "takeProfit", 0.02);

        double[] c = new double[n];
        for (int i = 0; i < n; i++) c[i] = data.get(i).close();
        double[] s1 = Indicators.sma(c, ms);
        double[] s2 = Indicators.sma(c, ml);
        double[] r  = Indicators.rsi(c, rp);

        double eq = initialEquity, peak = eq, mdd = 0;
        int dir = 0; double entry = 0, qty = 0; long entryT = 0;
        List<TradeRecord> recs = new ArrayList<>();
        double[] curve = new double[n];
        String sym = "BACKTEST";
        int warm = Math.max(ml, rp) + 1;

        for (int i = 0; i < n; i++) {
            double px = c[i];
            if (i >= warm && !Double.isNaN(s2[i]) && !Double.isNaN(s2[i - 1]) && !Double.isNaN(r[i])) {
                boolean up      = s1[i] > s2[i];
                boolean crossUp = s1[i - 1] <= s2[i - 1] && s1[i] > s2[i];
                boolean crossDn = s1[i - 1] >= s2[i - 1] && s1[i] < s2[i];
                Signal sig = Signal.HOLD;
                if (crossUp || (r[i] < os &&  up)) sig = Signal.BUY;
                else if (crossDn || (r[i] > ob && !up)) sig = Signal.SELL;

                if (dir != 0) {
                    double chg = (px - entry) / entry * dir;
                    boolean exit = chg <= -sl || chg >= tp
                            || (dir > 0 && sig == Signal.SELL) || (dir < 0 && sig == Signal.BUY);
                    if (exit) {
                        double pnl = (px - entry) * qty * dir - FEE * qty * (entry + px);
                        eq += pnl;
                        recs.add(new TradeRecord(sym, dir > 0 ? OrderSide.BUY : OrderSide.SELL,
                                entryT, data.get(i).timestamp(), entry, px, qty, pnl));
                        dir = 0;
                    }
                }
                if (dir == 0 && sig != Signal.HOLD && eq > 0) {
                    dir = sig == Signal.BUY ? 1 : -1;
                    entry = px; entryT = data.get(i).timestamp();
                    qty = eq * riskPct * leverage / px;
                }
            }
            if (eq > peak) peak = eq;
            double dd = (peak - eq) / peak;
            if (dd > mdd) mdd = dd;
            curve[i] = eq;
        }

        double roi = (eq - initialEquity) / initialEquity;
        int wins = 0; double mean = 0;
        for (TradeRecord t : recs) { if (t.realizedPnl() > 0) wins++; mean += ret(t); }
        double winRate = recs.isEmpty() ? 0 : (double) wins / recs.size();
        double sharpe = 0;
        if (recs.size() > 1) {
            mean /= recs.size();
            double var = 0;
            for (TradeRecord t : recs) { double x = ret(t) - mean; var += x * x; }
            double sd = Math.sqrt(var / (recs.size() - 1));
            sharpe = sd == 0 ? 0 : mean / sd * Math.sqrt(recs.size());
        }
        return new Result(roi, mdd, recs.size(), winRate, sharpe, recs, curve);
    }

    private static double ret(TradeRecord t) { return t.realizedPnl() / (t.entryPrice() * t.quantity()); }
    private static int    pi(Map<String, Object> p, String k, int d)    { Object v = p.get(k); return v == null ? d : (int) Math.round(((Number) v).doubleValue()); }
    private static double pd(Map<String, Object> p, String k, double d) { Object v = p.get(k); return v == null ? d : ((Number) v).doubleValue(); }
}

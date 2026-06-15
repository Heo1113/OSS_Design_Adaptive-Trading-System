package ats.domain;

import java.util.ArrayList;
import java.util.List;

/** TradeRecord 집계 → ROI / MDD / 승률 / Sharpe / 자본곡선 (Design 2.2) */
public class PerformanceReport {
    public static final double INITIAL_EQUITY = 10_000;
    private final List<TradeRecord> records;

    public PerformanceReport(List<TradeRecord> records) {
        this.records = new ArrayList<>(records);
        this.records.sort((a, b) -> Long.compare(a.exitTime(), b.exitTime()));
    }

    public List<TradeRecord> getRecords() { return new ArrayList<>(records); }
    public int tradeCount() { return records.size(); }

    public double roi() {
        double eq = INITIAL_EQUITY;
        for (TradeRecord t : records) eq += t.realizedPnl();
        return (eq - INITIAL_EQUITY) / INITIAL_EQUITY;
    }

    public double maxDrawdown() {
        double eq = INITIAL_EQUITY, peak = eq, mdd = 0;
        for (TradeRecord t : records) {
            eq += t.realizedPnl();
            if (eq > peak) peak = eq;
            double dd = (peak - eq) / peak;
            if (dd > mdd) mdd = dd;
        }
        return mdd;
    }

    public double winRate() {
        if (records.isEmpty()) return 0;
        long wins = records.stream().filter(t -> t.realizedPnl() > 0).count();
        return (double) wins / records.size();
    }

    /** 거래 단위 수익률의 평균/표준편차 기반 (단순화된 per-trade Sharpe) */
    public double sharpeRatio() {
        if (records.size() < 2) return 0;
        double mean = 0;
        for (TradeRecord t : records) mean += ret(t);
        mean /= records.size();
        double var = 0;
        for (TradeRecord t : records) { double x = ret(t) - mean; var += x * x; }
        double sd = Math.sqrt(var / (records.size() - 1));
        return sd == 0 ? 0 : mean / sd * Math.sqrt(records.size());
    }

    public List<Double> equityCurve() {
        List<Double> c = new ArrayList<>();
        double eq = INITIAL_EQUITY;
        c.add(eq);
        for (TradeRecord t : records) { eq += t.realizedPnl(); c.add(eq); }
        return c;
    }

    private static double ret(TradeRecord t) { return t.realizedPnl() / (t.entryPrice() * t.quantity()); }
}

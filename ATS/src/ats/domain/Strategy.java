package ats.domain;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * 매매 전략. RSI 평균회귀 + 이동평균 추세 필터/크로스오버 혼합 전략.
 * 파라미터는 GAOptimizer가 갱신함. (Design 2.2 Strategy)
 */
public class Strategy {
    private final Map<String, Object> params = new LinkedHashMap<>();
    private final String symbol;

    public Strategy(Map<String, Object> initialParams, String symbol) {
        if (initialParams != null) params.putAll(initialParams);
        this.symbol = symbol;
        defaults();
    }

    private void defaults() {
        params.putIfAbsent("rsiPeriod", 14);
        params.putIfAbsent("rsiOversold", 35.0);
        params.putIfAbsent("rsiOverbought", 65.0);
        params.putIfAbsent("maShort", 7);
        params.putIfAbsent("maLong", 25);
        params.putIfAbsent("stopLoss", 0.010);
        params.putIfAbsent("takeProfit", 0.020);
    }

    /** 최근 데이터 윈도우에 대해 매수/매도/홀드 신호 산출 */
    public Signal evaluate(List<MarketData> window) {
        int rp = i("rsiPeriod"), ms = i("maShort"), ml = i("maLong");
        int need = Math.max(ml, rp) + 3;
        if (window.size() < need) return Signal.HOLD;

        int from = Math.max(0, window.size() - (need + 10));
        double[] c = new double[window.size() - from];
        for (int k = 0; k < c.length; k++) c[k] = window.get(from + k).close();

        double[] s1 = Indicators.sma(c, ms);
        double[] s2 = Indicators.sma(c, ml);
        double[] r  = Indicators.rsi(c, rp);
        int last = c.length - 1;
        if (Double.isNaN(s2[last]) || Double.isNaN(s2[last - 1]) || Double.isNaN(r[last]))
            return Signal.HOLD;

        boolean up      = s1[last] > s2[last];
        boolean crossUp = s1[last - 1] <= s2[last - 1] && s1[last] > s2[last];
        boolean crossDn = s1[last - 1] >= s2[last - 1] && s1[last] < s2[last];
        double rsi = r[last];

        if (crossUp || (rsi < d("rsiOversold")  &&  up)) return Signal.BUY;
        if (crossDn || (rsi > d("rsiOverbought") && !up)) return Signal.SELL;
        return Signal.HOLD;
    }

    /** GA 결과 반영 */
    public void updateParams(Map<String, Object> p) { params.putAll(p); }

    public Map<String, Object> getParams() { return new LinkedHashMap<>(params); }
    public String getSymbol() { return symbol; }

    public int    i(String k) { return (int) Math.round(((Number) params.get(k)).doubleValue()); }
    public double d(String k) { return ((Number) params.get(k)).doubleValue(); }
}

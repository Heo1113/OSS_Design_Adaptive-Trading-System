package ats.domain;

import java.util.Arrays;

/** 기술적 지표 계산 (외부 라이브러리 없이 직접 구현) */
public final class Indicators {
    private Indicators() { }

    /** 단순이동평균. warmup 이전 구간은 NaN */
    public static double[] sma(double[] c, int period) {
        int n = c.length;
        double[] r = new double[n];
        Arrays.fill(r, Double.NaN);
        double sum = 0;
        for (int i = 0; i < n; i++) {
            sum += c[i];
            if (i >= period) sum -= c[i - period];
            if (i >= period - 1) r[i] = sum / period;
        }
        return r;
    }

    /** Wilder RSI. warmup 이전 구간은 NaN */
    public static double[] rsi(double[] c, int period) {
        int n = c.length;
        double[] r = new double[n];
        Arrays.fill(r, Double.NaN);
        if (n <= period) return r;
        double gain = 0, loss = 0;
        for (int i = 1; i <= period; i++) {
            double d = c[i] - c[i - 1];
            if (d > 0) gain += d; else loss -= d;
        }
        gain /= period; loss /= period;
        r[period] = loss == 0 ? 100 : 100 - 100 / (1 + gain / loss);
        for (int i = period + 1; i < n; i++) {
            double d = c[i] - c[i - 1];
            double up = Math.max(d, 0), dn = Math.max(-d, 0);
            gain = (gain * (period - 1) + up) / period;
            loss = (loss * (period - 1) + dn) / period;
            r[i] = loss == 0 ? 100 : 100 - 100 / (1 + gain / loss);
        }
        return r;
    }
}

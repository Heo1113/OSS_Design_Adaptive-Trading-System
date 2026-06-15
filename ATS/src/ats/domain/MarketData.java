package ats.domain;

/** OHLCV 한 캔들 (Design 2.2 데이터 클래스) */
public record MarketData(long timestamp, double open, double high,
                         double low, double close, double volume) { }

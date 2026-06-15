package ats.domain;

/** BinanceConnector → TradingEngine 시세 콜백 */
@FunctionalInterface
public interface MarketDataListener { void onMarketData(MarketData data); }

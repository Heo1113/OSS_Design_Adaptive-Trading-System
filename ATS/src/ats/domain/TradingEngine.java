package ats.domain;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.function.Consumer;

/**
 * 시스템 전체의 가동 상태를 관리하는 핵심 오케스트레이터 (Design 2.2 / 4.1).
 * 상태 전이: STOPPED --start()--> RUNNING --stop()--> STOPPED, RUNNING --fatalError--> ERROR.
 * 시세 콜백(onMarketData)마다 전략 평가 → 신호 발생 시 주문 → 포지션/기록 갱신.
 */
public class TradingEngine {

    /** GUI 등 외부 관찰자용 이벤트 리스너 */
    public interface EngineListener {
        default void onState(EngineState s) { }
        default void onCandle(MarketData d) { }
        default void onPosition(Position p) { }       // null = 포지션 없음
        default void onTrade(TradeRecord r) { }
        default void onAlert(String msg) { }
    }

    private volatile EngineState state = EngineState.STOPPED;
    private final Strategy strategy;
    private final Object connectorLock = new Object();
    private final ConnectorPort connector;
    private final Consumer<TradeRecord> tradeSink;     // DAO 저장 위임
    private final Deque<MarketData> window = new ArrayDeque<>();
    private final List<EngineListener> listeners = new CopyOnWriteArrayList<>();

    private String symbol = "BTCUSDT";
    private double riskPct = 0.10;
    private int leverage = 5;
    private volatile Position position;
    private volatile double equity = 10_000;

    /** Infrastructure 의존 역전용 포트 (BinanceConnector가 구현) */
    public interface ConnectorPort {
        List<MarketData> fetchHistorical(String symbol, String interval, int limit) throws Exception;
        void openStream(String symbol, MarketDataListener listener) throws Exception;
        void closeStream();
        Map<String, Object> placeOrder(Order order);
        boolean isDemo();
    }

    public TradingEngine(Strategy strategy, ConnectorPort connector, Consumer<TradeRecord> tradeSink) {
        this.strategy = strategy;
        this.connector = connector;
        this.tradeSink = tradeSink == null ? r -> { } : tradeSink;
    }

    public void configure(String symbol, double riskPct, int leverage) {
        this.symbol = symbol; this.riskPct = riskPct; this.leverage = leverage;
    }

    public void addListener(EngineListener l) { listeners.add(l); }

    public EngineState getState() { return state; }
    public double getEquity()     { return equity; }
    public Position getPosition() { return position; }
    public Strategy getStrategy() { return strategy; }

    public List<MarketData> snapshotWindow() {
        synchronized (window) { return new ArrayList<>(window); }
    }

    /** STOPPED → RUNNING. 워밍업 데이터 적재 후 실시간 스트림 개통 */
    public synchronized boolean start() {
        if (state == EngineState.RUNNING) return false;
        try {
            synchronized (window) { window.clear(); }
            List<MarketData> warm = connector.fetchHistorical(symbol, "1m", 300);
            synchronized (window) { window.addAll(warm); }
            synchronized (connectorLock) { connector.openStream(symbol, this::onMarketData); }
            setState(EngineState.RUNNING);
            alert("엔진 시작 — " + (connector.isDemo() ? "DEMO" : "LIVE") + " / " + symbol
                    + " / lev x" + leverage);
            return true;
        } catch (Exception e) {
            setState(EngineState.ERROR);
            alert("시작 실패: " + e.getMessage());
            return false;
        }
    }

    /** RUNNING/ERROR → STOPPED */
    public synchronized boolean stop() {
        if (state == EngineState.STOPPED) return false;
        synchronized (connectorLock) { connector.closeStream(); }
        setState(EngineState.STOPPED);
        alert("엔진 정지");
        return true;
    }

    /** 신규 캔들 수신 콜백 (Design Seq 3.4 Generate Signal → Place Order) */
    public void onMarketData(MarketData d) {
        try {
            synchronized (window) {
                window.addLast(d);
                while (window.size() > 600) window.removeFirst();
            }
            for (EngineListener l : listeners) l.onCandle(d);
            if (state != EngineState.RUNNING) return;

            double px = d.close();
            Signal sig = strategy.evaluate(snapshotWindow());

            Position pos = position;
            if (pos != null) {
                pos.updateMark(px);
                for (EngineListener l : listeners) l.onPosition(pos);
                double chg = (px - pos.getEntryPrice()) / pos.getEntryPrice() * pos.dir();
                double sl = strategy.d("stopLoss"), tp = strategy.d("takeProfit");
                String reason = null;
                if (chg <= -sl) reason = "손절(SL)";
                else if (chg >= tp) reason = "익절(TP)";
                else if ((pos.dir() > 0 && sig == Signal.SELL) || (pos.dir() < 0 && sig == Signal.BUY))
                    reason = "반대신호";
                if (reason != null) closePosition(px, d.timestamp(), reason);
            } else if (sig == Signal.BUY || sig == Signal.SELL) {
                openPosition(sig, px, d.timestamp());
            }
        } catch (Exception e) {
            setState(EngineState.ERROR);
            alert("런타임 오류: " + e);
        }
    }

    private void openPosition(Signal sig, double px, long ts) {
        OrderSide side = sig == Signal.BUY ? OrderSide.BUY : OrderSide.SELL;
        double qty = Math.max(0.001, Math.round(equity * riskPct * leverage / px * 1000) / 1000.0);
        Order o = new Order(side, OrderType.MARKET, qty, px);
        o.submit();
        Map<String, Object> resp = connector.placeOrder(o);
        if (o.getStatus() != OrderStatus.FILLED) {
            alert("주문 실패: " + resp.get("msg"));
            return;
        }
        Position p = new Position(symbol, side, qty, o.getPrice(), leverage);
        p.setEntryTime(ts);
        position = p;
        alert((side == OrderSide.BUY ? "롱" : "숏") + " 진입 " + qty + " @ " + fmt(o.getPrice()));
        for (EngineListener l : listeners) l.onPosition(p);
    }

    private void closePosition(double px, long ts, String reason) {
        Position pos = position;
        if (pos == null) return;
        OrderSide closeSide = pos.getSide() == OrderSide.BUY ? OrderSide.SELL : OrderSide.BUY;
        Order o = new Order(closeSide, OrderType.MARKET, pos.getSize(), px);
        o.submit();
        connector.placeOrder(o);
        double exitPx = o.getStatus() == OrderStatus.FILLED ? o.getPrice() : px;

        double fee = Backtester.FEE * pos.getSize() * (pos.getEntryPrice() + exitPx);
        double pnl = (exitPx - pos.getEntryPrice()) * pos.getSize() * pos.dir() - fee;
        equity += pnl;

        TradeRecord r = new TradeRecord(symbol, pos.getSide(), pos.getEntryTime(), ts,
                pos.getEntryPrice(), exitPx, pos.getSize(), pnl);
        tradeSink.accept(r);
        position = null;
        alert("청산[" + reason + "] PnL " + fmt(pnl) + " / equity " + fmt(equity));
        for (EngineListener l : listeners) { l.onTrade(r); l.onPosition(null); }
    }

    private void setState(EngineState s) {
        state = s;
        for (EngineListener l : listeners) l.onState(s);
    }

    private void alert(String msg) {
        for (EngineListener l : listeners) l.onAlert(msg);
    }

    private static String fmt(double v) { return String.format("%,.2f", v); }
}

package ats.domain;

/** 현재 보유 포지션 (Design 2.2 데이터 클래스) */
public class Position {
    private final String symbol;
    private final OrderSide side;          // LONG=BUY, SHORT=SELL
    private final double size;
    private final double entryPrice;
    private final int leverage;
    private long entryTime;
    private double unrealizedPnl;

    public Position(String symbol, OrderSide side, double size, double entryPrice, int leverage) {
        this.symbol = symbol; this.side = side; this.size = size;
        this.entryPrice = entryPrice; this.leverage = leverage;
    }

    /** 현재가 기준 미실현 손익 갱신 */
    public void updateMark(double markPrice) {
        unrealizedPnl = (markPrice - entryPrice) * size * dir();
    }

    /** 주문 반영 (동일 방향=증량 개념, 반대 방향=감량) — 본 구현에선 엔진이 전량 단위로 관리 */
    public void update(Order o) {
        if (o.getSide() != side) unrealizedPnl = 0;
    }

    public int dir() { return side == OrderSide.BUY ? 1 : -1; }

    public String getSymbol()        { return symbol; }
    public OrderSide getSide()       { return side; }
    public double getSize()          { return size; }
    public double getEntryPrice()    { return entryPrice; }
    public int getLeverage()         { return leverage; }
    public double getUnrealizedPnl() { return unrealizedPnl; }
    public long getEntryTime()       { return entryTime; }
    public void setEntryTime(long t) { this.entryTime = t; }
}

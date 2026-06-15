package ats.domain;

/** 체결 완료된 거래 1건 (Design 2.2 데이터 클래스) */
public record TradeRecord(String symbol, OrderSide side,
                          long entryTime, long exitTime,
                          double entryPrice, double exitPrice,
                          double quantity, double realizedPnl) {

    public String toCsv() {
        return symbol + "," + side + "," + entryTime + "," + exitTime + ","
                + entryPrice + "," + exitPrice + "," + quantity + "," + realizedPnl;
    }

    public static TradeRecord fromCsv(String line) {
        String[] t = line.split(",");
        return new TradeRecord(t[0], OrderSide.valueOf(t[1]),
                Long.parseLong(t[2]), Long.parseLong(t[3]),
                Double.parseDouble(t[4]), Double.parseDouble(t[5]),
                Double.parseDouble(t[6]), Double.parseDouble(t[7]));
    }
}

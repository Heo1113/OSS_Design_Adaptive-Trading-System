package ats.domain;

/** 단일 주문. 상태 전이는 Design 4.2 State machine을 따름 */
public class Order {
    private final String orderId;
    private final OrderSide side;
    private final OrderType type;
    private final double quantity;
    private double price;            // 요청가 → 체결 시 평균 체결가로 갱신
    private OrderStatus status;

    public Order(OrderSide side, OrderType type, double quantity, double price) {
        this.orderId = "ORD-" + System.nanoTime();
        this.side = side;
        this.type = type;
        this.quantity = quantity;
        this.price = price;
        this.status = OrderStatus.NEW;
    }

    /** NEW → SUBMITTED */
    public void submit() { if (status == OrderStatus.NEW) status = OrderStatus.SUBMITTED; }

    /** SUBMITTED/PARTIALLY_FILLED → FILLED (체결가 기록) */
    public void fill(double avgPrice) { this.price = avgPrice; this.status = OrderStatus.FILLED; }

    /** → CANCELED */
    public void cancel() { this.status = OrderStatus.CANCELED; }

    public String getOrderId()    { return orderId; }
    public OrderSide getSide()    { return side; }
    public OrderType getType()    { return type; }
    public double getQuantity()   { return quantity; }
    public double getPrice()      { return price; }
    public OrderStatus getStatus(){ return status; }
}

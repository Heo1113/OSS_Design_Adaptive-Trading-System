package ats.infrastructure;

import ats.domain.MarketData;
import ats.domain.MarketDataListener;
import ats.domain.Order;
import ats.domain.OrderSide;
import ats.domain.TradingEngine;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.net.http.WebSocket;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.CompletionStage;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

/**
 * 바이낸스 선물 거래소 게이트웨이 (Design 2.2 BinanceConnector).
 * - LIVE  : API 키 설정 시 REST(/fapi)·WebSocket(kline_1m) 실연동 — JDK java.net.http만 사용
 * - DEMO  : 키 미설정 시 합성 시세(regime-switching random walk)로 전체 기능 시연 가능
 * 외부 라이브러리 없이 동작하며, 모드는 자격증명 유무로 동적으로 결정됨.
 */
public class BinanceConnector implements TradingEngine.ConnectorPort {

    private static final long CANDLE_MS = 60_000;

    private final APICredentialManager credential;
    private final HttpClient http = HttpClient.newHttpClient();

    private volatile WebSocket ws;
    private volatile ScheduledExecutorService demoFeed;
    private volatile MarketDataListener listener;
    private volatile String activeSymbol = "BTCUSDT";
    private volatile double lastPrice = 65_000;

    // DEMO 시세 시뮬레이션 상태
    private final Random rnd = new Random();
    private double simPrice = 65_000;
    private int simDrift = 1;
    private long simTime = System.currentTimeMillis();

    public BinanceConnector(APICredentialManager credential) {
        this.credential = credential;
    }

    /** 자격증명 미설정 → DEMO (동적 판정: 키 저장 후 엔진 재시작 시 LIVE 전환) */
    @Override
    public boolean isDemo() { return !credential.isConfigured(); }

    public double getLastPrice() { return lastPrice; }

    private String restBase() {
        return credential.isTestnet() ? "https://testnet.binancefuture.com"
                                      : "https://fapi.binance.com";
    }

    private String wsBase() {
        return credential.isTestnet() ? "wss://stream.binancefuture.com/ws"
                                      : "wss://fstream.binance.com/ws";
    }

    // ------------------------------------------------------------------
    // fetchHistorical (Design Seq 3.3)
    // ------------------------------------------------------------------

    @Override
    public List<MarketData> fetchHistorical(String symbol, String interval, int limit) throws Exception {
        activeSymbol = symbol;
        if (isDemo()) return generateSeries(limit, 42);

        String url = restBase() + "/fapi/v1/klines?symbol=" + symbol
                + "&interval=" + interval + "&limit=" + limit;
        HttpResponse<String> resp = http.send(
                HttpRequest.newBuilder(URI.create(url)).GET().build(),
                HttpResponse.BodyHandlers.ofString());
        if (resp.statusCode() != 200)
            throw new IllegalStateException("klines HTTP " + resp.statusCode());

        List<?> rows = (List<?>) MiniJson.parse(resp.body());
        List<MarketData> out = new ArrayList<>(rows.size());
        for (Object o : rows) {
            List<?> r = (List<?>) o;
            out.add(new MarketData(
                    ((Number) r.get(0)).longValue(),
                    Double.parseDouble((String) r.get(1)),
                    Double.parseDouble((String) r.get(2)),
                    Double.parseDouble((String) r.get(3)),
                    Double.parseDouble((String) r.get(4)),
                    Double.parseDouble((String) r.get(5))));
        }
        if (!out.isEmpty()) lastPrice = out.get(out.size() - 1).close();
        return out;
    }

    // ------------------------------------------------------------------
    // openStream / closeStream (Design Seq 3.1)
    // ------------------------------------------------------------------

    @Override
    public void openStream(String symbol, MarketDataListener l) throws Exception {
        activeSymbol = symbol;
        this.listener = l;

        if (isDemo()) {
            demoFeed = Executors.newSingleThreadScheduledExecutor(r -> {
                Thread t = new Thread(r, "ats-demo-feed");
                t.setDaemon(true);
                return t;
            });
            // 데모: 0.7초마다 1분봉 1개 생성(가속 시뮬레이션)
            demoFeed.scheduleAtFixedRate(() -> {
                MarketData d = nextCandle();
                lastPrice = d.close();
                MarketDataListener li = listener;
                if (li != null) li.onMarketData(d);
            }, 700, 700, TimeUnit.MILLISECONDS);
            return;
        }

        String url = wsBase() + "/" + symbol.toLowerCase() + "@kline_1m";
        StringBuilder buf = new StringBuilder();
        ws = http.newWebSocketBuilder().buildAsync(URI.create(url), new WebSocket.Listener() {
            @Override
            public CompletionStage<?> onText(WebSocket w, CharSequence data, boolean last) {
                buf.append(data);
                if (last) { handleWsMessage(buf.toString()); buf.setLength(0); }
                w.request(1);
                return null;
            }
        }).join();
    }

    private void handleWsMessage(String json) {
        try {
            Object parsed = MiniJson.parse(json);
            if (!(parsed instanceof Map)) return;
            Object kObj = ((Map<?, ?>) parsed).get("k");
            if (!(kObj instanceof Map)) return;
            Map<?, ?> k = (Map<?, ?>) kObj;
            if (!Boolean.TRUE.equals(k.get("x"))) return;          // 캔들 마감 이벤트만
            MarketData d = new MarketData(
                    ((Number) k.get("t")).longValue(),
                    Double.parseDouble((String) k.get("o")),
                    Double.parseDouble((String) k.get("h")),
                    Double.parseDouble((String) k.get("l")),
                    Double.parseDouble((String) k.get("c")),
                    Double.parseDouble((String) k.get("v")));
            lastPrice = d.close();
            MarketDataListener li = listener;
            if (li != null) li.onMarketData(d);
        } catch (Exception ignored) {
            // 스트림 파싱 오류는 개별 메시지 단위로 무시 (연결 유지)
        }
    }

    @Override
    public void closeStream() {
        if (demoFeed != null) { demoFeed.shutdownNow(); demoFeed = null; }
        WebSocket w = ws;
        if (w != null) {
            try { w.sendClose(WebSocket.NORMAL_CLOSURE, "bye"); } catch (Exception ignored) { }
            ws = null;
        }
        listener = null;
    }

    // ------------------------------------------------------------------
    // placeOrder (Design Seq 3.4)
    // ------------------------------------------------------------------

    @Override
    public Map<String, Object> placeOrder(Order order) {
        Map<String, Object> resp = new HashMap<>();
        if (isDemo()) {
            double slip = lastPrice * 0.0003 * rnd.nextDouble();   // 슬리피지 시뮬레이션
            double fill = order.getSide() == OrderSide.BUY ? lastPrice + slip : lastPrice - slip;
            order.fill(fill);
            resp.put("status", "FILLED");
            resp.put("avgPrice", fill);
            resp.put("orderId", order.getOrderId());
            return resp;
        }
        try {
            long ts = System.currentTimeMillis();
            String q = "symbol=" + activeSymbol + "&side=" + order.getSide()
                    + "&type=MARKET&quantity=" + String.format("%.3f", order.getQuantity())
                    + "&timestamp=" + ts;
            String url = restBase() + "/fapi/v1/order?" + q + "&signature=" + credential.sign(q);
            HttpResponse<String> r = http.send(
                    HttpRequest.newBuilder(URI.create(url))
                            .header("X-MBX-APIKEY", credential.getApiKey())
                            .POST(HttpRequest.BodyPublishers.noBody()).build(),
                    HttpResponse.BodyHandlers.ofString());
            if (r.statusCode() == 200) {
                Map<?, ?> body = (Map<?, ?>) MiniJson.parse(r.body());
                double fill = lastPrice;
                Object ap = body.get("avgPrice");
                if (ap != null) {
                    try {
                        double v = Double.parseDouble(String.valueOf(ap));
                        if (v > 0) fill = v;
                    } catch (Exception ignored) { }
                }
                order.fill(fill);
                resp.put("status", "FILLED");
                resp.put("avgPrice", fill);
                resp.put("orderId", String.valueOf(body.get("orderId")));
            } else {
                order.cancel();
                resp.put("status", "ERROR");
                resp.put("msg", "HTTP " + r.statusCode() + " " + r.body());
            }
        } catch (Exception e) {
            order.cancel();
            resp.put("status", "ERROR");
            resp.put("msg", String.valueOf(e));
        }
        return resp;
    }

    // ------------------------------------------------------------------
    // validateCredential (Design Seq 3.2, «include»)
    // ------------------------------------------------------------------

    /** DEMO 모드는 항상 true, LIVE는 서명 요청으로 계좌 조회 성공 여부 확인 */
    public boolean validateCredential() {
        if (isDemo()) return true;
        try {
            long ts = System.currentTimeMillis();
            String q = "timestamp=" + ts;
            String url = restBase() + "/fapi/v2/account?" + q + "&signature=" + credential.sign(q);
            HttpResponse<String> r = http.send(
                    HttpRequest.newBuilder(URI.create(url))
                            .header("X-MBX-APIKEY", credential.getApiKey()).GET().build(),
                    HttpResponse.BodyHandlers.ofString());
            return r.statusCode() == 200;
        } catch (Exception e) {
            return false;
        }
    }

    // ------------------------------------------------------------------
    // DEMO 합성 시세 (추세 전환이 있는 random walk → GA가 유의미하게 학습 가능)
    // ------------------------------------------------------------------

    private MarketData nextCandle() {
        if (rnd.nextDouble() < 0.02) simDrift = -simDrift;          // 2% 확률로 추세 전환
        double ret = simDrift * 0.0005 + rnd.nextGaussian() * 0.0035;
        double open = simPrice;
        simPrice = Math.max(1_000, simPrice * Math.exp(ret));
        double hi = Math.max(open, simPrice) * (1 + rnd.nextDouble() * 0.0008);
        double lo = Math.min(open, simPrice) * (1 - rnd.nextDouble() * 0.0008);
        simTime += CANDLE_MS;
        return new MarketData(simTime, open, hi, lo, simPrice, 50 + rnd.nextDouble() * 150);
    }

    private List<MarketData> generateSeries(int limit, long seed) {
        Random r = new Random(seed);                                // 재현 가능한 학습 데이터
        List<MarketData> out = new ArrayList<>(limit);
        double p = 60_000;
        int drift = 1;
        long t = System.currentTimeMillis() - (long) limit * CANDLE_MS;
        for (int i = 0; i < limit; i++) {
            if (r.nextDouble() < 0.02) drift = -drift;
            double ret = drift * 0.0005 + r.nextGaussian() * 0.0035;
            double open = p;
            p = Math.max(1_000, p * Math.exp(ret));
            double hi = Math.max(open, p) * (1 + r.nextDouble() * 0.0008);
            double lo = Math.min(open, p) * (1 - r.nextDouble() * 0.0008);
            out.add(new MarketData(t, open, hi, lo, p, 50 + r.nextDouble() * 150));
            t += CANDLE_MS;
        }
        simPrice = p;
        lastPrice = p;
        simTime = System.currentTimeMillis();
        return out;
    }
}

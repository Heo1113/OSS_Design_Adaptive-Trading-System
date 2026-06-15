package ats;

import ats.domain.Backtester;
import ats.domain.GAOptimizer;
import ats.domain.HistoricalDataset;
import ats.domain.MarketData;
import ats.domain.Pair;
import ats.domain.PerformanceReport;
import ats.domain.WalkForwardValidator;
import ats.infrastructure.APICredentialManager;
import ats.infrastructure.BinanceConnector;
import ats.infrastructure.MiniJson;

import java.util.List;
import java.util.Map;

/** GUI 없이 핵심 파이프라인(데이터→GA→WFO→리포트)을 검증하는 콘솔 데모 */
final class ConsoleDemo {
    private ConsoleDemo() { }

    static void run() {
        System.out.println("=== ATS Console Demo (DEMO simulation mode) ===");

        // [0] JSON 파서 자가 점검 (Binance kline 메시지 형식)
        Map<?, ?> j = (Map<?, ?>) MiniJson.parse(
                "{\"e\":\"kline\",\"k\":{\"t\":1718000000000,\"o\":\"65000.1\","
                        + "\"c\":\"65100.5\",\"x\":true}}");
        Map<?, ?> k = (Map<?, ?>) j.get("k");
        if (!"65100.5".equals(k.get("c")))
            throw new IllegalStateException("MiniJson self-test failed");
        System.out.println("[0] MiniJson self-test OK");

        // [1] 과거 데이터 (DEMO: 합성 시세, 고정 seed → 재현 가능)
        APICredentialManager cred = new APICredentialManager();
        BinanceConnector con = new BinanceConnector(cred);
        List<MarketData> data;
        try {
            data = con.fetchHistorical("BTCUSDT", "1m", 3000);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        System.out.println("[1] historical candles: " + data.size());

        HistoricalDataset ds = new HistoricalDataset(data);
        Pair<HistoricalDataset, HistoricalDataset> sp = ds.split(0.7);

        // [2] GA 최적화
        GAOptimizer ga = new GAOptimizer(40, 30, 0.10, sp.first());
        ga.setListener((g, t, f) -> {
            if (g == 1 || g % 5 == 0)
                System.out.printf("    gen %2d/%d  best fitness %.4f%n", g, t, f);
        });
        long t0 = System.currentTimeMillis();
        Map<String, Object> best = ga.run();
        System.out.printf("[2] GA done in %.1fs%n    best params: %s%n",
                (System.currentTimeMillis() - t0) / 1000.0, best);

        // [3] 학습/검증 구간 성과
        Backtester.Result tr = Backtester.run(best, sp.first().getData(), 10_000, 0.10, 5);
        Backtester.Result te = Backtester.run(best, sp.second().getData(), 10_000, 0.10, 5);
        System.out.printf("[3] train ROI %+.2f%%  MDD %.2f%%  trades %d  win %.0f%%%n",
                tr.roi() * 100, tr.mdd() * 100, tr.trades(), tr.winRate() * 100);
        System.out.printf("    test  ROI %+.2f%%  MDD %.2f%%  trades %d  win %.0f%%%n",
                te.roi() * 100, te.mdd() * 100, te.trades(), te.winRate() * 100);

        // [4] Walk-Forward 검증
        Map<String, Object> wfo = new WalkForwardValidator(4, ds).validate(best);
        double is = (Double) wfo.get("inSampleRoi");
        double oos = (Double) wfo.get("outSampleRoi");
        double ratio = (Double) wfo.get("overfitRatio");
        System.out.printf("[4] WFO  IS %+.2f%%  OOS %+.2f%%  ratio %.2f (%s)%n",
                is * 100, oos * 100, ratio, ratio >= 0.5 ? "OK" : "overfit?");

        // [5] 성과 리포트
        PerformanceReport rep = new PerformanceReport(te.records());
        System.out.printf("[5] report  ROI %+.2f%%  MDD %.2f%%  win %.0f%%  "
                        + "sharpe %.2f  curve points %d%n",
                rep.roi() * 100, rep.maxDrawdown() * 100, rep.winRate() * 100,
                rep.sharpeRatio(), rep.equityCurve().size());

        System.out.println("=== Demo finished successfully ===");
    }
}

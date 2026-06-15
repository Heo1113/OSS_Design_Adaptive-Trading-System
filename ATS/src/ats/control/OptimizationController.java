package ats.control;

import ats.domain.Backtester;
import ats.domain.GAOptimizer;
import ats.domain.HistoricalDataset;
import ats.domain.MarketData;
import ats.domain.Pair;
import ats.domain.Strategy;
import ats.domain.WalkForwardValidator;
import ats.infrastructure.BinanceConnector;
import ats.infrastructure.HistoricalDataDAO;

import java.util.List;
import java.util.Map;
import java.util.function.BiConsumer;
import java.util.function.Consumer;

/**
 * GA 최적화 use case 조율 (Design Seq 3.3).
 * 데이터 수집 → GA 실행 → (선택) WFO 검증 → 전략 반영.
 */
public class OptimizationController {
    private final AppConfig cfg;
    private final BinanceConnector connector;
    private final Strategy strategy;
    private final HistoricalDataDAO histDao = new HistoricalDataDAO();

    private volatile List<Double> lastHistory = List.of();
    private volatile Map<String, Object> lastBest;
    private volatile Backtester.Result lastTrain, lastTest;

    public OptimizationController(AppConfig cfg, BinanceConnector connector, Strategy strategy) {
        this.cfg = cfg;
        this.connector = connector;
        this.strategy = strategy;
    }

    /** 백그라운드 스레드에서 최적화 실행. wfo는 비활성 시 null */
    public void optimize(GAOptimizer.Listener progress,
                         BiConsumer<Map<String, Object>, Map<String, Object>> onDone,
                         Consumer<String> onError) {
        Thread t = new Thread(() -> {
            try {
                List<MarketData> data = connector.fetchHistorical(cfg.symbol, "1m", 3000);
                if (data.size() < 500)
                    throw new IllegalStateException("과거 데이터 부족: " + data.size() + "개");
                histDao.save(cfg.symbol, data);

                HistoricalDataset ds = new HistoricalDataset(data);
                Pair<HistoricalDataset, HistoricalDataset> sp = ds.split(0.7);

                GAOptimizer ga = new GAOptimizer(
                        cfg.populationSize, cfg.generations, cfg.mutationRate, sp.first());
                ga.setListener(progress);
                Map<String, Object> best = ga.run();

                lastHistory = ga.getBestHistory();
                lastBest = best;
                lastTrain = Backtester.run(best, sp.first().getData(), 10_000, cfg.riskPct, cfg.leverage);
                lastTest  = Backtester.run(best, sp.second().getData(), 10_000, cfg.riskPct, cfg.leverage);

                Map<String, Object> wfo = null;
                if (cfg.useWfo) wfo = new WalkForwardValidator(4, ds).validate(best);

                strategy.updateParams(best);
                onDone.accept(best, wfo);
            } catch (Exception e) {
                onError.accept(String.valueOf(e.getMessage()));
            }
        }, "ga-optimizer");
        t.setDaemon(true);
        t.start();
    }

    public List<Double> lastHistory()        { return lastHistory; }
    public Map<String, Object> lastBest()    { return lastBest; }
    public Backtester.Result lastTrain()     { return lastTrain; }
    public Backtester.Result lastTest()      { return lastTest; }
}

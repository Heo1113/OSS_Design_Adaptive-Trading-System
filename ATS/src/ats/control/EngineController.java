package ats.control;

import ats.domain.Strategy;
import ats.domain.TradingEngine;
import ats.infrastructure.APICredentialManager;
import ats.infrastructure.BinanceConnector;
import ats.infrastructure.TradeLogDAO;

/**
 * 엔진 가동/정지/설정 use case 조율 (Design 2.2 Control 계층, Seq 3.1·3.2).
 * View → Controller → Domain/Infrastructure 단방향 의존.
 */
public class EngineController {
    private final AppConfig cfg;
    private final APICredentialManager credential;
    private final BinanceConnector connector;
    private final Strategy strategy;
    private final TradingEngine engine;
    private final TradeLogDAO tradeLog;

    public EngineController(AppConfig cfg) {
        this.cfg = cfg;
        this.credential = new APICredentialManager();
        credential.load();                                  // .env 없으면 DEMO
        this.connector = new BinanceConnector(credential);
        this.strategy = new Strategy(null, cfg.symbol);
        this.tradeLog = new TradeLogDAO();
        this.engine = new TradingEngine(strategy, connector, tradeLog::save);
        engine.configure(cfg.symbol, cfg.riskPct, cfg.leverage);
    }

    public boolean start() {
        engine.configure(cfg.symbol, cfg.riskPct, cfg.leverage);
        return engine.start();
    }

    public boolean stop() { return engine.stop(); }

    /**
     * 설정 저장 + 자격증명 검증 (Seq 3.2, «include» Validate API Credentials).
     * @return null=성공, 그 외 오류 메시지
     */
    public String saveConfig(String apiKey, String apiSecret, boolean testnet,
                             String symbol, int leverage, double riskPct,
                             int populationSize, int generations, double mutationRate,
                             boolean useWfo) {
        cfg.symbol = symbol; cfg.leverage = leverage; cfg.riskPct = riskPct;
        cfg.populationSize = populationSize; cfg.generations = generations;
        cfg.mutationRate = mutationRate; cfg.useWfo = useWfo;
        cfg.save();
        engine.configure(symbol, riskPct, leverage);

        if (apiKey != null && !apiKey.isBlank()) {
            credential.set(apiKey, apiSecret, testnet);
            if (!connector.validateCredential()) {
                credential.set("", "", testnet);            // 롤백 → DEMO 유지
                return "API 자격증명 검증 실패 — DEMO 모드 유지";
            }
            try { credential.save(); }
            catch (Exception e) { return ".env 저장 실패: " + e.getMessage(); }
        }
        return null;
    }

    /** 키 유효성만 검사 (저장하지 않음). 키가 비어 있으면 DEMO로 간주 → true */
    public boolean testCredential(String key, String secret, boolean tn) {
        if (key == null || key.isBlank()) return true;
        String pk = credential.getApiKey(), ps = credential.getApiSecret();
        boolean pt = credential.isTestnet();
        credential.set(key, secret, tn);
        boolean ok = connector.validateCredential();
        credential.set(pk, ps, pt);
        return ok;
    }

    public TradingEngine engine()        { return engine; }
    public BinanceConnector connector()  { return connector; }
    public Strategy strategy()           { return strategy; }
    public TradeLogDAO tradeLog()        { return tradeLog; }
    public AppConfig config()            { return cfg; }
}

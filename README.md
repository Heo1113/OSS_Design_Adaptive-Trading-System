# ATS — Adaptive Trading System

바이낸스 선물 시장을 대상으로 **유전 알고리즘(GA)** 으로 매매 파라미터를 스스로
최적화하고, **Walk-Forward Optimization(WFO)** 으로 과적합을 검증한 뒤
실시간 자동매매를 수행하는 데스크톱 애플리케이션임.

Conceptualization → Analysis → Design 문서에서 정의한 클래스 구조를
그대로 Java로 구현함.

## 요구사항

* **JDK 17 이상** — 이것이 전부임.
* 외부 라이브러리 **0개**. 거래소 통신(`java.net.http`), 요청 서명(`javax.crypto`),
  GUI(Swing), JSON 파서, 기술적 지표(SMA/RSI), 유전 알고리즘 모두
  JDK 표준 라이브러리 위에서 직접 구현하여 환경 독립적으로 동작함.

## 빌드

```bash
# Linux / macOS
./build.sh

# Windows
build.bat
```

성공 시 실행 가능한 `ATS.jar` 가 생성됨.

## 실행

```bash
# GUI (기본)
java -jar ATS.jar

# 콘솔 데모 — 데이터 수집 → GA 최적화 → WFO 검증 → 성과 리포트를
# 터미널에서 한 번에 검증 (헤드리스 환경용)
java -jar ATS.jar --cli
```

## DEMO 모드 vs LIVE 모드

| 모드 | 조건 | 동작 |
| --- | --- | --- |
| **DEMO** | API 키 미설정 (기본) | 추세 전환이 있는 합성 시세(regime-switching random walk)로 전체 기능 시연. 주문은 슬리피지·수수료가 반영된 내부 체결로 처리 |
| **LIVE** | `.env`에 키 설정 | Binance Futures REST/WebSocket 실연동. `BINANCE_TESTNET=true`면 테스트넷(가상 자금) |

LIVE 사용 시 `.env.example`을 `.env`로 복사해 키를 입력하거나,
GUI의 **Configuration** 탭에서 입력 후 Save(검증 통과 시 `.env`에 저장됨).

## 화면 구성

`Dashboard`(엔진 Start/Stop·실시간 캔들·포지션·알림) /
`Configuration`(API·거래·GA 파라미터) /
`Optimization`(GA 실행·적합도 수렴 그래프·WFO 결과) /
`Performance`(ROI·MDD·승률·Sharpe·자본곡선·주별 손익) /
`History`(거래 내역 테이블·CSV 내보내기)

## 프로젝트 구조 (Design 문서 매핑)

```
src/ats/
├── Main.java, ConsoleDemo.java
├── domain/            # Design 2.1 Domain 계층
│   ├── TradingEngine, Strategy, GAOptimizer, WalkForwardValidator
│   ├── Backtester, Indicators, PerformanceReport, HistoricalDataset
│   ├── Order, Position, MarketData, TradeRecord, Pair
│   └── EngineState, Signal, OrderSide, OrderType, OrderStatus (enums)
├── infrastructure/    # Design 2.1 Infrastructure 계층
│   ├── BinanceConnector, APICredentialManager
│   ├── TradeLogDAO, HistoricalDataDAO, MiniJson
├── control/           # Design 2.1 Control 계층
│   ├── EngineController, OptimizationController, ReportController, AppConfig
└── presentation/      # Design 2.1 Presentation 계층
    ├── MainFrame, DashboardView, ConfigurationView
    ├── OptimizationView, ReportView, HistoryView
    └── UiTheme, CandlePanel, LinePanel, BarPanel
```

상태 머신(Design 4장)은 `EngineState`(STOPPED↔RUNNING↔ERROR)와
`OrderStatus`(NEW→SUBMITTED→…→FILLED/CANCELED) 전이로 구현됨.

## GitHub에 올리기

```bash
git init -b main
git add .
git commit -m "Implement ATS"
git remote add origin https://github.com/<계정>/OSS_Design_Adaptive-Trading-System.git
git push -u origin main
```

## 라이선스

GPL-3.0 — `LICENSE` 참조.

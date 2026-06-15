package ats.presentation;

import ats.control.AppConfig;
import ats.control.EngineController;
import ats.control.OptimizationController;
import ats.control.ReportController;
import ats.domain.EngineState;
import ats.domain.TradingEngine;

import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JTabbedPane;
import javax.swing.SwingUtilities;
import java.awt.BorderLayout;

/** 메인 윈도우. View 5개 탭 + 상태바. (Design 2.1 Presentation 계층 조립) */
public class MainFrame extends JFrame {

    public MainFrame() {
        super("ATS — Adaptive Trading System");

        AppConfig cfg = new AppConfig();
        cfg.load();
        EngineController ec = new EngineController(cfg);
        OptimizationController oc = new OptimizationController(cfg, ec.connector(), ec.strategy());
        ReportController rc = new ReportController(ec.tradeLog());

        DashboardView dashboard = new DashboardView(ec);
        ConfigurationView config = new ConfigurationView(ec);
        OptimizationView optimization = new OptimizationView(oc);
        ReportView report = new ReportView(rc);
        HistoryView history = new HistoryView(rc);

        JTabbedPane tabs = new JTabbedPane();
        tabs.addTab("Dashboard", dashboard);
        tabs.addTab("Configuration", config);
        tabs.addTab("Optimization", optimization);
        tabs.addTab("Performance", report);
        tabs.addTab("History", history);
        tabs.addChangeListener(e -> {
            if (tabs.getSelectedComponent() == report) report.refresh();
            if (tabs.getSelectedComponent() == history) history.refresh();
        });

        JLabel status = new JLabel("  mode: "
                + (ec.connector().isDemo() ? "DEMO (no API key — synthetic feed)" : "LIVE"));
        status.setForeground(UiTheme.SUB);

        setLayout(new BorderLayout());
        add(tabs, BorderLayout.CENTER);
        add(status, BorderLayout.SOUTH);

        ec.engine().addListener(new TradingEngine.EngineListener() {
            @Override public void onState(EngineState s) {
                SwingUtilities.invokeLater(() -> status.setText(
                        "  mode: " + (ec.connector().isDemo() ? "DEMO" : "LIVE")
                                + "   |   engine: " + s));
            }
        });

        setDefaultCloseOperation(EXIT_ON_CLOSE);
        addWindowListener(new java.awt.event.WindowAdapter() {
            @Override public void windowClosing(java.awt.event.WindowEvent e) { ec.stop(); }
        });
        setSize(1280, 800);
        setLocationRelativeTo(null);

        // 데모/스크린샷 편의 옵션: -Dats.tab=N, -Dats.autostart=true
        int tab = Integer.getInteger("ats.tab", 0);
        if (tab >= 0 && tab < tabs.getTabCount()) tabs.setSelectedIndex(tab);
        if (Boolean.getBoolean("ats.autostart")) {
            new Thread(() -> {
                ec.start();
                SwingUtilities.invokeLater(dashboard::refreshChartFromEngine);
            }, "autostart").start();
        }
    }
}

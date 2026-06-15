package ats.presentation;

import ats.control.EngineController;
import ats.domain.EngineState;
import ats.domain.MarketData;
import ats.domain.OrderSide;
import ats.domain.Position;
import ats.domain.TradeRecord;
import ats.domain.TradingEngine;

import javax.swing.BorderFactory;
import javax.swing.Box;
import javax.swing.BoxLayout;
import javax.swing.DefaultListModel;
import javax.swing.JButton;
import javax.swing.JLabel;
import javax.swing.JList;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.SwingUtilities;
import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.FlowLayout;
import java.awt.Font;
import java.text.SimpleDateFormat;
import java.util.Date;

/** 메인 대시보드 (Analysis UI prototype #2 Main Dashboard 구현) */
public class DashboardView extends JPanel {

    private final EngineController ec;
    private final JButton startStop = new JButton("Start engine");
    private final JLabel statePill = new JLabel(" STOPPED ");
    private final JLabel priceLabel = new JLabel("-");
    private final JLabel equityLabel = new JLabel("");
    private final CandlePanel chart = new CandlePanel();
    private final JLabel posText = new JLabel("<html><i>No open position</i></html>");
    private final DefaultListModel<String> alerts = new DefaultListModel<>();
    private final SimpleDateFormat tf = new SimpleDateFormat("HH:mm:ss");

    public DashboardView(EngineController ec) {
        this.ec = ec;
        setLayout(new BorderLayout(10, 10));
        setBackground(UiTheme.BG);
        setBorder(BorderFactory.createEmptyBorder(12, 12, 12, 12));

        // ── 상단 바: Start/Stop + 상태 + 현재가 + equity
        JPanel top = new JPanel(new FlowLayout(FlowLayout.LEFT, 10, 0));
        top.setOpaque(false);
        statePill.setOpaque(true);
        stylePill(EngineState.STOPPED);
        priceLabel.setFont(priceLabel.getFont().deriveFont(Font.BOLD, 16f));
        equityLabel.setForeground(UiTheme.SUB);
        top.add(startStop);
        top.add(statePill);
        top.add(Box.createHorizontalStrut(16));
        top.add(priceLabel);
        top.add(equityLabel);
        add(top, BorderLayout.NORTH);

        // ── 중앙: 실시간 차트 카드
        JPanel chartCard = UiTheme.card();
        chartCard.setLayout(new BorderLayout(0, 6));
        chartCard.add(UiTheme.title("Realtime chart (1m)"), BorderLayout.NORTH);
        chartCard.add(chart, BorderLayout.CENTER);
        add(chartCard, BorderLayout.CENTER);

        // ── 우측: 포지션 + 알림
        JPanel east = new JPanel();
        east.setOpaque(false);
        east.setLayout(new BoxLayout(east, BoxLayout.Y_AXIS));

        JPanel posCard = UiTheme.card();
        posCard.setLayout(new BorderLayout(0, 6));
        posCard.add(UiTheme.title("Current position"), BorderLayout.NORTH);
        posCard.add(posText, BorderLayout.CENTER);
        posCard.setPreferredSize(new Dimension(280, 150));
        posCard.setMaximumSize(new Dimension(280, 150));

        JPanel alertCard = UiTheme.card();
        alertCard.setLayout(new BorderLayout(0, 6));
        alertCard.add(UiTheme.title("Alerts"), BorderLayout.NORTH);
        JList<String> list = new JList<>(alerts);
        list.setFont(list.getFont().deriveFont(11f));
        alertCard.add(new JScrollPane(list), BorderLayout.CENTER);
        alertCard.setPreferredSize(new Dimension(280, 360));

        east.add(posCard);
        east.add(Box.createVerticalStrut(10));
        east.add(alertCard);
        add(east, BorderLayout.EAST);

        // ── 동작
        startStop.addActionListener(e -> toggle());

        ec.engine().addListener(new TradingEngine.EngineListener() {
            @Override public void onState(EngineState s) {
                SwingUtilities.invokeLater(() -> stylePill(s));
            }
            @Override public void onCandle(MarketData d) {
                SwingUtilities.invokeLater(() -> {
                    chart.add(d);
                    priceLabel.setText(String.format("%,.2f", d.close()));
                });
            }
            @Override public void onPosition(Position p) {
                SwingUtilities.invokeLater(() -> updatePos(p));
            }
            @Override public void onTrade(TradeRecord r) {
                SwingUtilities.invokeLater(() ->
                        equityLabel.setText("   equity " + UiTheme.fmt(ec.engine().getEquity())));
            }
            @Override public void onAlert(String msg) {
                SwingUtilities.invokeLater(() -> {
                    alerts.add(0, "[" + tf.format(new Date()) + "] " + msg);
                    while (alerts.size() > 60) alerts.remove(alerts.size() - 1);
                });
            }
        });
    }

    /** MainFrame autostart 등에서 워밍업 캔들을 차트에 반영 */
    public void refreshChartFromEngine() {
        chart.setAll(ec.engine().snapshotWindow());
    }

    private void toggle() {
        startStop.setEnabled(false);
        boolean running = ec.engine().getState() == EngineState.RUNNING;
        new Thread(() -> {
            if (running) {
                ec.stop();
            } else {
                ec.start();
                SwingUtilities.invokeLater(this::refreshChartFromEngine);
            }
            SwingUtilities.invokeLater(() -> startStop.setEnabled(true));
        }, "engine-toggle").start();
    }

    private void stylePill(EngineState s) {
        statePill.setText(" " + s + " ");
        switch (s) {
            case RUNNING -> {
                statePill.setBackground(new Color(0xE1F5EE));
                statePill.setForeground(new Color(0x0F6E56));
                startStop.setText("Stop engine");
            }
            case ERROR -> {
                statePill.setBackground(new Color(0xFCEBEB));
                statePill.setForeground(new Color(0xA32D2D));
                startStop.setText("Start engine");
            }
            default -> {
                statePill.setBackground(new Color(0xFAEEDA));
                statePill.setForeground(new Color(0x854F0B));
                startStop.setText("Start engine");
            }
        }
    }

    private void updatePos(Position p) {
        if (p == null) {
            posText.setText("<html><i>No open position</i></html>");
            return;
        }
        String pnlColor = p.getUnrealizedPnl() >= 0 ? "#1D9E75" : "#E24B4A";
        posText.setText(String.format(
                "<html>%s <b>%s</b> x%d<br>size %.3f<br>entry %,.2f<br>"
                        + "uPnL <font color='%s'>%,.2f</font></html>",
                p.getSymbol(), p.getSide() == OrderSide.BUY ? "LONG" : "SHORT",
                p.getLeverage(), p.getSize(), p.getEntryPrice(),
                pnlColor, p.getUnrealizedPnl()));
    }
}

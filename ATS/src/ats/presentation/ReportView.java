package ats.presentation;

import ats.control.ReportController;
import ats.domain.PerformanceReport;
import ats.domain.TradeRecord;

import javax.swing.BorderFactory;
import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.SwingConstants;
import javax.swing.SwingUtilities;
import java.awt.BorderLayout;
import java.awt.FlowLayout;
import java.awt.Font;
import java.awt.GridLayout;
import java.util.List;

/** 성과 리포트 화면 (Analysis UI prototype #4 구현, Design Seq 3.5) */
public class ReportView extends JPanel {

    private final ReportController rc;
    private final JComboBox<String> period =
            new JComboBox<>(new String[]{"All time", "Last 30 days", "Last 7 days"});
    private final JLabel kRoi = kpiVal(), kMdd = kpiVal(), kWin = kpiVal(),
            kSharpe = kpiVal(), kTrades = kpiVal();
    private final LinePanel equity =
            new LinePanel(UiTheme.GREEN, "Equity curve appears after trades are recorded");
    private final BarPanel weekly = new BarPanel();

    public ReportView(ReportController rc) {
        this.rc = rc;
        setLayout(new BorderLayout(10, 10));
        setBackground(UiTheme.BG);
        setBorder(BorderFactory.createEmptyBorder(12, 12, 12, 12));

        JPanel north = new JPanel(new BorderLayout(0, 8));
        north.setOpaque(false);
        JPanel filter = new JPanel(new FlowLayout(FlowLayout.RIGHT, 8, 0));
        filter.setOpaque(false);
        JButton refresh = new JButton("Refresh");
        filter.add(period);
        filter.add(refresh);
        north.add(filter, BorderLayout.NORTH);

        JPanel kpis = new JPanel(new GridLayout(1, 5, 10, 0));
        kpis.setOpaque(false);
        kpis.add(kpiCard("ROI", kRoi));
        kpis.add(kpiCard("MDD", kMdd));
        kpis.add(kpiCard("Win rate", kWin));
        kpis.add(kpiCard("Sharpe", kSharpe));
        kpis.add(kpiCard("Trades", kTrades));
        north.add(kpis, BorderLayout.CENTER);
        add(north, BorderLayout.NORTH);

        JPanel eqCard = UiTheme.card();
        eqCard.setLayout(new BorderLayout(0, 6));
        eqCard.add(UiTheme.title("Equity curve"), BorderLayout.NORTH);
        eqCard.add(equity, BorderLayout.CENTER);
        add(eqCard, BorderLayout.CENTER);

        JPanel wkCard = UiTheme.card();
        wkCard.setLayout(new BorderLayout(0, 6));
        wkCard.add(UiTheme.title("Weekly P&L"), BorderLayout.NORTH);
        wkCard.add(weekly, BorderLayout.CENTER);
        add(wkCard, BorderLayout.SOUTH);

        refresh.addActionListener(e -> refresh());
        period.addActionListener(e -> refresh());
        refresh();
    }

    private static JLabel kpiVal() {
        JLabel l = new JLabel("-", SwingConstants.LEFT);
        l.setFont(l.getFont().deriveFont(Font.BOLD, 18f));
        return l;
    }

    private JPanel kpiCard(String title, JLabel val) {
        JPanel c = UiTheme.card();
        c.setLayout(new BorderLayout(0, 4));
        c.add(UiTheme.sub(title), BorderLayout.NORTH);
        c.add(val, BorderLayout.CENTER);
        return c;
    }

    public void refresh() {
        String p = switch (period.getSelectedIndex()) {
            case 1 -> "30d";
            case 2 -> "7d";
            default -> "all";
        };
        new Thread(() -> {
            PerformanceReport r = rc.getReport(p);
            List<Double> curve = r.equityCurve();
            double[] ys = new double[curve.size()];
            for (int i = 0; i < ys.length; i++) ys[i] = curve.get(i);

            List<TradeRecord> recs = r.getRecords();
            double[] bars;
            String[] labels;
            if (recs.isEmpty()) {
                bars = new double[0];
                labels = new String[0];
            } else {
                long first = recs.get(0).exitTime();
                long week = 7L * 86_400_000;
                int nb = (int) ((recs.get(recs.size() - 1).exitTime() - first) / week) + 1;
                nb = Math.min(Math.max(nb, 1), 10);
                bars = new double[nb];
                labels = new String[nb];
                for (TradeRecord t : recs) {
                    int b = (int) ((t.exitTime() - first) / week);
                    if (b >= nb) b = nb - 1;
                    bars[b] += t.realizedPnl();
                }
                for (int i = 0; i < nb; i++) labels[i] = "W" + (i + 1);
            }

            final double roi = r.roi(), mdd = r.maxDrawdown(),
                    win = r.winRate(), sh = r.sharpeRatio();
            final int n = r.tradeCount();
            final double[] fb = bars;
            final String[] fl = labels;
            SwingUtilities.invokeLater(() -> {
                kRoi.setText(UiTheme.pct(roi));
                kRoi.setForeground(roi >= 0 ? UiTheme.GREEN : UiTheme.RED);
                kMdd.setText(String.format("-%.2f%%", mdd * 100));
                kMdd.setForeground(UiTheme.RED);
                kWin.setText(String.format("%.0f%%", win * 100));
                kSharpe.setText(String.format("%.2f", sh));
                kTrades.setText(String.valueOf(n));
                equity.setSeries(ys);
                weekly.setBars(fb, fl);
            });
        }, "report-refresh").start();
    }
}

package ats.presentation;

import ats.domain.MarketData;

import javax.swing.JPanel;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.util.ArrayList;
import java.util.List;

/** 실시간 캔들 차트 (직접 페인팅, 외부 차트 라이브러리 없음) */
public class CandlePanel extends JPanel {
    private static final int MAX = 80;
    private final List<MarketData> data = new ArrayList<>();

    public CandlePanel() {
        setBackground(Color.WHITE);
        setPreferredSize(new Dimension(600, 320));
    }

    public synchronized void add(MarketData d) {
        data.add(d);
        while (data.size() > MAX) data.remove(0);
        repaint();
    }

    public synchronized void setAll(List<MarketData> ds) {
        data.clear();
        int from = Math.max(0, ds.size() - MAX);
        data.addAll(ds.subList(from, ds.size()));
        repaint();
    }

    @Override
    protected synchronized void paintComponent(Graphics g0) {
        super.paintComponent(g0);
        Graphics2D g = (Graphics2D) g0;
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        int w = getWidth(), h = getHeight();
        if (data.isEmpty()) {
            g.setColor(UiTheme.SUB);
            g.drawString("Start engine to stream candles", 20, h / 2);
            return;
        }
        double min = Double.MAX_VALUE, max = -Double.MAX_VALUE;
        for (MarketData d : data) {
            min = Math.min(min, d.low());
            max = Math.max(max, d.high());
        }
        double pad = (max - min) * 0.05 + 1e-9;
        min -= pad; max += pad;
        int n = data.size();
        double cw = (double) (w - 70) / n;
        for (int i = 0; i < n; i++) {
            MarketData d = data.get(i);
            int cx = (int) (10 + i * cw + cw / 2);
            int yH = y(d.high(), min, max, h), yL = y(d.low(), min, max, h);
            int yO = y(d.open(), min, max, h), yC = y(d.close(), min, max, h);
            g.setColor(d.close() >= d.open() ? UiTheme.GREEN : UiTheme.RED);
            g.drawLine(cx, yH, cx, yL);
            int bw = Math.max(2, (int) (cw * 0.6));
            g.fillRect(cx - bw / 2, Math.min(yO, yC), bw, Math.max(1, Math.abs(yC - yO)));
        }
        MarketData last = data.get(n - 1);
        g.setColor(UiTheme.TXT);
        g.drawString(String.format("%,.2f", last.close()), w - 64, y(last.close(), min, max, h));
    }

    private static int y(double v, double min, double max, int h) {
        return (int) (h - 10 - (v - min) / (max - min) * (h - 20));
    }
}

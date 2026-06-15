package ats.presentation;

import javax.swing.JPanel;
import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;

/** 범용 라인 차트 (자본곡선 / GA 적합도 수렴) */
public class LinePanel extends JPanel {
    private double[] series = new double[0];
    private final Color lineColor;
    private final String emptyText;

    public LinePanel(Color color, String emptyText) {
        this.lineColor = color;
        this.emptyText = emptyText;
        setBackground(Color.WHITE);
        setPreferredSize(new Dimension(500, 200));
    }

    public synchronized void setSeries(double[] ys) {
        series = ys == null ? new double[0] : ys.clone();
        repaint();
    }

    @Override
    protected synchronized void paintComponent(Graphics g0) {
        super.paintComponent(g0);
        Graphics2D g = (Graphics2D) g0;
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        int w = getWidth(), h = getHeight();
        if (series.length < 2) {
            g.setColor(UiTheme.SUB);
            g.drawString(emptyText, 20, h / 2);
            return;
        }
        double min = Double.MAX_VALUE, max = -Double.MAX_VALUE;
        for (double v : series) { min = Math.min(min, v); max = Math.max(max, v); }
        if (max - min < 1e-12) max = min + 1;
        int n = series.length;
        int[] xs = new int[n], ys = new int[n];
        for (int i = 0; i < n; i++) {
            xs[i] = 10 + (int) ((double) i / (n - 1) * (w - 80));
            ys[i] = (int) (h - 15 - (series[i] - min) / (max - min) * (h - 30));
        }
        g.setColor(lineColor);
        g.setStroke(new BasicStroke(1.6f));
        g.drawPolyline(xs, ys, n);
        g.setColor(UiTheme.SUB);
        g.drawString(String.format("%,.2f", max), w - 66, 16);
        g.drawString(String.format("%,.2f", min), w - 66, h - 8);
    }
}

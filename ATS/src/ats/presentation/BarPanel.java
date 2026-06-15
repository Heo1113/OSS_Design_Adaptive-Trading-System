package ats.presentation;

import javax.swing.JPanel;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Graphics2D;

/** 주별 손익 막대 차트 */
public class BarPanel extends JPanel {
    private double[] values = new double[0];
    private String[] labels = new String[0];

    public BarPanel() {
        setBackground(Color.WHITE);
        setPreferredSize(new Dimension(500, 120));
    }

    public synchronized void setBars(double[] v, String[] lab) {
        values = v == null ? new double[0] : v.clone();
        labels = lab == null ? new String[0] : lab.clone();
        repaint();
    }

    @Override
    protected synchronized void paintComponent(Graphics g0) {
        super.paintComponent(g0);
        Graphics2D g = (Graphics2D) g0;
        int w = getWidth(), h = getHeight();
        if (values.length == 0) {
            g.setColor(UiTheme.SUB);
            g.drawString("No trade data yet", 20, h / 2);
            return;
        }
        double maxAbs = 1e-9;
        for (double v : values) maxAbs = Math.max(maxAbs, Math.abs(v));
        int base = h / 2;
        int n = values.length;
        int slot = (w - 40) / n;
        int bw = Math.max(8, slot - 12);
        g.setColor(UiTheme.BORDER);
        g.drawLine(10, base, w - 10, base);
        for (int i = 0; i < n; i++) {
            int x = 20 + i * slot;
            int bh = (int) (Math.abs(values[i]) / maxAbs * (h / 2.0 - 18));
            g.setColor(values[i] >= 0 ? UiTheme.GREEN : UiTheme.RED);
            if (values[i] >= 0) g.fillRect(x, base - bh, bw, bh);
            else                g.fillRect(x, base, bw, bh);
            g.setColor(UiTheme.SUB);
            if (i < labels.length) g.drawString(labels[i], x, h - 4);
        }
    }
}

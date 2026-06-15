package ats.presentation;

import javax.swing.BorderFactory;
import javax.swing.JLabel;
import javax.swing.JPanel;
import java.awt.Color;
import java.awt.Font;

/** Analysis 단계 UI prototype의 색상 팔레트를 그대로 사용 */
public final class UiTheme {
    public static final Color GREEN  = new Color(0x1D9E75);
    public static final Color RED    = new Color(0xE24B4A);
    public static final Color BG     = new Color(0xF4F3EE);
    public static final Color CARD   = Color.WHITE;
    public static final Color BORDER = new Color(0xD3D1C7);
    public static final Color TXT    = new Color(0x2C2C2A);
    public static final Color SUB    = new Color(0x5F5E5A);

    private UiTheme() { }

    public static JPanel card() {
        JPanel p = new JPanel();
        p.setBackground(CARD);
        p.setBorder(BorderFactory.createCompoundBorder(
                BorderFactory.createLineBorder(BORDER),
                BorderFactory.createEmptyBorder(10, 12, 10, 12)));
        return p;
    }

    public static JLabel title(String text) {
        JLabel l = new JLabel(text);
        l.setFont(l.getFont().deriveFont(Font.BOLD, 14f));
        l.setForeground(TXT);
        return l;
    }

    public static JLabel sub(String text) {
        JLabel l = new JLabel(text);
        l.setFont(l.getFont().deriveFont(11f));
        l.setForeground(SUB);
        return l;
    }

    public static String fmt(double v) { return String.format("%,.2f", v); }
    public static String pct(double v) { return String.format("%+.2f%%", v * 100); }
}

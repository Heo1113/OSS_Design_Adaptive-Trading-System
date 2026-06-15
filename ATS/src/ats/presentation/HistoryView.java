package ats.presentation;

import ats.control.ReportController;
import ats.domain.TradeRecord;

import javax.swing.BorderFactory;
import javax.swing.JButton;
import javax.swing.JFileChooser;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTable;
import javax.swing.SwingUtilities;
import javax.swing.table.DefaultTableModel;
import java.awt.BorderLayout;
import java.awt.FlowLayout;
import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.List;

/** 거래 내역 화면 (Analysis UI prototype #5 구현) */
public class HistoryView extends JPanel {

    private final ReportController rc;
    private final DefaultTableModel model = new DefaultTableModel(
            new Object[]{"Exit time", "Symbol", "Side", "Qty", "Entry", "Exit", "PnL"}, 0) {
        @Override public boolean isCellEditable(int r, int c) { return false; }
    };
    private final SimpleDateFormat df = new SimpleDateFormat("MM-dd HH:mm:ss");

    public HistoryView(ReportController rc) {
        this.rc = rc;
        setLayout(new BorderLayout(10, 10));
        setBackground(UiTheme.BG);
        setBorder(BorderFactory.createEmptyBorder(12, 12, 12, 12));

        JPanel top = new JPanel(new FlowLayout(FlowLayout.RIGHT, 8, 0));
        top.setOpaque(false);
        JButton refresh = new JButton("Refresh");
        JButton export = new JButton("Export CSV");
        top.add(refresh);
        top.add(export);
        add(top, BorderLayout.NORTH);

        JTable table = new JTable(model);
        table.setRowHeight(24);
        add(new JScrollPane(table), BorderLayout.CENTER);

        refresh.addActionListener(e -> refresh());
        export.addActionListener(e -> {
            JFileChooser fc = new JFileChooser();
            fc.setSelectedFile(new File("trades_export.csv"));
            if (fc.showSaveDialog(this) == JFileChooser.APPROVE_OPTION) {
                try {
                    Path src = Path.of("data", "trades.csv");
                    if (Files.exists(src)) {
                        Files.copy(src, fc.getSelectedFile().toPath(),
                                StandardCopyOption.REPLACE_EXISTING);
                        JOptionPane.showMessageDialog(this, "Exported.");
                    } else {
                        JOptionPane.showMessageDialog(this, "No trade data yet.");
                    }
                } catch (Exception ex) {
                    JOptionPane.showMessageDialog(this, "Export failed: " + ex.getMessage());
                }
            }
        });
        refresh();
    }

    public void refresh() {
        new Thread(() -> {
            List<TradeRecord> recs = rc.getReport("all").getRecords();
            SwingUtilities.invokeLater(() -> {
                model.setRowCount(0);
                for (int i = recs.size() - 1; i >= 0; i--) {
                    TradeRecord t = recs.get(i);
                    model.addRow(new Object[]{
                            df.format(new Date(t.exitTime())), t.symbol(), t.side(),
                            String.format("%.3f", t.quantity()),
                            String.format("%,.2f", t.entryPrice()),
                            String.format("%,.2f", t.exitPrice()),
                            String.format("%+,.2f", t.realizedPnl())});
                }
            });
        }, "history-refresh").start();
    }
}

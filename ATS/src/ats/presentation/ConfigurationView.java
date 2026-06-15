package ats.presentation;

import ats.control.AppConfig;
import ats.control.EngineController;

import javax.swing.BorderFactory;
import javax.swing.Box;
import javax.swing.BoxLayout;
import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JComboBox;
import javax.swing.JComponent;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JPasswordField;
import javax.swing.JScrollPane;
import javax.swing.JSpinner;
import javax.swing.SpinnerNumberModel;
import javax.swing.SwingUtilities;
import java.awt.BorderLayout;
import java.awt.Dimension;
import java.awt.FlowLayout;
import java.awt.GridLayout;

/** 설정 화면 (Analysis UI prototype #1 Configuration 구현, Design Seq 3.2) */
public class ConfigurationView extends JPanel {

    private final EngineController ec;
    private final JPasswordField keyField = new JPasswordField(24);
    private final JPasswordField secretField = new JPasswordField(24);
    private final JCheckBox testnet = new JCheckBox("Use Binance Testnet", true);
    private final JComboBox<String> symbol =
            new JComboBox<>(new String[]{"BTCUSDT", "ETHUSDT", "SOLUSDT"});
    private final JSpinner leverage = new JSpinner(new SpinnerNumberModel(5, 1, 125, 1));
    private final JSpinner risk = new JSpinner(new SpinnerNumberModel(0.10, 0.01, 0.50, 0.01));
    private final JSpinner pop = new JSpinner(new SpinnerNumberModel(40, 10, 200, 5));
    private final JSpinner gen = new JSpinner(new SpinnerNumberModel(30, 5, 200, 5));
    private final JSpinner mut = new JSpinner(new SpinnerNumberModel(0.10, 0.01, 0.50, 0.01));
    private final JCheckBox wfo = new JCheckBox("Walk-Forward validation", true);
    private final JLabel status = new JLabel(" ");

    public ConfigurationView(EngineController ec) {
        this.ec = ec;
        AppConfig c = ec.config();
        symbol.setEditable(true);
        symbol.setSelectedItem(c.symbol);
        leverage.setValue(c.leverage);
        risk.setValue(c.riskPct);
        pop.setValue(c.populationSize);
        gen.setValue(c.generations);
        mut.setValue(c.mutationRate);
        wfo.setSelected(c.useWfo);

        setLayout(new BorderLayout());
        setBackground(UiTheme.BG);

        JPanel col = new JPanel();
        col.setOpaque(false);
        col.setLayout(new BoxLayout(col, BoxLayout.Y_AXIS));
        col.setBorder(BorderFactory.createEmptyBorder(12, 12, 12, 12));

        col.add(section("API settings  (leave empty for DEMO mode)",
                new String[]{"API Key", "API Secret", ""},
                new JComponent[]{keyField, secretField, testnet}));
        col.add(Box.createVerticalStrut(10));
        col.add(section("Trading parameters",
                new String[]{"Symbol", "Leverage", "Risk per trade"},
                new JComponent[]{symbol, leverage, risk}));
        col.add(Box.createVerticalStrut(10));
        col.add(section("GA parameters",
                new String[]{"Population size", "Generations", "Mutation rate", ""},
                new JComponent[]{pop, gen, mut, wfo}));
        col.add(Box.createVerticalStrut(10));

        JPanel btns = new JPanel(new FlowLayout(FlowLayout.RIGHT));
        btns.setOpaque(false);
        JButton validate = new JButton("Validate API key");
        JButton save = new JButton("Save configuration");
        btns.add(status);
        btns.add(validate);
        btns.add(save);
        btns.setMaximumSize(new Dimension(640, 50));
        btns.setAlignmentX(LEFT_ALIGNMENT);
        col.add(btns);
        col.add(Box.createVerticalGlue());

        JScrollPane sp = new JScrollPane(col);
        sp.setBorder(null);
        sp.getViewport().setBackground(UiTheme.BG);
        add(sp, BorderLayout.CENTER);

        validate.addActionListener(e -> {
            validate.setEnabled(false);
            status.setText("validating...");
            status.setForeground(UiTheme.SUB);
            String k = new String(keyField.getPassword());
            String s = new String(secretField.getPassword());
            boolean tn = testnet.isSelected();
            new Thread(() -> {
                boolean ok = ec.testCredential(k, s, tn);
                SwingUtilities.invokeLater(() -> {
                    status.setText(ok ? "API key valid" : "invalid API key");
                    status.setForeground(ok ? UiTheme.GREEN : UiTheme.RED);
                    validate.setEnabled(true);
                });
            }, "validate-key").start();
        });

        save.addActionListener(e -> {
            save.setEnabled(false);
            status.setText("saving...");
            status.setForeground(UiTheme.SUB);
            String k = new String(keyField.getPassword());
            String s = new String(secretField.getPassword());
            boolean tn = testnet.isSelected();
            new Thread(() -> {
                String err = ec.saveConfig(k, s, tn,
                        String.valueOf(symbol.getSelectedItem()),
                        (Integer) leverage.getValue(), (Double) risk.getValue(),
                        (Integer) pop.getValue(), (Integer) gen.getValue(),
                        (Double) mut.getValue(), wfo.isSelected());
                SwingUtilities.invokeLater(() -> {
                    status.setText(err == null ? "saved" : err);
                    status.setForeground(err == null ? UiTheme.GREEN : UiTheme.RED);
                    save.setEnabled(true);
                });
            }, "save-config").start();
        });
    }

    private JPanel section(String title, String[] labels, JComponent[] fields) {
        JPanel card = UiTheme.card();
        card.setLayout(new BorderLayout(0, 8));
        card.add(UiTheme.title(title), BorderLayout.NORTH);
        JPanel grid = new JPanel(new GridLayout(labels.length, 2, 8, 6));
        grid.setOpaque(false);
        for (int i = 0; i < labels.length; i++) {
            grid.add(UiTheme.sub(labels[i]));
            grid.add(fields[i]);
        }
        card.add(grid, BorderLayout.CENTER);
        card.setMaximumSize(new Dimension(640, Integer.MAX_VALUE));
        card.setAlignmentX(LEFT_ALIGNMENT);
        return card;
    }
}

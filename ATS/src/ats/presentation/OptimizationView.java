package ats.presentation;

import ats.control.OptimizationController;
import ats.domain.Backtester;

import javax.swing.BorderFactory;
import javax.swing.JButton;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JProgressBar;
import javax.swing.JScrollPane;
import javax.swing.JTextArea;
import javax.swing.SwingUtilities;
import java.awt.BorderLayout;
import java.awt.GridLayout;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/** GA 최적화 화면 (Analysis UI prototype #3 구현, Design Seq 3.3) */
public class OptimizationView extends JPanel {

    private final OptimizationController oc;
    private final JButton run = new JButton("Run GA Optimization");
    private final JProgressBar progress = new JProgressBar();
    private final LinePanel convergence =
            new LinePanel(UiTheme.GREEN, "Run optimization to see fitness convergence");
    private final JTextArea params = new JTextArea(8, 22);
    private final JLabel kpi = new JLabel(" ");
    private final List<Double> hist = new ArrayList<>();

    public OptimizationView(OptimizationController oc) {
        this.oc = oc;
        setLayout(new BorderLayout(10, 10));
        setBackground(UiTheme.BG);
        setBorder(BorderFactory.createEmptyBorder(12, 12, 12, 12));

        JPanel top = new JPanel(new BorderLayout(10, 0));
        top.setOpaque(false);
        progress.setStringPainted(true);
        progress.setString("idle");
        top.add(run, BorderLayout.WEST);
        top.add(progress, BorderLayout.CENTER);
        add(top, BorderLayout.NORTH);

        JPanel center = new JPanel(new GridLayout(1, 2, 10, 0));
        center.setOpaque(false);
        JPanel left = UiTheme.card();
        left.setLayout(new BorderLayout(0, 6));
        left.add(UiTheme.title("Best parameters"), BorderLayout.NORTH);
        params.setEditable(false);
        left.add(new JScrollPane(params), BorderLayout.CENTER);
        JPanel right = UiTheme.card();
        right.setLayout(new BorderLayout(0, 6));
        right.add(UiTheme.title("Fitness convergence"), BorderLayout.NORTH);
        right.add(convergence, BorderLayout.CENTER);
        center.add(left);
        center.add(right);
        add(center, BorderLayout.CENTER);

        JPanel south = UiTheme.card();
        south.setLayout(new BorderLayout());
        kpi.setFont(kpi.getFont().deriveFont(13f));
        south.add(kpi, BorderLayout.CENTER);
        add(south, BorderLayout.SOUTH);

        run.addActionListener(e -> startOptimization());
    }

    private void startOptimization() {
        run.setEnabled(false);
        hist.clear();
        convergence.setSeries(new double[0]);
        params.setText("");
        kpi.setText("optimizing...");

        oc.optimize(
                (g, total, best) -> SwingUtilities.invokeLater(() -> {
                    progress.setMaximum(total);
                    progress.setValue(g);
                    progress.setString("Gen " + g + "/" + total
                            + "   best fitness " + String.format("%.4f", best));
                    hist.add(best);
                    double[] ys = new double[hist.size()];
                    for (int i = 0; i < ys.length; i++) ys[i] = hist.get(i);
                    convergence.setSeries(ys);
                }),
                (best, wfoResult) -> SwingUtilities.invokeLater(() -> {
                    StringBuilder b = new StringBuilder();
                    for (Map.Entry<String, Object> en : best.entrySet())
                        b.append(en.getKey()).append(" = ").append(en.getValue()).append('\n');
                    params.setText(b.toString());

                    Backtester.Result tr = oc.lastTrain();
                    Backtester.Result te = oc.lastTest();
                    StringBuilder k = new StringBuilder("<html>");
                    k.append(String.format(
                            "In-sample ROI <b>%s</b> / MDD %.2f%% / %d trades &nbsp;|&nbsp; "
                                    + "Hold-out ROI <b>%s</b>",
                            UiTheme.pct(tr.roi()), tr.mdd() * 100, tr.trades(),
                            UiTheme.pct(te.roi())));
                    if (wfoResult != null) {
                        double is = (Double) wfoResult.get("inSampleRoi");
                        double oos = (Double) wfoResult.get("outSampleRoi");
                        double ratio = (Double) wfoResult.get("overfitRatio");
                        k.append(String.format(
                                "<br>Walk-Forward: IS %s vs OOS %s (ratio %.2f) — %s",
                                UiTheme.pct(is), UiTheme.pct(oos), ratio,
                                ratio >= 0.5 ? "OK" : "overfitting suspected"));
                    }
                    k.append("<br><i>Best parameters applied to the live strategy.</i></html>");
                    kpi.setText(k.toString());
                    run.setEnabled(true);
                }),
                err -> SwingUtilities.invokeLater(() -> {
                    kpi.setText("error: " + err);
                    run.setEnabled(true);
                })
        );
    }
}

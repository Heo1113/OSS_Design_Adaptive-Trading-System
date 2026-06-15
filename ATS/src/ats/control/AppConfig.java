package ats.control;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Properties;

/** 사용자 설정 (data/config.properties 영속화) */
public class AppConfig {
    public String symbol = "BTCUSDT";
    public int leverage = 5;
    public double riskPct = 0.10;
    public int populationSize = 40;
    public int generations = 30;
    public double mutationRate = 0.10;
    public boolean useWfo = true;

    private final Path file = Path.of("data", "config.properties");

    public void load() {
        try {
            if (!Files.exists(file)) return;
            Properties p = new Properties();
            try (var in = Files.newInputStream(file)) { p.load(in); }
            symbol = p.getProperty("symbol", symbol);
            leverage = Integer.parseInt(p.getProperty("leverage", String.valueOf(leverage)));
            riskPct = Double.parseDouble(p.getProperty("riskPct", String.valueOf(riskPct)));
            populationSize = Integer.parseInt(p.getProperty("populationSize", String.valueOf(populationSize)));
            generations = Integer.parseInt(p.getProperty("generations", String.valueOf(generations)));
            mutationRate = Double.parseDouble(p.getProperty("mutationRate", String.valueOf(mutationRate)));
            useWfo = Boolean.parseBoolean(p.getProperty("useWfo", String.valueOf(useWfo)));
        } catch (Exception ignored) { }
    }

    public void save() {
        try {
            Files.createDirectories(file.getParent());
            Properties p = new Properties();
            p.setProperty("symbol", symbol);
            p.setProperty("leverage", String.valueOf(leverage));
            p.setProperty("riskPct", String.valueOf(riskPct));
            p.setProperty("populationSize", String.valueOf(populationSize));
            p.setProperty("generations", String.valueOf(generations));
            p.setProperty("mutationRate", String.valueOf(mutationRate));
            p.setProperty("useWfo", String.valueOf(useWfo));
            try (var out = Files.newOutputStream(file)) { p.store(out, "ATS config"); }
        } catch (Exception ignored) { }
    }
}

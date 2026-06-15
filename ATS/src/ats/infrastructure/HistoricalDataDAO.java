package ats.infrastructure;

import ats.domain.MarketData;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/** 과거 시세 캐시 영속화 (CSV). Design 2.1 Infrastructure 계층 */
public class HistoricalDataDAO {

    public void save(String symbol, List<MarketData> data) {
        try {
            Path f = Path.of("data", "hist_" + symbol + ".csv");
            Files.createDirectories(f.getParent());
            StringBuilder b = new StringBuilder();
            for (MarketData d : data)
                b.append(d.timestamp()).append(',').append(d.open()).append(',')
                 .append(d.high()).append(',').append(d.low()).append(',')
                 .append(d.close()).append(',').append(d.volume()).append('\n');
            Files.writeString(f, b.toString());
        } catch (Exception e) {
            System.err.println("HistoricalDataDAO.save 실패: " + e);
        }
    }

    public List<MarketData> load(String symbol) {
        List<MarketData> out = new ArrayList<>();
        try {
            Path f = Path.of("data", "hist_" + symbol + ".csv");
            if (!Files.exists(f)) return out;
            for (String line : Files.readAllLines(f)) {
                if (line.isBlank()) continue;
                String[] t = line.split(",");
                out.add(new MarketData(Long.parseLong(t[0]),
                        Double.parseDouble(t[1]), Double.parseDouble(t[2]),
                        Double.parseDouble(t[3]), Double.parseDouble(t[4]),
                        Double.parseDouble(t[5])));
            }
        } catch (Exception e) {
            System.err.println("HistoricalDataDAO.load 실패: " + e);
        }
        return out;
    }
}

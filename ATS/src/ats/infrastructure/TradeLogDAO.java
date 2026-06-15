package ats.infrastructure;

import ats.domain.TradeRecord;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;

/** 거래 기록 영속화 (CSV). Design 2.1 Infrastructure 계층 */
public class TradeLogDAO {
    private final Path file = Path.of("data", "trades.csv");

    public synchronized void save(TradeRecord r) {
        try {
            Files.createDirectories(file.getParent());
            Files.writeString(file, r.toCsv() + System.lineSeparator(),
                    StandardOpenOption.CREATE, StandardOpenOption.APPEND);
        } catch (Exception e) {
            System.err.println("TradeLogDAO.save 실패: " + e);
        }
    }

    /** period: "all" | "7d" | "30d" */
    public synchronized List<TradeRecord> queryRecords(String period) {
        List<TradeRecord> out = new ArrayList<>();
        try {
            if (!Files.exists(file)) return out;
            long cutoff = switch (period) {
                case "7d"  -> System.currentTimeMillis() - 7L * 86_400_000;
                case "30d" -> System.currentTimeMillis() - 30L * 86_400_000;
                default    -> 0L;
            };
            for (String line : Files.readAllLines(file)) {
                if (line.isBlank()) continue;
                try {
                    TradeRecord r = TradeRecord.fromCsv(line);
                    if (r.exitTime() >= cutoff) out.add(r);
                } catch (Exception ignored) { }
            }
        } catch (Exception e) {
            System.err.println("TradeLogDAO.query 실패: " + e);
        }
        return out;
    }
}

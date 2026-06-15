package ats.control;

import ats.domain.PerformanceReport;
import ats.infrastructure.TradeLogDAO;

/** 성과 리포트 use case 조율 (Design Seq 3.5) */
public class ReportController {
    private final TradeLogDAO dao;

    public ReportController(TradeLogDAO dao) { this.dao = dao; }

    /** period: "all" | "30d" | "7d" */
    public PerformanceReport getReport(String period) {
        return new PerformanceReport(dao.queryRecords(period));
    }
}

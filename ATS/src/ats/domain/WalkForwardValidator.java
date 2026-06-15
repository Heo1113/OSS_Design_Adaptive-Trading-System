package ats.domain;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Walk-Forward 검증 (Design 2.2). 데이터를 windowCount개 구간으로 나누고
 * 각 구간을 학습(70%)/검증(30%)으로 분할해 In-Sample vs Out-of-Sample ROI를 비교,
 * 과적합 여부를 진단함.
 */
public class WalkForwardValidator {
    private final int windowCount;
    private final HistoricalDataset dataset;

    public WalkForwardValidator(int windowCount, HistoricalDataset dataset) {
        this.windowCount = windowCount;
        this.dataset = dataset;
    }

    public Map<String, Object> validate(Map<String, Object> params) {
        List<MarketData> d = dataset.getData();
        int n = d.size(), w = n / windowCount;
        double isSum = 0, oosSum = 0;
        for (int i = 0; i < windowCount; i++) {
            int s = i * w, e = (i == windowCount - 1) ? n : s + w;
            List<MarketData> seg = d.subList(s, e);
            int cut = (int) (seg.size() * 0.7);
            isSum  += Backtester.run(params, seg.subList(0, cut), 10_000, 0.10, 5).roi();
            oosSum += Backtester.run(params, seg.subList(cut, seg.size()), 10_000, 0.10, 5).roi();
        }
        double is = isSum / windowCount, oos = oosSum / windowCount;
        Map<String, Object> r = new LinkedHashMap<>();
        r.put("windows", windowCount);
        r.put("inSampleRoi", is);
        r.put("outSampleRoi", oos);
        r.put("overfitRatio", is == 0 ? 0 : oos / is);
        return r;
    }
}

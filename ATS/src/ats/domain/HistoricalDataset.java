package ats.domain;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/** 백테스트·GA 학습용 과거 시세 묶음 (Design 2.2) */
public class HistoricalDataset {
    private final List<MarketData> data;

    public HistoricalDataset(List<MarketData> data) {
        this.data = Collections.unmodifiableList(new ArrayList<>(data));
    }

    public List<MarketData> getData() { return data; }
    public int size() { return data.size(); }

    /** 학습/검증 분할 */
    public Pair<HistoricalDataset, HistoricalDataset> split(double trainRatio) {
        int cut = (int) (data.size() * trainRatio);
        return new Pair<>(new HistoricalDataset(data.subList(0, cut)),
                          new HistoricalDataset(data.subList(cut, data.size())));
    }
}

package ats.domain;

/** 단순 2-튜플 (HistoricalDataset.split 반환용) */
public record Pair<A, B>(A first, B second) { }

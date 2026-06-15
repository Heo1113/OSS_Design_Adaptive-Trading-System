package ats.domain;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * 유전 알고리즘으로 전략 파라미터를 최적화 (Design 2.2 GAOptimizer).
 * 유전자: [rsiPeriod, rsiOversold, rsiOverbought, maShort, maLong, stopLoss, takeProfit]
 * 적합도: ROI - 0.5 * MDD  (학습 구간 백테스트)
 * 연산: 엘리트 보존(2) + 토너먼트 선택(k=3) + 블렌드 교차 + 가우시안 돌연변이
 */
public class GAOptimizer {

    @FunctionalInterface
    public interface Listener { void onGeneration(int gen, int total, double bestFitness); }

    private static final double[] LO = {5, 15, 55, 4, 15, 0.004, 0.008};
    private static final double[] HI = {30, 45, 90, 15, 60, 0.040, 0.090};
    private static final double LAMBDA = 0.5;

    private final int populationSize;
    private final int generations;
    private final double mutationRate;
    private final HistoricalDataset dataset;
    private final Random rnd = new Random(7);
    private final List<double[]> population = new ArrayList<>();
    private final List<Double> bestHistory = new ArrayList<>();
    private Listener listener = (g, t, f) -> { };

    public GAOptimizer(int populationSize, int generations, double mutationRate,
                       HistoricalDataset dataset) {
        this.populationSize = populationSize;
        this.generations = generations;
        this.mutationRate = mutationRate;
        this.dataset = dataset;
    }

    public void setListener(Listener l) { if (l != null) listener = l; }
    public List<Double> getBestHistory() { return new ArrayList<>(bestHistory); }

    /** 최적화 실행 → 최적 파라미터 반환 */
    public Map<String, Object> run() {
        population.clear(); bestHistory.clear();
        for (int i = 0; i < populationSize; i++) population.add(randomGene());

        double[] best = null; double bestFit = -1e18;
        for (int gen = 1; gen <= generations; gen++) {
            double[] fit = new double[populationSize];
            for (int i = 0; i < populationSize; i++) fit[i] = evaluateFitness(population.get(i));

            Integer[] idx = new Integer[populationSize];
            for (int i = 0; i < populationSize; i++) idx[i] = i;
            java.util.Arrays.sort(idx, Comparator.comparingDouble(i -> -fit[i]));

            if (fit[idx[0]] > bestFit) { bestFit = fit[idx[0]]; best = population.get(idx[0]).clone(); }
            bestHistory.add(bestFit);
            listener.onGeneration(gen, generations, bestFit);

            evolve(idx, fit);
        }
        return decode(best);
    }

    /** 적합도 = ROI - λ·MDD */
    private double evaluateFitness(double[] gene) {
        Backtester.Result r = Backtester.run(decode(gene), dataset.getData(), 10_000, 0.10, 5);
        return r.roi() - LAMBDA * r.mdd();
    }

    /** 선택·교차·돌연변이로 다음 세대 생성 */
    private void evolve(Integer[] sortedIdx, double[] fit) {
        List<double[]> next = new ArrayList<>(populationSize);
        next.add(population.get(sortedIdx[0]).clone());           // elitism
        next.add(population.get(sortedIdx[1]).clone());
        while (next.size() < populationSize) {
            double[] a = tournament(fit), b = tournament(fit);
            double[] child = new double[7];
            for (int g = 0; g < 7; g++) {
                double w = rnd.nextDouble();
                child[g] = w * a[g] + (1 - w) * b[g];              // blend crossover
                if (rnd.nextDouble() < mutationRate)
                    child[g] += rnd.nextGaussian() * (HI[g] - LO[g]) * 0.10;
                child[g] = Math.min(HI[g], Math.max(LO[g], child[g]));
            }
            repair(child);
            next.add(child);
        }
        population.clear(); population.addAll(next);
    }

    private double[] tournament(double[] fit) {
        int bestI = rnd.nextInt(populationSize);
        for (int k = 1; k < 3; k++) {
            int c = rnd.nextInt(populationSize);
            if (fit[c] > fit[bestI]) bestI = c;
        }
        return population.get(bestI);
    }

    private double[] randomGene() {
        double[] g = new double[7];
        for (int i = 0; i < 7; i++) g[i] = LO[i] + rnd.nextDouble() * (HI[i] - LO[i]);
        repair(g);
        return g;
    }

    /** 제약 보정: maLong > maShort+3, overbought > oversold+15 */
    private void repair(double[] g) {
        if (g[4] < g[3] + 3) g[4] = g[3] + 3;
        if (g[2] < g[1] + 15) g[2] = g[1] + 15;
    }

    public static Map<String, Object> decode(double[] g) {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("rsiPeriod",     (int) Math.round(g[0]));
        m.put("rsiOversold",   round1(g[1]));
        m.put("rsiOverbought", round1(g[2]));
        m.put("maShort",       (int) Math.round(g[3]));
        m.put("maLong",        (int) Math.round(g[4]));
        m.put("stopLoss",      round4(g[5]));
        m.put("takeProfit",    round4(g[6]));
        return m;
    }

    private static double round1(double v) { return Math.round(v * 10) / 10.0; }
    private static double round4(double v) { return Math.round(v * 10_000) / 10_000.0; }
}

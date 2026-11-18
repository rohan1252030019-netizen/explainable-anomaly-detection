import { describe, expect, it } from "vitest";
import { DEFAULT_MAX_POINTS, extent, sampleSeries } from "../downsample";

interface Point {
  id: number;
  anomaly: boolean;
}

const makePoints = (n: number, anomalyEvery = 50): Point[] =>
  Array.from({ length: n }, (_, i) => ({
    id: i,
    anomaly: i % anomalyEvery === 0,
  }));

describe("sampleSeries", () => {
  it("returns the input untouched when it is already small", () => {
    const points = makePoints(10);
    const result = sampleSeries(points, 100);
    expect(result.points).toHaveLength(10);
    expect(result.sampled).toBe(false);
    expect(result.total).toBe(10);
    expect(result.points[0]).toBe(points[0]);
  });

  it("caps the render set", () => {
    const result = sampleSeries(makePoints(50_000), 4000);
    expect(result.total).toBe(50_000);
    expect(result.sampled).toBe(true);
    expect(result.points.length).toBeLessThanOrEqual(4000);
  });

  it("never drops an anomaly when anomalies fit the budget", () => {
    const points = makePoints(20_000, 100); // 200 anomalies
    const result = sampleSeries(points, 2000, (p) => p.anomaly);
    const keptAnomalies = result.points.filter((p) => p.anomaly).length;
    const totalAnomalies = points.filter((p) => p.anomaly).length;
    expect(keptAnomalies).toBe(totalAnomalies);
  });

  it("keeps anomalies even when they exceed half the budget", () => {
    // Every row is an anomaly: the priority budget must still bound output.
    const points = makePoints(1000, 1);
    const result = sampleSeries(points, 100, (p) => p.anomaly);
    expect(result.points.length).toBeLessThanOrEqual(100);
    expect(result.points.length).toBeGreaterThan(0);
  });

  it("preserves input order", () => {
    const result = sampleSeries(makePoints(10_000), 500, (p) => p.anomaly);
    const ids = result.points.map((p) => p.id);
    const sorted = [...ids].sort((a, b) => a - b);
    expect(ids).toEqual(sorted);
  });

  it("is deterministic", () => {
    const points = makePoints(10_000);
    const a = sampleSeries(points, 300, (p) => p.anomaly).points.map((p) => p.id);
    const b = sampleSeries(points, 300, (p) => p.anomaly).points.map((p) => p.id);
    expect(a).toEqual(b);
  });

  it("orders magnitudes that break Math.min(...values) via spread", () => {
    const spread = Array.from({ length: 300_000 }, (_, i) => i);
    expect(() => Math.min(...spread)).toThrow(RangeError);
    expect(extent(spread)).toEqual([0, 299_999]);
  });

  it("exports a sane default cap", () => {
    expect(DEFAULT_MAX_POINTS).toBeGreaterThan(100);
    expect(DEFAULT_MAX_POINTS).toBeLessThan(20_000);
  });
});

describe("extent", () => {
  it("handles empty and single-element input", () => {
    expect(extent([])).toEqual([0, 0]);
    expect(extent([7])).toEqual([7, 7]);
  });

  it("finds negative and fractional bounds", () => {
    expect(extent([-3.5, 2, 0.25])).toEqual([-3.5, 2]);
  });
});

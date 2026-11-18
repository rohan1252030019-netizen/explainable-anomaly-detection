/**
 * frontend/src/lib/downsample.ts
 *
 * Deterministic down-sampling for SVG charts.
 *
 * The PCA scatter and the score timeline render one DOM node per observation.
 * At 200k rows that is 200k `<circle>` elements, which locks up the main
 * thread for seconds and then makes every hover re-layout the whole tree.
 *
 * The rule kept here: never drop an anomaly to save a frame.  Anomalies are
 * kept first, then the remaining budget is filled with an evenly spaced
 * stride of the normal points so the overall shape of the distribution is
 * preserved.  The selection is a pure function of the input order, so the same
 * dataset always renders the same subset.
 */

export const DEFAULT_MAX_POINTS = 4000;

export interface SampledSeries<T> {
  /** Points to render, in their original order. */
  points: T[];
  /** Total number of input points. */
  total: number;
  /** True when points were dropped. */
  sampled: boolean;
}

/**
 * Keep every priority point (up to ``maxPoints``), fill the rest with a stride.
 *
 * @param points     Input points in display order.
 * @param maxPoints  Upper bound on the returned length.
 * @param isPriority Points that must not be dropped (e.g. anomalies).
 */
export function sampleSeries<T>(
  points: readonly T[],
  maxPoints: number = DEFAULT_MAX_POINTS,
  isPriority: (point: T) => boolean = () => false
): SampledSeries<T> {
  const total = points.length;
  if (maxPoints <= 0) {
    return { points: [], total, sampled: total > 0 };
  }
  if (total <= maxPoints) {
    return { points: [...points], total, sampled: false };
  }

  const priority: number[] = [];
  const rest: number[] = [];
  for (let i = 0; i < total; i += 1) {
    if (isPriority(points[i])) priority.push(i);
    else rest.push(i);
  }

  const selected = new Set<number>();

  // Priority points share the budget with the rest if there are too many of
  // them; otherwise they all survive.
  const priorityBudget = Math.min(priority.length, Math.floor(maxPoints / 2));
  const priorityStep = priority.length > priorityBudget
    ? priority.length / priorityBudget
    : 1;
  for (let i = 0; i < priorityBudget; i += 1) {
    selected.add(priority[Math.min(priority.length - 1, Math.floor(i * priorityStep))]);
  }

  const remaining = maxPoints - selected.size;
  if (remaining > 0 && rest.length > 0) {
    const step = rest.length / remaining;
    for (let i = 0; i < remaining; i += 1) {
      selected.add(rest[Math.min(rest.length - 1, Math.floor(i * step))]);
    }
  }

  return {
    points: [...selected].sort((a, b) => a - b).map((index) => points[index]),
    total,
    sampled: selected.size < total,
  };
}

/** Min/max without ``Math.min(...values)``, which overflows the stack on large arrays. */
export function extent(values: readonly number[]): [number, number] {
  if (values.length === 0) return [0, 0];
  let min = values[0];
  let max = values[0];
  for (let i = 1; i < values.length; i += 1) {
    const value = values[i];
    if (value < min) min = value;
    if (value > max) max = value;
  }
  return [min, max];
}

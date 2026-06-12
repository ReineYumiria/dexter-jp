export type PriceBar = {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
};

export type BollingerBands = {
  middle: number;
  sigma: number;
  upper2: number;
  lower2: number;
  bbPosition: number | null;
};

export type BollingerState =
  | "BB_DOWN_WALK"
  | "BB_REBOUND"
  | "BB_MIDDLE_RECOVER"
  | "BB_LOW_NEAR"
  | "BB_NEUTRAL"
  | "BB_UNKNOWN";

export type VolumeReboundState =
  | "STRONG_VOLUME_REBOUND"
  | "VOLUME_REBOUND"
  | "NO_VOLUME_REBOUND"
  | "VOLUME_UNKNOWN";

export type IchimokuState =
  | "ABOVE_CLOUD"
  | "CLOUD_BREAK_NEAR"
  | "IN_CLOUD"
  | "BELOW_CLOUD"
  | "UNKNOWN";

export type TechnicalIndicators = {
  latestClose: number | null;
  latestVolume: number | null;
  previousClose: number | null;
  bollinger: BollingerBands | null;
  bbState: BollingerState;
  volumeRebound: VolumeReboundState;
  ichimokuState: IchimokuState;
  middleLineRecovered: boolean | null;
};

const BOLLINGER_PERIOD = 20;
const ICHIMOKU_TENKAN_PERIOD = 9;
const ICHIMOKU_KIJUN_PERIOD = 26;
const ICHIMOKU_SENKOU_B_PERIOD = 52;

function average(values: number[]): number {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function standardDeviation(values: number[]): number {
  const mean = average(values);
  const variance =
    values.reduce((sum, value) => sum + Math.pow(value - mean, 2), 0) /
    values.length;

  return Math.sqrt(variance);
}

function midpointOfHighLow(bars: PriceBar[]): number {
  const highestHigh = Math.max(...bars.map((bar) => bar.high));
  const lowestLow = Math.min(...bars.map((bar) => bar.low));

  return (highestHigh + lowestLow) / 2;
}

export function calculateBollingerBands(
  bars: PriceBar[],
  period = BOLLINGER_PERIOD,
): BollingerBands | null {
  if (bars.length < period) {
    return null;
  }

  const targetBars = bars.slice(-period);
  const closes = targetBars.map((bar) => bar.close);
  const latestClose = closes[closes.length - 1];

  const middle = average(closes);
  const sigma = standardDeviation(closes);
  const upper2 = middle + 2 * sigma;
  const lower2 = middle - 2 * sigma;
  const bbPosition = sigma === 0 ? null : (latestClose - middle) / sigma;

  return {
    middle,
    sigma,
    upper2,
    lower2,
    bbPosition,
  };
}

export function classifyBollingerState(bars: PriceBar[]): BollingerState {
  if (bars.length < BOLLINGER_PERIOD + 1) {
    return "BB_UNKNOWN";
  }

  const latest = bars[bars.length - 1];
  const previous = bars[bars.length - 2];

  const latestBands = calculateBollingerBands(bars);
  const previousBands = calculateBollingerBands(bars.slice(0, -1));

  if (!latestBands || !previousBands) {
    return "BB_UNKNOWN";
  }

  const recentBars = bars.slice(-3);
  const lowerTouchCount = recentBars.filter((_, index) => {
    const endIndex = bars.length - recentBars.length + index + 1;
    const bands = calculateBollingerBands(bars.slice(0, endIndex));

    return bands ? bars[endIndex - 1].close <= bands.lower2 : false;
  }).length;

  if (lowerTouchCount >= 2) {
    return "BB_DOWN_WALK";
  }

  if (previous.close < previousBands.lower2 && latest.close >= latestBands.lower2) {
    return "BB_REBOUND";
  }

  if (previous.close < previousBands.middle && latest.close >= latestBands.middle) {
    return "BB_MIDDLE_RECOVER";
  }

  if (
    latestBands.bbPosition !== null &&
    latestBands.bbPosition <= -1.8
  ) {
    return "BB_LOW_NEAR";
  }

  return "BB_NEUTRAL";
}

export function classifyVolumeRebound(bars: PriceBar[]): VolumeReboundState {
  if (bars.length < BOLLINGER_PERIOD + 1) {
    return "VOLUME_UNKNOWN";
  }

  const latest = bars[bars.length - 1];
  const previous = bars[bars.length - 2];
  const recentVolumes = bars.slice(-BOLLINGER_PERIOD).map((bar) => bar.volume);
  const averageVolume = average(recentVolumes);

  if (averageVolume <= 0) {
    return "VOLUME_UNKNOWN";
  }

  const isPriceUp = latest.close > previous.close;

  if (latest.volume >= averageVolume * 1.8 && isPriceUp) {
    return "STRONG_VOLUME_REBOUND";
  }

  if (latest.volume >= averageVolume * 1.3 && isPriceUp) {
    return "VOLUME_REBOUND";
  }

  return "NO_VOLUME_REBOUND";
}

export function calculateIchimokuState(bars: PriceBar[]): IchimokuState {
  if (bars.length < ICHIMOKU_SENKOU_B_PERIOD) {
    return "UNKNOWN";
  }

  const latestClose = bars[bars.length - 1].close;

  const tenkan = midpointOfHighLow(bars.slice(-ICHIMOKU_TENKAN_PERIOD));
  const kijun = midpointOfHighLow(bars.slice(-ICHIMOKU_KIJUN_PERIOD));
  const senkouA = (tenkan + kijun) / 2;
  const senkouB = midpointOfHighLow(bars.slice(-ICHIMOKU_SENKOU_B_PERIOD));

  const cloudTop = Math.max(senkouA, senkouB);
  const cloudBottom = Math.min(senkouA, senkouB);

  if (latestClose > cloudTop) {
    return "ABOVE_CLOUD";
  }

  if (latestClose < cloudTop && latestClose >= cloudTop * 0.97) {
    return "CLOUD_BREAK_NEAR";
  }

  if (latestClose >= cloudBottom && latestClose <= cloudTop) {
    return "IN_CLOUD";
  }

  if (latestClose < cloudBottom) {
    return "BELOW_CLOUD";
  }

  return "UNKNOWN";
}

export function calculateTechnicalIndicators(
  bars: PriceBar[],
): TechnicalIndicators {
  const latest = bars[bars.length - 1] ?? null;
  const previous = bars[bars.length - 2] ?? null;
  const bollinger = calculateBollingerBands(bars);
  const previousBollinger = bars.length >= BOLLINGER_PERIOD + 1
    ? calculateBollingerBands(bars.slice(0, -1))
    : null;

  return {
    latestClose: latest?.close ?? null,
    latestVolume: latest?.volume ?? null,
    previousClose: previous?.close ?? null,
    bollinger,
    bbState: classifyBollingerState(bars),
    volumeRebound: classifyVolumeRebound(bars),
    ichimokuState: calculateIchimokuState(bars),
    middleLineRecovered:
      latest && previous && bollinger && previousBollinger
        ? previous.close < previousBollinger.middle &&
          latest.close >= bollinger.middle
        : null,
  };
}

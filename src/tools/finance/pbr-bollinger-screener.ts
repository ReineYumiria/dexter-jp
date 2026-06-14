import { DynamicStructuredTool } from '@langchain/core/tools';
import { z } from 'zod';
import { formatToolResult } from '../types.js';
import { logger } from '../../utils/logger.js';
import { api as edinetApi } from './api.js';
import { resolveEdinetCode } from './resolver.js';
import {
  calculateTechnicalIndicators,
  type PriceBar,
} from './indicators.js';

export type ResearchClassificationCode =
  | 'EXCLUDED'
  | 'DANGER_OBSERVATION'
  | 'STRONG_CAUTION'
  | 'PRIORITY_RESEARCH'
  | 'NORMAL_OBSERVATION'
  | 'LOW_PRIORITY_OBSERVATION';

export type ResearchClassificationLabel =
  | '除外'
  | '危険観察'
  | '強警戒'
  | '優先深掘り'
  | '通常観察'
  | '低優先観察';

export type ResearchCautionBucket =
  | '通常'
  | '注意'
  | '強警戒'
  | '危険観察'
  | '除外';

export type ResearchOutputBucket =
  | '通常候補TSV'
  | '危険観察TSV'
  | '除外';

export type ResearchClassificationResult = {
  code: ResearchClassificationCode;
  label: ResearchClassificationLabel;
  reason: string;
  cautionBucket: ResearchCautionBucket;
  outputBucket: ResearchOutputBucket;
};

export const RESEARCH_CLASSIFICATION_METADATA = {
  EXCLUDED: {
    label: '除外',
    cautionBucket: '除外',
    outputBucket: '除外',
  },
  DANGER_OBSERVATION: {
    label: '危険観察',
    cautionBucket: '危険観察',
    outputBucket: '危険観察TSV',
  },
  STRONG_CAUTION: {
    label: '強警戒',
    cautionBucket: '強警戒',
    outputBucket: '通常候補TSV',
  },
  PRIORITY_RESEARCH: {
    label: '優先深掘り',
    cautionBucket: '通常',
    outputBucket: '通常候補TSV',
  },
  NORMAL_OBSERVATION: {
    label: '通常観察',
    cautionBucket: '通常',
    outputBucket: '通常候補TSV',
  },
  LOW_PRIORITY_OBSERVATION: {
    label: '低優先観察',
    cautionBucket: '注意',
    outputBucket: '通常候補TSV',
  },
} as const satisfies Record<
  ResearchClassificationCode,
  {
    label: ResearchClassificationLabel;
    cautionBucket: ResearchCautionBucket;
    outputBucket: ResearchOutputBucket;
  }
>;

const JQUANTS_BASE = 'https://api.jquants.com/v2';

const DEFAULT_TARGETS = [
  { code: '7203', name: 'トヨタ自動車' },
  { code: '6758', name: 'ソニーグループ' },
  { code: '7974', name: '任天堂' },
] as const;

const NORMAL_CANDIDATE_TSV_HEADER = [
  'コード',
  '銘柄名',
  '市場',
  '業種',
  '終値',
  '出来高',
  '時価総額',
  'PBR',
  'PER',
  '配当利回り',
  '自己資本比率',
  'ROE',
  '営業利益傾向',
  '純利益傾向',
  '営業CF傾向',
  'BB状態',
  'BB位置',
  'ミドルライン回復',
  '出来高反発',
  '一目状態',
  '割安性',
  '財務安全性',
  '成長改善',
  'テクニカル',
  'リスク過熱感',
  '総合点',
  '分類',
  '即除外フラグ',
  '強警戒フラグ',
  '減点フラグ',
  '危険観察フラグ',
  '注意フラグ',
  '除外理由',
  '次に見ること',
  'メモ',
] as const;

type ScreenerTarget = {
  code: string;
  name: string;
};

type JQuantsDailyBar = Record<string, unknown>;

type FinancialMetrics = {
  per: number | null;
  pbr: number | null;
  dividendYield: number | null;
  equityRatio: number | null;
  roe: number | null;
  bps: number | null;
  dps: number | null;
  financialNotes: string[];
};

type TechnicalScore = {
  score: number;
  notes: string[];
};

type ValueScore = {
  score: number;
  notes: string[];
};

type SafetyScore = {
  score: number;
  notes: string[];
};

type RiskScore = {
  score: number;
  notes: string[];
};

type ScreenerResult = {
  code: string;
  name: string;
  latestClose: number | null;
  latestVolume: number | null;
  previousClose: number | null;
  per: number | null;
  pbr: number | null;
  dividendYield: number | null;
  equityRatio: number | null;
  roe: number | null;
  bps: number | null;
  dps: number | null;
  technicalScore: number;
  technicalScoreNotes: string[];
  valueScore: number;
  valueScoreNotes: string[];
  safetyScore: number;
  safetyScoreNotes: string[];
  riskScore: number;
  riskScoreNotes: string[];
  bbState: string;
  bbStateLabel: string;
  bbPosition: number | null;
  middle: number | null;
  upper2: number | null;
  lower2: number | null;
  middleLineRecovered: boolean | null;
  volumeRebound: string;
  ichimokuState: string;
  barCount: number;
  from: string;
  to: string | null;
  notes: string[];
};

function readObject(value: unknown): Record<string, unknown> | null {
  return value && typeof value === 'object' && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function getJQuantsApiKey(): string {
  return process.env.JQUANTS_API_KEY || '';
}

function normalizeSecuritiesCode(code: string): string {
  const digits = code.replace(/\D/g, '');

  if (/^\d{5}$/.test(digits)) {
    return digits.slice(0, 4);
  }

  return digits;
}

function resolveJQuantsCode(code: string): string {
  const normalized = code.replace(/\D/g, '');

  if (/^\d{5}$/.test(normalized)) {
    return normalized;
  }

  if (/^\d{4}$/.test(normalized)) {
    return `${normalized}0`;
  }

  throw new Error(`Invalid securities code: ${code}`);
}

function toIsoDate(value: Date): string {
  return value.toISOString().slice(0, 10);
}

function defaultFromDate(): string {
  const date = new Date();
  date.setDate(date.getDate() - 180);
  return toIsoDate(date);
}

function toNumber(value: unknown): number | null {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value;
  }

  if (typeof value === 'string' && value.trim() !== '') {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }

  return null;
}

function toStringValue(value: unknown): string | null {
  if (typeof value === 'string' && value.trim() !== '') {
    return value;
  }

  if (typeof value === 'number' && Number.isFinite(value)) {
    return String(value);
  }

  return null;
}

function normalizeDailyBar(raw: JQuantsDailyBar): PriceBar | null {
  const date = toStringValue(raw.Date);
  const open = toNumber(raw.AdjO);
  const high = toNumber(raw.AdjH);
  const low = toNumber(raw.AdjL);
  const close = toNumber(raw.AdjC);
  const volume = toNumber(raw.AdjVo);

  if (
    !date ||
    open === null ||
    high === null ||
    low === null ||
    close === null ||
    volume === null
  ) {
    return null;
  }

  return {
    date,
    open,
    high,
    low,
    close,
    volume,
  };
}

async function jquantsGet(
  endpoint: string,
  params: Record<string, string | undefined>,
): Promise<Record<string, unknown>> {
  const apiKey = getJQuantsApiKey();

  if (!apiKey) {
    throw new Error('JQUANTS_API_KEY not set');
  }

  const url = new URL(`${JQUANTS_BASE}${endpoint}`);

  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined) {
      url.searchParams.set(key, value);
    }
  }

  const response = await fetch(url.toString(), {
    headers: { 'x-api-key': apiKey },
  });

  if (!response.ok) {
    const detail = `${response.status} ${response.statusText}`;
    logger.error(`[J-Quants PBR-BB screener] API error: ${detail}`);
    throw new Error(`J-Quants API error: ${detail}`);
  }

  return (await response.json()) as Record<string, unknown>;
}

async function fetchDailyBars(
  code: string,
  from: string,
  to?: string,
): Promise<PriceBar[]> {
  const response = await jquantsGet('/equities/bars/daily', {
    code: resolveJQuantsCode(code),
    from,
    to,
  });

  const rawBars = response.data as JQuantsDailyBar[] | undefined;

  if (!Array.isArray(rawBars)) {
    return [];
  }

  return rawBars
    .map(normalizeDailyBar)
    .filter((bar): bar is PriceBar => bar !== null)
    .sort((a, b) => a.date.localeCompare(b.date));
}

async function fetchCompanyInfo(ticker: string): Promise<Record<string, unknown>> {
  const edinetCode = await resolveEdinetCode(ticker);
  const { data: response } = await edinetApi.get(`/companies/${edinetCode}`, {});
  const responseObject = readObject(response);
  const dataObject = readObject(responseObject?.data);

  return dataObject ?? responseObject ?? {};
}

function calculateFinancialMetrics(
  companyInfo: Record<string, unknown>,
  latestClose: number | null,
): FinancialMetrics {
  const latestFinancials = readObject(companyInfo.latest_financials);

  const per = toNumber(latestFinancials?.per);
  const equityRatio = toNumber(latestFinancials?.equity_ratio_official);
  const roe = toNumber(latestFinancials?.roe_official);
  const bps = toNumber(latestFinancials?.bps);
  const dps = toNumber(latestFinancials?.dividend_per_share);

  const pbr =
    latestClose !== null && bps !== null && bps > 0
      ? latestClose / bps
      : null;

  const dividendYield =
    latestClose !== null && dps !== null && latestClose > 0
      ? dps / latestClose
      : null;

  const financialNotes = [
    'Financial metrics are v0.3 provisional values',
    'PBR is calculated from J-Quants adjusted close and EDINET DB BPS',
    'Dividend yield is calculated from EDINET DB dividend_per_share and J-Quants adjusted close',
  ];

  if (per === null) {
    financialNotes.push('PER_NA: latest_financials.per is missing or invalid');
  }

  if (equityRatio === null) {
    financialNotes.push('EQUITY_RATIO_NA: latest_financials.equity_ratio_official is missing or invalid');
  }

  if (roe === null) {
    financialNotes.push('ROE_NA: latest_financials.roe_official is missing or invalid');
  }

  if (bps === null) {
    financialNotes.push('BPS_NA: latest_financials.bps is missing or invalid');
  }

  if (dps === null) {
    financialNotes.push('DPS_NA: latest_financials.dividend_per_share is missing or invalid');
  }

  if (pbr === null) {
    financialNotes.push('PBR_NA: latestClose or BPS is missing or invalid');
  }

  if (dividendYield === null) {
    financialNotes.push('DIVIDEND_YIELD_NA: latestClose or DPS is missing or invalid');
  }

  return {
    per,
    pbr,
    dividendYield,
    equityRatio,
    roe,
    bps,
    dps,
    financialNotes,
  };
}

function calculateTechnicalScore(result: {
  bbState: string;
  bbPosition: number | null;
  middleLineRecovered: boolean | null;
  volumeRebound: string;
  ichimokuState: string;
}): TechnicalScore {
  let score = 0;
  const notes: string[] = [];

  switch (result.bbState) {
    case 'BB_REBOUND':
      score += 8;
      notes.push('BB position: rebound from below -2σ');
      break;
    case 'BB_LOW_NEAR':
      score += 6;
      notes.push('BB position: near or below lower band');
      break;
    case 'BB_MIDDLE_RECOVER':
      score += 5;
      notes.push('BB position: recovered middle line');
      break;
    case 'BB_NEUTRAL':
      score += 3;
      notes.push('BB position: neutral');
      break;
    case 'BB_DOWN_WALK':
      score += 1;
      notes.push('BB position: lower-band down walk caution');
      break;
    default:
      notes.push('BB position: unknown');
      break;
  }

  if (result.bbState === 'BB_REBOUND') {
    score += 4;
    notes.push('-2σ rebound: confirmed');
  } else {
    notes.push('-2σ rebound: not confirmed');
  }

  if (result.middleLineRecovered === true) {
    score += 3;
    notes.push('Middle line recovery: confirmed');
  } else if (result.middleLineRecovered === false) {
    notes.push('Middle line recovery: not confirmed');
  } else {
    notes.push('Middle line recovery: unknown');
  }

  if (result.volumeRebound === 'STRONG_VOLUME_REBOUND') {
    score += 3;
    notes.push('Volume rebound: strong');
  } else if (result.volumeRebound === 'VOLUME_REBOUND') {
    score += 2;
    notes.push('Volume rebound: confirmed');
  } else if (result.volumeRebound === 'NO_VOLUME_REBOUND') {
    notes.push('Volume rebound: not confirmed');
  } else {
    notes.push('Volume rebound: unknown');
  }

  if (result.ichimokuState === 'ABOVE_CLOUD') {
    score += 2;
    notes.push('Ichimoku: above cloud');
  } else if (result.ichimokuState === 'CLOUD_BREAK_NEAR') {
    score += 2;
    notes.push('Ichimoku: near cloud break');
  } else if (result.ichimokuState === 'IN_CLOUD') {
    score += 1;
    notes.push('Ichimoku: in cloud');
  } else if (result.ichimokuState === 'BELOW_CLOUD') {
    notes.push('Ichimoku: below cloud');
  } else {
    notes.push('Ichimoku: unknown');
  }

  return {
    score,
    notes,
  };
}

function calculateValueScore(result: {
  pbr: number | null;
  per: number | null;
  dividendYield: number | null;
}): ValueScore {
  let score = 0;
  const notes: string[] = [];

  if (result.pbr === null) {
    notes.push('PBR level: unknown');
  } else if (result.pbr <= 0) {
    notes.push('PBR level: invalid');
  } else if (result.pbr <= 0.5) {
    score += 10;
    notes.push('PBR level: 0.5 or lower');
  } else if (result.pbr <= 0.75) {
    score += 9;
    notes.push('PBR level: 0.75 or lower');
  } else if (result.pbr <= 1.0) {
    score += 8;
    notes.push('PBR level: 1.0 or lower');
  } else if (result.pbr <= 1.5) {
    score += 5;
    notes.push('PBR level: 1.5 or lower');
  } else if (result.pbr <= 2.0) {
    score += 3;
    notes.push('PBR level: 2.0 or lower');
  } else {
    notes.push('PBR level: above 2.0');
  }

  let perDividendScore = 0;

  if (result.per === null) {
    notes.push('PER: unknown');
  } else if (result.per > 0 && result.per <= 10) {
    perDividendScore += 2;
    notes.push('PER: 10 or lower');
  } else if (result.per > 0 && result.per <= 15) {
    perDividendScore += 1;
    notes.push('PER: 15 or lower');
  } else if (result.per <= 0) {
    notes.push('PER: invalid or negative');
  } else {
    notes.push('PER: above 15');
  }

  if (result.dividendYield === null) {
    notes.push('Dividend yield: unknown');
  } else if (result.dividendYield >= 0.04) {
    perDividendScore += 2;
    notes.push('Dividend yield: 4% or higher');
  } else if (result.dividendYield >= 0.03) {
    perDividendScore += 1;
    notes.push('Dividend yield: 3% or higher');
  } else {
    notes.push('Dividend yield: below 3%');
  }

  score += Math.min(perDividendScore, 4);

  notes.push('Industry average comparison: not implemented');
  notes.push('Historical range comparison: not implemented');

  return {
    score,
    notes,
  };
}

function calculateSafetyScore(result: {
  equityRatio: number | null;
  roe: number | null;
}): SafetyScore {
  let score = 0;
  const notes: string[] = [];

  if (result.equityRatio === null) {
    notes.push('Equity ratio: unknown');
  } else if (result.equityRatio >= 0.6) {
    score += 5;
    notes.push('Equity ratio: 60% or higher');
  } else if (result.equityRatio >= 0.4) {
    score += 4;
    notes.push('Equity ratio: 40% or higher');
  } else if (result.equityRatio >= 0.3) {
    score += 3;
    notes.push('Equity ratio: 30% or higher');
  } else if (result.equityRatio >= 0.2) {
    score += 1;
    notes.push('Equity ratio: 20% or higher');
  } else {
    notes.push('Equity ratio: below 20%');
  }

  if (result.roe === null) {
    notes.push('ROE: unknown');
  } else if (result.roe >= 0.15) {
    score += 3;
    notes.push('ROE: 15% or higher');
  } else if (result.roe >= 0.1) {
    score += 2;
    notes.push('ROE: 10% or higher');
  } else if (result.roe > 0) {
    score += 1;
    notes.push('ROE: positive');
  } else {
    notes.push('ROE: zero or negative');
  }

  notes.push('Operating profit stability: not implemented');
  notes.push('Net income stability: not implemented');
  notes.push('Operating cash flow / FCF: not implemented');
  notes.push('Interest-bearing debt and financial capacity: not implemented');

  return {
    score,
    notes,
  };
}

function calculateRiskScore(result: {
  bbState: string;
  latestVolume: number | null;
}): RiskScore {
  let score = 0;
  const notes: string[] = [];

  // Not overheated / not aggressively extended: max 4
  switch (result.bbState) {
    case 'BB_NEUTRAL':
      score += 4;
      notes.push('Overheat check: neutral');
      break;
    case 'BB_LOW_NEAR':
      score += 3;
      notes.push('Overheat check: low zone, not overheated');
      break;
    case 'BB_REBOUND':
      score += 3;
      notes.push('Overheat check: rebound, not overheated');
      break;
    case 'BB_MIDDLE_RECOVER':
      score += 2;
      notes.push('Overheat check: middle recovery after decline');
      break;
    case 'BB_DOWN_WALK':
      notes.push('Overheat check: down walk caution');
      break;
    default:
      notes.push('Overheat check: unknown');
      break;
  }

  // Minimum liquidity: max 3
  if (result.latestVolume === null) {
    notes.push('Liquidity: unknown');
  } else if (result.latestVolume >= 5_000_000) {
    score += 3;
    notes.push('Liquidity: 5,000,000 shares or more');
  } else if (result.latestVolume >= 1_000_000) {
    score += 2;
    notes.push('Liquidity: 1,000,000 shares or more');
  } else if (result.latestVolume >= 300_000) {
    score += 1;
    notes.push('Liquidity: 300,000 shares or more');
  } else {
    notes.push('Liquidity: below 300,000 shares');
  }

  notes.push('Securities report risk tolerance: not implemented');
  notes.push('Industry headwind weakness: not implemented');
  notes.push('Undiscovered / low-hype check: not implemented');

  return {
    score,
    notes,
  };
}

function formatBbStateLabel(bbState: string): string {
  switch (bbState) {
    case 'BB_DOWN_WALK':
      return '下限バンド沿い下落';
    case 'BB_REBOUND':
      return '-2σ割れ後の復帰';
    case 'BB_MIDDLE_RECOVER':
      return 'ミドルライン回復';
    case 'BB_LOW_NEAR':
      return '低位ゾーン（-2σ付近または下側）';
    case 'BB_NEUTRAL':
      return '中立';
    case 'BB_UNKNOWN':
      return '不明';
    default:
      return bbState;
  }
}

function formatNumber(value: number | null, digits = 2): string {
  return value === null ? 'NA' : value.toFixed(digits);
}

function formatInteger(value: number | null): string {
  return value === null ? 'NA' : String(Math.round(value));
}

function formatPercent(value: number | null, digits = 2): string {
  return value === null ? 'NA' : `${(value * 100).toFixed(digits)}%`;
}

function formatBoolean(value: boolean | null): string {
  if (value === null) {
    return 'UNKNOWN';
  }

  return value ? 'true' : 'false';
}

function escapeTsvValue(value: string): string {
  return value.replace(/\t/g, ' ').replace(/\r?\n/g, ' ');
}

function buildNormalCandidateTsvRow(result: ScreenerResult): string[] {
  const cautionFlags = [
    '暫定財務指標',
    'PBR暫定計算',
    '配当利回り暫定計算',
    '調整後株価使用',
    '株価財務ソース混在',
  ];

  if (result.pbr !== null && result.pbr <= 1) {
    cautionFlags.push('低PBR警戒');
  }

  if (result.volumeRebound === 'NO_VOLUME_REBOUND') {
    cautionFlags.push('出来高反発未確認');
  }

  if (result.ichimokuState === 'BELOW_CLOUD') {
    cautionFlags.push('一目雲下');
  }

  const nextToCheck = [
    '有報リスク',
    '決算内容',
    'PBR低下理由',
    '業績トレンド',
  ].join(' / ');

  const memo = [
    'v0.3.1 step5',
    `部品点 V=${result.valueScore}/14 S=${result.safetyScore}/8 T=${result.technicalScore}/20 R=${result.riskScore}/7`,
    '合算禁止',
    '暫定指標は注意フラグ参照',
    '総合点なし',
    '分類なし',
    '売買判断なし',
  ].join(' / ');

  return [
    result.code,
    result.name,
    'UNKNOWN',
    'UNKNOWN',
    formatNumber(result.latestClose, 2),
    formatInteger(result.latestVolume),
    'NA',
    formatNumber(result.pbr, 2),
    formatNumber(result.per, 2),
    formatPercent(result.dividendYield, 2),
    formatPercent(result.equityRatio, 1),
    formatPercent(result.roe, 1),
    '未実装',
    '未実装',
    '未実装',
    `${result.bbState} (${result.bbStateLabel})`,
    formatNumber(result.bbPosition, 4),
    formatBoolean(result.middleLineRecovered),
    result.volumeRebound,
    result.ichimokuState,
    formatInteger(result.valueScore),
    formatInteger(result.safetyScore),
    '未実装',
    formatInteger(result.technicalScore),
    formatInteger(result.riskScore),
    '未実装',
    '未分類',
    'false',
    'false',
    'false',
    'false',
    cautionFlags.join(','),
    '',
    nextToCheck,
    memo,
  ].map(escapeTsvValue);
}

function buildTsv(results: ScreenerResult[]): {
  header: readonly string[];
  rows: string[][];
  text: string;
} {
  const rows = results.map(buildNormalCandidateTsvRow);
  const text = [
    NORMAL_CANDIDATE_TSV_HEADER.join('\t'),
    ...rows.map((row) => row.join('\t')),
  ].join('\n');

  return {
    header: NORMAL_CANDIDATE_TSV_HEADER,
    rows,
    text,
  };
}

function parseTargets(inputTargets?: string[]): ScreenerTarget[] {
  if (!inputTargets || inputTargets.length === 0) {
    return [...DEFAULT_TARGETS];
  }

  return inputTargets.map((code) => {
    const normalizedCode = normalizeSecuritiesCode(code);

    return {
      code: normalizedCode,
      name:
        DEFAULT_TARGETS.find((target) => target.code === normalizedCode)?.name ??
        'UNKNOWN',
    };
  });
}

export const PBR_BOLLINGER_SCREENER_DESCRIPTION = `
Runs a minimal PBR x Bollinger Band screener step for Japanese equities.

v0.3.1 step 5:
- Fetches daily adjusted OHLCV from J-Quants
- Calculates Bollinger Band state
- Calculates volume rebound state
- Calculates simplified Ichimoku cloud state
- Fetches provisional financial metrics from EDINET DB
- Calculates provisional PBR and dividend yield from adjusted close and per-share values
- Calculates provisional value, financial safety, and technical component scores
- Calculates provisional financial safety component score from implemented fields only
- Calculates provisional risk / overheat component score from implemented fields only
- Clarifies that component scores are incomplete and must not be summed as a total score
- Clarifies provisional calculation sources for PBR and dividend yield
- Adds display labels for BB states while keeping internal enum values unchanged
- Keeps TSV memo compact and moves detailed warnings to caution flags and notes
- Confirms TSV output is intended for spreadsheet paste workflow
- Does not calculate total score or A/B/C classification
- Does not provide buy/sell recommendations

v0.4 step 1:
- Defines research classification labels before total score implementation
- Labels are for research prioritization only, not buy/sell judgment
- A/B/C labels must not be presented as investment ranks
- "A候補" means "優先深掘り候補", not "買い候補"
- Keeps component scores non-additive
- Does not calculate total score yet
- Does not provide buy/sell recommendations

Research classification label definitions:
- 優先深掘り: Low valuation, financial safety, and technical reaction signals are relatively aligned; prioritized for additional research only.
- 通常観察: Some conditions are met, but one or more of valuation, safety, technical, or risk factors require confirmation.
- 低優先観察: Signal alignment is weak or there are many unresolved confirmation points.
- 強警戒: Attractive-looking valuation or reaction signs exist, but important caution factors are present.
- 危険観察: Useful for research, but prone to misuse as an investment signal; should be separated from normal candidates.
- 除外: Outside the current research target due to missing data, abnormal indicators, low liquidity, weak safety, or rule mismatch.

v0.4 step 2:
- Defines classification decision order before implementing classification output
- Applies exclusion and danger-observation rules before any positive research label
- Keeps "優先深掘り" as a research-priority label only
- Does not output classification columns yet
- Does not calculate total score yet

Classification decision order:
1. 除外: Remove from normal research candidates when data is insufficient, indicators are abnormal, liquidity is too low, or the company does not match the current research scope.
2. 危険観察: Separate from normal candidates when low valuation or technical reaction exists but the signal is likely to be misused, such as sharp decline, extreme risk, severe financial weakness, or unstable data.
3. 強警戒: Keep visible as a caution-heavy research item when attractive-looking signals exist but important warnings are present.
4. 優先深掘り: Use only when valuation, financial safety, and technical reaction signals are relatively aligned and risk warnings are not dominant.
5. 通常観察: Use when some signals are present but confirmation points remain.
6. 低優先観察: Use when signal alignment is weak or research priority is low.

v0.4 step 3:
- Defines concrete exclusion conditions before implementing classification logic
- Exclusion rules are applied before danger-observation, caution, or positive research-priority labels
- Exclusion means "outside the current research target", not "sell" or "bad company"
- Excluded items should not be mixed into normal candidate TSV output unless explicitly requested

Exclusion condition draft:
- Missing essential price data required for BB or technical calculations
- Missing essential financial indicators required for valuation or safety checks
- Abnormal or unusable indicator values, such as negative or zero BPS when PBR-based screening is required
- Extremely low liquidity where price reaction signals may be unreliable
- Security is outside the current research scope
- Calculation result contains insufficient confidence for normal comparison

v0.4 step 4:
- Defines danger-observation conditions before implementing classification logic
- Danger-observation items are research-only and should be separated from normal candidates
- Danger-observation does not mean sell, short, or automatic rejection
- Low PBR, BB lower-band contact, or technical rebound may not override danger-observation conditions

Danger-observation condition draft:
- Very high riskScore caused by multiple caution factors
- Low PBR appears together with severe financial weakness
- Technical rebound signal appears after a sharp decline without enough safety confirmation
- BB lower-band contact appears with weak liquidity or unstable price data
- Negative or deteriorating profitability indicators require separate observation
- Data source limitations make the apparent undervaluation unreliable
- Strong caution flags dominate value, safety, or technical signals

v0.4 step 5:
- Defines strong-caution conditions before implementing classification logic
- Strong-caution items may remain visible, but must not be treated as positive research-priority labels
- Strong-caution does not mean sell, short, or automatic rejection
- Positive value, safety, or technical signals may not erase strong caution flags

Strong-caution condition draft:
- riskScore is elevated, but not severe enough for danger-observation
- Low PBR exists, but safetyScore is weak or caution flags remain important
- Technical reaction exists, but volume confirmation is weak or unstable
- Ichimoku signals are mixed or not enough to support the technical context
- Profitability or capital safety indicators require additional confirmation
- Multiple minor caution flags exist across value, safety, technical, or data reliability

v0.4 step 6:
- Defines priority-research conditions before implementing classification logic
- Priority-research means "優先深掘り", not buy candidate or investment rank
- Priority-research can only be assigned after exclusion, danger-observation, and strong-caution checks
- Priority-research requires aligned signals across valuation, safety, and technical context
- Low PBR alone, BB lower-band contact alone, or technical rebound alone is insufficient

Priority-research condition draft:
- valueScore is relatively favorable, but not used alone
- safetyScore is sufficient and no severe financial caution dominates
- technicalScore shows supportive reaction context, but not as a standalone reason
- riskScore is not dominant and strong caution flags are not present
- PBR-based undervaluation is supported by usable BPS and financial data
- Volume or trend context provides enough confirmation for research prioritization
- Candidate remains a research target requiring human review

v0.4 step 7:
- Defines normal-observation and low-priority-observation conditions before implementing classification logic
- Normal-observation means the item remains worth watching, not that it is attractive for investment
- Low-priority-observation means research priority is currently low, not that the company is bad
- These labels are only assigned after exclusion, danger-observation, strong-caution, and priority-research checks

Normal-observation condition draft:
- Some value, safety, or technical signals are present
- Signal alignment is incomplete or mixed
- Confirmation points remain across valuation, safety, technical, or data reliability
- No exclusion, danger-observation, or strong-caution condition dominates
- Candidate may be reviewed later as part of research monitoring

Low-priority-observation condition draft:
- Signal alignment is weak
- Value, safety, or technical scores do not clearly support deeper research
- Confirmation points are numerous or unresolved
- No urgent research-priority signal is present
- Candidate remains in the research universe but is not prioritized

v0.4 step 8:
- Defines TSV separation policy before implementing classification output
- Normal research candidates and danger-observation items should be separable
- Danger-observation items should not be mixed into the default normal candidate TSV
- Excluded items should not be output by default
- Classification labels may be added later only as research-priority labels
- Spreadsheet workflow should keep normal candidates and danger-observation items visually separated

TSV separation draft:
- Default TSV: normal research candidates only
- Danger-observation TSV: separate optional output for research review
- Excluded items: omitted by default, optionally summarized in notes or diagnostics later
- Strong-caution items: may remain in default TSV only if clearly labeled and not treated as positive candidates
- Priority-research / normal-observation / low-priority-observation: may share the normal candidate TSV if labels are explicitly research-priority labels

v0.4 step 9:
- Defines display column names before implementing classification output
- Classification columns should use Japanese labels in spreadsheet output
- A/B/C-style rank labels should not be shown as primary display labels
- Classification columns are for research organization only, not investment judgment
- Column design must keep normal candidates and danger-observation items distinguishable

Classification output column draft:
- 研究分類: Human-readable research label such as 優先深掘り, 通常観察, 低優先観察, 強警戒, 危険観察, or 除外.
- 分類コード: Internal stable code for sorting or future logic; not intended as an investment rank.
- 分類理由: Short explanation of why the research classification was assigned.
- 注意区分: Caution bucket such as 通常, 注意, 強警戒, 危険観察, or 除外.
- 出力枠: Output bucket such as 通常候補TSV, 危険観察TSV, or 除外.

v0.4 step 10:
- Defines internal classification codes before implementing classification output
- Codes are for sorting and stable processing only, not investment ranks
- Display labels should remain Japanese research labels
- A/B/C-style display should not be used as the primary spreadsheet output

Classification code draft:
- EXCLUDED: 除外
- DANGER_OBSERVATION: 危険観察
- STRONG_CAUTION: 強警戒
- PRIORITY_RESEARCH: 優先深掘り
- NORMAL_OBSERVATION: 通常観察
- LOW_PRIORITY_OBSERVATION: 低優先観察

Code display rule:
- Internal codes may be used for filtering, sorting, testing, or future logic.
- Spreadsheet users should primarily read 研究分類, 注意区分, and 出力枠.
- Codes must not be described as investment ranks or buy/sell signals.

v0.4 step 11:
- Defines TypeScript type design before implementing classification logic
- Classification types should preserve separation between display labels, internal codes, caution bucket, and output bucket
- Types are for research organization only and must not represent investment judgment
- Classification logic should be implemented only after these types are reviewed

Type design draft:
- ResearchClassificationCode: EXCLUDED | DANGER_OBSERVATION | STRONG_CAUTION | PRIORITY_RESEARCH | NORMAL_OBSERVATION | LOW_PRIORITY_OBSERVATION
- ResearchClassificationLabel: 除外 | 危険観察 | 強警戒 | 優先深掘り | 通常観察 | 低優先観察
- ResearchCautionBucket: 通常 | 注意 | 強警戒 | 危険観察 | 除外
- ResearchOutputBucket: 通常候補TSV | 危険観察TSV | 除外
- ResearchClassificationResult: code, label, reason, cautionBucket, outputBucket

v0.4 step 12:
- Adds TypeScript classification types and metadata map
- Adds ResearchClassificationCode, ResearchClassificationLabel, ResearchCautionBucket, ResearchOutputBucket, and ResearchClassificationResult
- Adds RESEARCH_CLASSIFICATION_METADATA for code-to-label, caution bucket, and output bucket mapping
- Keeps classification logic unimplemented
- Does not change TSV output
- Does not calculate total score
- Does not provide buy/sell recommendations

v0.4 step 13:
- Defines classification helper function design before implementing logic
- Helper function should return ResearchClassificationResult
- Helper function must apply exclusion, danger-observation, and strong-caution checks before positive research labels
- Helper function must not calculate total score
- Helper function must not provide buy/sell recommendations
- Helper function must not merge component scores into a single investment score

Classification helper draft:
- Function name: classifyResearchCandidate
- Input: existing candidate metrics, component scores, caution flags, and data-confidence context
- Output: ResearchClassificationResult
- Decision order: EXCLUDED → DANGER_OBSERVATION → STRONG_CAUTION → PRIORITY_RESEARCH → NORMAL_OBSERVATION → LOW_PRIORITY_OBSERVATION
- Reason field should be short and spreadsheet-friendly
- Classification should be treated as research organization only

v0.4 step 14:
- Defines classification input design before implementing helper logic
- Classification input should use existing component scores and caution context without merging them into a total investment score
- Input design should preserve separation between value, safety, technical, risk, data quality, and output-bucket decisions
- Missing or unreliable data must remain visible to the classification helper
- Input design must not imply buy/sell judgment or investment ranking

Classification input draft:
- Candidate identity: code, name, market, and available company identifiers
- Valuation context: PBR, PER, dividend yield, BPS, DPS, and valueScore
- Safety context: equity ratio, ROE, safetyScore, and financial caution flags
- Technical context: BB state, BB position, volume reaction, Ichimoku summary, and technicalScore
- Risk context: riskScore, caution flags, abnormal indicator flags, low-liquidity flags, and sharp-decline context
- Data confidence context: missing data, provisional source notes, calculation confidence, and unusable indicator reasons
- Output context: whether the item can remain in normal candidate TSV, should move to danger-observation TSV, or should be excluded

Input guard rule draft:
- Classification input must not collapse component scores into a single total score
- Classification input must keep caution flags available even when value or technical signals look favorable
- Classification input must allow exclusion and danger-observation checks to run before positive research labels
- Classification input must support short, spreadsheet-friendly classification reasons

v0.4 step 15:
- Defines classification input type design before implementing TypeScript input types
- Input type should group candidate identity, valuation, safety, technical, risk, data confidence, and output context
- Input type should avoid a single totalScore field
- Input type should keep component scores and caution flags independently available
- Input type should support exclusion and danger-observation checks before positive research labels

Classification input type draft:
- ResearchClassificationInput:
  - identity: candidate code, name, market, and optional identifiers
  - valuation: PBR, PER, dividend yield, BPS, DPS, valueScore, and provisional source notes
  - safety: equity ratio, ROE, safetyScore, and financial caution flags
  - technical: BB state, BB position, volume reaction, Ichimoku summary, technicalScore, and technical caution flags
  - risk: riskScore, caution flags, abnormal indicator flags, low-liquidity flags, and sharp-decline context
  - dataConfidence: missing fields, unreliable fields, unusable indicator reasons, and calculation confidence notes
  - outputPolicy: canUseNormalCandidateTsv, canUseDangerObservationTsv, shouldExcludeByDefault

Input type guard rule draft:
- Do not add totalScore to ResearchClassificationInput
- Do not add buy/sell/hold fields
- Do not encode A/B/C ranking as primary output
- Keep reason generation separate from score calculation
- Keep output bucket decisions separate from score values

Guard rule:
No positive research label may override exclusion, danger-observation, or strong-caution conditions.

This tool is for research candidate extraction only.
`.trim();

const PbrBollingerScreenerInputSchema = z.object({
  codes: z
    .array(z.string())
    .optional()
    .describe(
      "Optional securities codes. If omitted, uses 7203, 6758, and 7974."
    ),
  from: z
    .string()
    .optional()
    .describe(
      'Start date for daily bars, YYYY-MM-DD. If omitted, uses about 180 calendar days ago.'
    ),
  to: z
    .string()
    .optional()
    .describe('End date for daily bars, YYYY-MM-DD. If omitted, uses latest available.'),
});

export const getPbrBollingerScreener = new DynamicStructuredTool({
  name: 'get_pbr_bollinger_screener',
  description: PBR_BOLLINGER_SCREENER_DESCRIPTION,
  schema: PbrBollingerScreenerInputSchema,
  func: async (input) => {
    const targets = parseTargets(input.codes);
    const from = input.from ?? defaultFromDate();

    const results: ScreenerResult[] = await Promise.all(
      targets.map(async (target) => {
        try {
          const bars = await fetchDailyBars(target.code, from, input.to);
          const indicators = calculateTechnicalIndicators(bars);
          const bollinger = indicators.bollinger;
          const companyInfo = await fetchCompanyInfo(target.code);
          const financialMetrics = calculateFinancialMetrics(
            companyInfo,
            indicators.latestClose,
          );
          const technicalScore = calculateTechnicalScore({
            bbState: indicators.bbState,
            bbPosition: bollinger?.bbPosition ?? null,
            middleLineRecovered: indicators.middleLineRecovered,
            volumeRebound: indicators.volumeRebound,
            ichimokuState: indicators.ichimokuState,
          });
          const valueScore = calculateValueScore({
            pbr: financialMetrics.pbr,
            per: financialMetrics.per,
            dividendYield: financialMetrics.dividendYield,
          });
          const safetyScore = calculateSafetyScore({
            equityRatio: financialMetrics.equityRatio,
            roe: financialMetrics.roe,
          });
          const riskScore = calculateRiskScore({
            bbState: indicators.bbState,
            latestVolume: indicators.latestVolume,
          });

          const notes = [
            'v0.3 step 7: value, financial safety, and technical component scores only',
            'Uses adjusted OHLCV from existing J-Quants field policy: AdjO/AdjH/AdjL/AdjC/AdjVo',
            'PBR, dividend yield, and per-share derived metrics are provisional',
            'Value score uses only implemented subitems; industry average and historical range are not implemented',
            'Financial safety score uses only implemented subitems; profit stability, cash flow, and debt capacity are not implemented',
            'Technical score is a provisional 20-point component',
            'Total score and A/B/C classification are not implemented yet',
            'Research candidate extraction only; no buy/sell recommendation',
          ];

          if (bars.length < 52) {
            notes.push('Insufficient bars for full Ichimoku calculation; ichimokuState may be UNKNOWN');
          }

          return {
            code: target.code,
            name: target.name,
            latestClose: indicators.latestClose,
            latestVolume: indicators.latestVolume,
            previousClose: indicators.previousClose,
            per: financialMetrics.per,
            pbr: financialMetrics.pbr,
            dividendYield: financialMetrics.dividendYield,
            equityRatio: financialMetrics.equityRatio,
            roe: financialMetrics.roe,
            bps: financialMetrics.bps,
            dps: financialMetrics.dps,
            technicalScore: technicalScore.score,
            technicalScoreNotes: technicalScore.notes,
            valueScore: valueScore.score,
            valueScoreNotes: valueScore.notes,
            safetyScore: safetyScore.score,
            safetyScoreNotes: safetyScore.notes,
            riskScore: riskScore.score,
            riskScoreNotes: riskScore.notes,
            bbState: indicators.bbState,
            bbStateLabel: formatBbStateLabel(indicators.bbState),
            bbPosition: bollinger?.bbPosition ?? null,
            middle: bollinger?.middle ?? null,
            upper2: bollinger?.upper2 ?? null,
            lower2: bollinger?.lower2 ?? null,
            middleLineRecovered: indicators.middleLineRecovered,
            volumeRebound: indicators.volumeRebound,
            ichimokuState: indicators.ichimokuState,
            barCount: bars.length,
            from,
            to: input.to ?? null,
            notes: [
              ...notes,
              ...financialMetrics.financialNotes,
              ...valueScore.notes,
              ...safetyScore.notes,
              ...technicalScore.notes,
            ],
          };
        } catch (error) {
          return {
            code: target.code,
            name: target.name,
            latestClose: null,
            latestVolume: null,
            previousClose: null,
            per: null,
            pbr: null,
            dividendYield: null,
            equityRatio: null,
            roe: null,
            bps: null,
            dps: null,
            technicalScore: 0,
            technicalScoreNotes: ['Technical score unavailable because calculation failed'],
            valueScore: 0,
            valueScoreNotes: ['Value score unavailable because calculation failed'],
            safetyScore: 0,
            safetyScoreNotes: ['Safety score unavailable because calculation failed'],
            riskScore: 0,
            riskScoreNotes: ['Risk / overheat score unavailable because calculation failed'],
            bbState: 'BB_UNKNOWN',
            bbStateLabel: formatBbStateLabel('BB_UNKNOWN'),
            bbPosition: null,
            middle: null,
            upper2: null,
            lower2: null,
            middleLineRecovered: null,
            volumeRebound: 'VOLUME_UNKNOWN',
            ichimokuState: 'UNKNOWN',
            barCount: 0,
            from,
            to: input.to ?? null,
            notes: [
              'Failed to fetch or calculate value, safety, technical, and financial metrics',
              error instanceof Error ? error.message : String(error),
              'Research candidate extraction only; no buy/sell recommendation',
            ],
          };
        }
      }),
    );

    const tsv = buildTsv(results);

    return formatToolResult(
      {
        version: 'v0.3.1-step5',
        scope: 'spreadsheet_ready_component_scores_tsv',
        targets: targets.map((target) => target.code),
        results,
        tsv,
        notes: [
          'This is not investment advice',
          'No buy/sell recommendation is provided',
          'Component scores are incomplete and must not be summed as a total score',
          'BB state labels are display helpers; internal enum values are unchanged',
          'BB_LOW_NEAR display label clarifies that it includes near or below the lower band',
          'PBR is provisionally calculated as J-Quants adjusted close divided by EDINET DB BPS',
          'Dividend yield is provisionally calculated as EDINET DB DPS divided by J-Quants adjusted close',
          'Value score is included as a provisional component with a maximum implemented score of 14 points',
          'Financial safety score is included as a provisional component with a maximum implemented score of 8 points',
          'Technical score is included as a provisional 20-point component',
          'Risk / overheat score is included as a provisional component with a maximum implemented score of 7 points',
          'Growth / improvement score is intentionally not implemented',
          'Financial metrics are included as provisional values',
          'TSV output is included for research candidate review',
          'TSV memo is intentionally compact; check caution flags and notes for detailed warnings',
          'TSV output is intended for spreadsheet paste workflow',
          'Total score and A/B/C classification are intentionally not implemented in v0.3.1 step 5',
        ],
      },
      [],
    );
  },
});

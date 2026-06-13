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
  bbState: string;
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
    '調整後株価使用',
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
    'v0.3 step4 TSV出力',
    '暫定財務指標',
    '調整後株価使用',
    'スコアリング未実装',
    '分類未実装',
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
    result.bbState,
    formatNumber(result.bbPosition, 4),
    formatBoolean(result.middleLineRecovered),
    result.volumeRebound,
    result.ichimokuState,
    '未実装',
    '未実装',
    '未実装',
    '未実装',
    '未実装',
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

v0.3 step 3:
- Fetches daily adjusted OHLCV from J-Quants
- Calculates Bollinger Band state
- Calculates volume rebound state
- Calculates simplified Ichimoku cloud state
- Fetches provisional financial metrics from EDINET DB
- Calculates provisional PBR and dividend yield from adjusted close and per-share values
- Does not calculate scoring or A/B/C classification
- Does not provide buy/sell recommendations

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

          const notes = [
            'v0.3 step 3: technical and provisional financial metrics only',
            'Uses adjusted OHLCV from existing J-Quants field policy: AdjO/AdjH/AdjL/AdjC/AdjVo',
            'PBR, dividend yield, and per-share derived metrics are provisional',
            'Scoring and A/B/C classification are not implemented yet',
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
            bbState: indicators.bbState,
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
            notes: [...notes, ...financialMetrics.financialNotes],
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
            bbState: 'BB_UNKNOWN',
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
              'Failed to fetch or calculate technical and financial metrics',
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
        version: 'v0.3-step4',
        scope: 'technical_and_financial_metrics_with_tsv',
        targets: targets.map((target) => target.code),
        results,
        tsv,
        notes: [
          'This is not investment advice',
          'No buy/sell recommendation is provided',
          'Financial metrics are included as provisional values',
          'TSV output is included for research candidate review',
          'Scoring and A/B/C classification are intentionally not implemented in step 4',
        ],
      },
      [],
    );
  },
});

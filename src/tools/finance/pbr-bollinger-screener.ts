import { DynamicStructuredTool } from '@langchain/core/tools';
import { z } from 'zod';
import { formatToolResult } from '../types.js';
import { logger } from '../../utils/logger.js';
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

type ScreenerTarget = {
  code: string;
  name: string;
};

type JQuantsDailyBar = Record<string, unknown>;

function getJQuantsApiKey(): string {
  return process.env.JQUANTS_API_KEY || '';
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

function parseTargets(inputTargets?: string[]): ScreenerTarget[] {
  if (!inputTargets || inputTargets.length === 0) {
    return [...DEFAULT_TARGETS];
  }

  return inputTargets.map((code) => ({
    code,
    name:
      DEFAULT_TARGETS.find((target) => target.code === code)?.name ??
      'UNKNOWN',
  }));
}

export const PBR_BOLLINGER_SCREENER_DESCRIPTION = `
Runs a minimal PBR x Bollinger Band screener step for Japanese equities.

v0.3 step 2 only:
- Fetches daily adjusted OHLCV from J-Quants
- Calculates Bollinger Band state
- Calculates volume rebound state
- Calculates simplified Ichimoku cloud state
- Does not calculate PBR, PER, dividends, financial safety, scoring, or A/B/C classification
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

    const results = await Promise.all(
      targets.map(async (target) => {
        try {
          const bars = await fetchDailyBars(target.code, from, input.to);
          const indicators = calculateTechnicalIndicators(bars);
          const bollinger = indicators.bollinger;

          const notes = [
            'v0.3 step 2: technical indicators only',
            'Uses adjusted OHLCV from existing J-Quants field policy: AdjO/AdjH/AdjL/AdjC/AdjVo',
            'Financial data, PBR, PER, dividends, scoring, and A/B/C classification are not implemented yet',
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
            notes,
          };
        } catch (error) {
          return {
            code: target.code,
            name: target.name,
            latestClose: null,
            latestVolume: null,
            previousClose: null,
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
              'Failed to fetch or calculate technical indicators',
              error instanceof Error ? error.message : String(error),
              'Research candidate extraction only; no buy/sell recommendation',
            ],
          };
        }
      }),
    );

    return formatToolResult(
      {
        version: 'v0.3-step2',
        scope: 'technical_only',
        targets: targets.map((target) => target.code),
        results,
        notes: [
          'This is not investment advice',
          'No buy/sell recommendation is provided',
          'PBR, financials, scoring, and classification are intentionally not implemented in step 2',
        ],
      },
      [],
    );
  },
});

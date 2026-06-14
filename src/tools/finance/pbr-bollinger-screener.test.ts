import { describe, expect, test } from 'bun:test';
import {
  classifyResearchCandidate,
  DEFAULT_RESEARCH_CLASSIFICATION_THRESHOLDS,
  type ResearchClassificationInput,
} from './pbr-bollinger-screener.js';

type ClassificationInputOverrides = {
  identity?: Partial<ResearchClassificationInput['identity']>;
  valuation?: Partial<ResearchClassificationInput['valuation']>;
  safety?: Partial<ResearchClassificationInput['safety']>;
  technical?: Partial<ResearchClassificationInput['technical']>;
  risk?: Partial<ResearchClassificationInput['risk']>;
  dataConfidence?: Partial<ResearchClassificationInput['dataConfidence']>;
  outputPolicy?: Partial<ResearchClassificationInput['outputPolicy']>;
};

function createClassificationInput(
  overrides: ClassificationInputOverrides = {},
): ResearchClassificationInput {
  const base: ResearchClassificationInput = {
    identity: {
      code: '7203',
      name: 'Test Company',
      market: 'Prime',
      identifiers: {},
    },
    valuation: {
      pbr: 0.9,
      per: 12,
      dividendYield: 0.03,
      bps: 1000,
      dps: 30,
      valueScore: 8,
      provisionalSourceNotes: [],
    },
    safety: {
      equityRatio: 0.45,
      roe: 0.1,
      safetyScore: 5,
      financialCautionFlags: [],
    },
    technical: {
      bbState: 'BB_NEUTRAL',
      bbPosition: 0,
      volumeReaction: 'NO_VOLUME_REBOUND',
      ichimokuSummary: 'IN_CLOUD',
      technicalScore: 10,
      technicalCautionFlags: [],
    },
    risk: {
      riskScore: 5,
      cautionFlags: [],
      abnormalIndicatorFlags: [],
      lowLiquidityFlags: [],
      sharpDeclineContext: [],
    },
    dataConfidence: {
      missingFields: [],
      unreliableFields: [],
      unusableIndicatorReasons: [],
      calculationConfidenceNotes: [],
    },
    outputPolicy: {
      canUseNormalCandidateTsv: true,
      canUseDangerObservationTsv: true,
      shouldExcludeByDefault: false,
    },
  };

  return {
    ...base,
    ...overrides,
    identity: {
      ...base.identity,
      ...overrides.identity,
    },
    valuation: {
      ...base.valuation,
      ...overrides.valuation,
    },
    safety: {
      ...base.safety,
      ...overrides.safety,
    },
    technical: {
      ...base.technical,
      ...overrides.technical,
    },
    risk: {
      ...base.risk,
      ...overrides.risk,
    },
    dataConfidence: {
      ...base.dataConfidence,
      ...overrides.dataConfidence,
    },
    outputPolicy: {
      ...base.outputPolicy,
      ...overrides.outputPolicy,
    },
  };
}

describe('classifyResearchCandidate', () => {
  test('returns EXCLUDED when shouldExcludeByDefault is true', () => {
    const result = classifyResearchCandidate(
      createClassificationInput({
        outputPolicy: {
          shouldExcludeByDefault: true,
        },
      }),
    );

    expect(result.code).toBe('EXCLUDED');
  });

  test('returns EXCLUDED when missing fields reach the exclusion threshold', () => {
    const result = classifyResearchCandidate(
      createClassificationInput({
        dataConfidence: {
          missingFields: Array.from(
            {
              length:
                DEFAULT_RESEARCH_CLASSIFICATION_THRESHOLDS.dataQuality
                  .missingFieldCountForExclusion,
            },
            (_, index) => `field-${index}`,
          ),
        },
      }),
    );

    expect(result.code).toBe('EXCLUDED');
  });

  test('returns DANGER_OBSERVATION when riskScore reaches the danger threshold', () => {
    const result = classifyResearchCandidate(
      createClassificationInput({
        risk: {
          riskScore:
            DEFAULT_RESEARCH_CLASSIFICATION_THRESHOLDS.risk
              .dangerObservationMaxRiskScore,
        },
      }),
    );

    expect(result.code).toBe('DANGER_OBSERVATION');
  });

  test('returns STRONG_CAUTION when riskScore reaches strong caution but not danger', () => {
    const result = classifyResearchCandidate(
      createClassificationInput({
        risk: {
          riskScore:
            DEFAULT_RESEARCH_CLASSIFICATION_THRESHOLDS.risk
              .strongCautionMaxRiskScore,
        },
      }),
    );

    expect(result.code).toBe('STRONG_CAUTION');
  });

  test('returns NORMAL_OBSERVATION when no negative condition applies', () => {
    const result = classifyResearchCandidate(createClassificationInput());

    expect(result.code).toBe('NORMAL_OBSERVATION');
  });

  test('prioritizes DANGER_OBSERVATION over STRONG_CAUTION', () => {
    const result = classifyResearchCandidate(
      createClassificationInput({
        risk: {
          riskScore:
            DEFAULT_RESEARCH_CLASSIFICATION_THRESHOLDS.risk
              .dangerObservationMaxRiskScore,
        },
        safety: {
          safetyScore:
            DEFAULT_RESEARCH_CLASSIFICATION_THRESHOLDS.safety
              .weakSafetyScoreThreshold,
        },
      }),
    );

    expect(result.code).toBe('DANGER_OBSERVATION');
  });
});

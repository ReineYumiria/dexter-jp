import { describe, expect, test } from 'bun:test';
import {
  buildNormalCandidateTsvRow,
  buildResearchClassificationInput,
  classifyResearchCandidate,
  DEFAULT_RESEARCH_CLASSIFICATION_THRESHOLDS,
  NORMAL_CANDIDATE_TSV_HEADER,
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

type TestScreenerResult = Parameters<typeof buildResearchClassificationInput>[0];

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

function createPriorityResearchInput(
  overrides: ClassificationInputOverrides = {},
): ResearchClassificationInput {
  return createClassificationInput({
    ...overrides,
    valuation: {
      pbr:
        DEFAULT_RESEARCH_CLASSIFICATION_THRESHOLDS.value
          .maximumPbrForPriorityResearch,
      valueScore:
        DEFAULT_RESEARCH_CLASSIFICATION_THRESHOLDS.value
          .minimumValueScoreForPriorityResearch,
      ...overrides.valuation,
    },
    safety: {
      safetyScore:
        DEFAULT_RESEARCH_CLASSIFICATION_THRESHOLDS.safety
          .minimumSafetyScoreForPriorityResearch,
      ...overrides.safety,
    },
    technical: {
      bbState: 'BB_LOW_NEAR',
      technicalScore:
        DEFAULT_RESEARCH_CLASSIFICATION_THRESHOLDS.technical
          .minimumTechnicalScoreForPriorityResearch,
      ...overrides.technical,
    },
    risk: {
      riskScore:
        DEFAULT_RESEARCH_CLASSIFICATION_THRESHOLDS.risk
          .minimumRiskScoreForPriorityResearch,
      ...overrides.risk,
    },
  });
}

function createScreenerResult(
  overrides: Partial<TestScreenerResult> = {},
): TestScreenerResult {
  return {
    code: '7203',
    name: 'Test Company',
    latestClose: 1000,
    latestVolume: 1_000_000,
    previousClose: 990,
    per: 12,
    pbr: 0.9,
    dividendYield: 0.03,
    equityRatio: 0.45,
    roe: 0.1,
    bps: 1100,
    dps: 30,
    technicalScore: 10,
    technicalScoreNotes: ['technical note'],
    valueScore: 8,
    valueScoreNotes: ['value note'],
    safetyScore: 5,
    safetyScoreNotes: ['safety note'],
    riskScore: 5,
    riskScoreNotes: ['risk note'],
    bbState: 'BB_LOW_NEAR',
    bbStateLabel: 'low zone',
    bbPosition: -1.8,
    middle: 1000,
    upper2: 1100,
    lower2: 900,
    middleLineRecovered: true,
    volumeRebound: 'VOLUME_REBOUND',
    ichimokuState: 'IN_CLOUD',
    barCount: 120,
    from: '2026-01-01',
    to: '2026-06-18',
    notes: ['candidate note'],
    ...overrides,
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

  test('returns PRIORITY_RESEARCH when all research conditions align', () => {
    const result = classifyResearchCandidate(createPriorityResearchInput());

    expect(result.code).toBe('PRIORITY_RESEARCH');
  });

  test('does not return PRIORITY_RESEARCH from low PBR alone', () => {
    const result = classifyResearchCandidate(
      createClassificationInput({
        valuation: {
          pbr:
            DEFAULT_RESEARCH_CLASSIFICATION_THRESHOLDS.value
              .maximumPbrForPriorityResearch,
          valueScore:
            DEFAULT_RESEARCH_CLASSIFICATION_THRESHOLDS.value
              .minimumValueScoreForPriorityResearch - 1,
        },
      }),
    );

    expect(result.code).not.toBe('PRIORITY_RESEARCH');
  });

  test('does not return PRIORITY_RESEARCH from BB_LOW_NEAR alone', () => {
    const result = classifyResearchCandidate(
      createClassificationInput({
        valuation: {
          pbr: null,
          valueScore:
            DEFAULT_RESEARCH_CLASSIFICATION_THRESHOLDS.value
              .minimumValueScoreForPriorityResearch - 1,
        },
        technical: {
          bbState: 'BB_LOW_NEAR',
        },
      }),
    );

    expect(result.code).not.toBe('PRIORITY_RESEARCH');
  });

  test('does not return PRIORITY_RESEARCH from BB_REBOUND alone', () => {
    const result = classifyResearchCandidate(
      createClassificationInput({
        valuation: {
          pbr: null,
          valueScore:
            DEFAULT_RESEARCH_CLASSIFICATION_THRESHOLDS.value
              .minimumValueScoreForPriorityResearch - 1,
        },
        technical: {
          bbState: 'BB_REBOUND',
        },
      }),
    );

    expect(result.code).not.toBe('PRIORITY_RESEARCH');
  });

  test('does not return PRIORITY_RESEARCH from valueScore alone', () => {
    const result = classifyResearchCandidate(
      createClassificationInput({
        valuation: {
          pbr: null,
          valueScore:
            DEFAULT_RESEARCH_CLASSIFICATION_THRESHOLDS.value
              .minimumValueScoreForPriorityResearch,
        },
        technical: {
          technicalScore: 0,
        },
      }),
    );

    expect(result.code).not.toBe('PRIORITY_RESEARCH');
  });

  test('does not return PRIORITY_RESEARCH from technicalScore alone', () => {
    const result = classifyResearchCandidate(
      createClassificationInput({
        valuation: {
          pbr: null,
          valueScore:
            DEFAULT_RESEARCH_CLASSIFICATION_THRESHOLDS.value
              .minimumValueScoreForPriorityResearch - 1,
        },
        technical: {
          technicalScore:
            DEFAULT_RESEARCH_CLASSIFICATION_THRESHOLDS.technical
              .minimumTechnicalScoreForPriorityResearch,
        },
      }),
    );

    expect(result.code).not.toBe('PRIORITY_RESEARCH');
  });

  test('does not return PRIORITY_RESEARCH when safetyScore is insufficient', () => {
    const result = classifyResearchCandidate(
      createPriorityResearchInput({
        safety: {
          safetyScore:
            DEFAULT_RESEARCH_CLASSIFICATION_THRESHOLDS.safety
              .minimumSafetyScoreForPriorityResearch - 1,
        },
      }),
    );

    expect(result.code).not.toBe('PRIORITY_RESEARCH');
  });

  test('does not return PRIORITY_RESEARCH when riskScore is insufficient', () => {
    const result = classifyResearchCandidate(
      createPriorityResearchInput({
        risk: {
          riskScore:
            DEFAULT_RESEARCH_CLASSIFICATION_THRESHOLDS.risk
              .minimumRiskScoreForPriorityResearch - 1,
        },
      }),
    );

    expect(result.code).not.toBe('PRIORITY_RESEARCH');
  });

  test('does not return PRIORITY_RESEARCH when caution flags exceed the limit', () => {
    const result = classifyResearchCandidate(
      createPriorityResearchInput({
        risk: {
          cautionFlags: ['CHECK_CAUTION'],
        },
      }),
    );

    expect(result.code).not.toBe('PRIORITY_RESEARCH');
  });

  test('does not return PRIORITY_RESEARCH when missing fields exceed the limit', () => {
    const result = classifyResearchCandidate(
      createPriorityResearchInput({
        dataConfidence: {
          missingFields: ['pbr-source'],
        },
      }),
    );

    expect(result.code).not.toBe('PRIORITY_RESEARCH');
  });

  test('does not return PRIORITY_RESEARCH when unusable indicators exceed the limit', () => {
    const result = classifyResearchCandidate(
      createPriorityResearchInput({
        dataConfidence: {
          unusableIndicatorReasons: ['bb-unavailable'],
        },
      }),
    );

    expect(result.code).not.toBe('PRIORITY_RESEARCH');
  });

  test('does not return PRIORITY_RESEARCH when low-liquidity flags exceed the limit', () => {
    const result = classifyResearchCandidate(
      createPriorityResearchInput({
        risk: {
          lowLiquidityFlags: ['LOW_VOLUME'],
        },
      }),
    );

    expect(result.code).not.toBe('PRIORITY_RESEARCH');
  });

  test('does not return PRIORITY_RESEARCH when sharp-decline context exceeds the limit', () => {
    const result = classifyResearchCandidate(
      createPriorityResearchInput({
        risk: {
          sharpDeclineContext: ['SHARP_DECLINE'],
        },
      }),
    );

    expect(result.code).not.toBe('PRIORITY_RESEARCH');
  });

  test('returns LOW_PRIORITY_OBSERVATION when value and safety materials are weak', () => {
    const result = classifyResearchCandidate(
      createClassificationInput({
        valuation: {
          valueScore:
            DEFAULT_RESEARCH_CLASSIFICATION_THRESHOLDS.value
              .minimumValueScoreForPriorityResearch - 1,
        },
        safety: {
          safetyScore:
            DEFAULT_RESEARCH_CLASSIFICATION_THRESHOLDS.safety
              .minimumSafetyScoreForPriorityResearch - 1,
        },
      }),
    );

    expect(result.code).toBe('LOW_PRIORITY_OBSERVATION');
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

  test('prioritizes DANGER_OBSERVATION over PRIORITY_RESEARCH', () => {
    const result = classifyResearchCandidate(
      createPriorityResearchInput({
        risk: {
          riskScore:
            DEFAULT_RESEARCH_CLASSIFICATION_THRESHOLDS.risk
              .dangerObservationMaxRiskScore,
        },
      }),
    );

    expect(result.code).toBe('DANGER_OBSERVATION');
  });
});

describe('get_pbr_bollinger_screener classification connection', () => {
  test('builds classification input from one screener candidate', () => {
    const input = buildResearchClassificationInput(
      createScreenerResult({
        latestVolume: null,
        pbr: null,
        bbState: 'BB_UNKNOWN',
        barCount: 0,
      }),
    );

    expect(input.identity.code).toBe('7203');
    expect(input.valuation.valueScore).toBe(8);
    expect(input.safety.safetyScore).toBe(5);
    expect(input.technical.technicalScore).toBe(10);
    expect(input.risk.riskScore).toBe(5);
    expect(input.dataConfidence.missingFields).toContain('pbr');
    expect(input.dataConfidence.unusableIndicatorReasons).toContain('NO_PRICE_BARS');
    expect(input.risk.lowLiquidityFlags).toContain('VOLUME_NA');
    expect(input.outputPolicy.shouldExcludeByDefault).toBe(false);

    const classificationResult = classifyResearchCandidate(input);

    expect(classificationResult.code).toBeTruthy();
  });

  test('keeps the existing TSV header shape after internal classification connection', () => {
    expect(NORMAL_CANDIDATE_TSV_HEADER).toHaveLength(35);
    expect(NORMAL_CANDIDATE_TSV_HEADER).not.toContain('classificationCode');
    expect(NORMAL_CANDIDATE_TSV_HEADER).not.toContain('classificationLabel');
    expect(NORMAL_CANDIDATE_TSV_HEADER).not.toContain('classificationReason');
    expect(NORMAL_CANDIDATE_TSV_HEADER).not.toContain('cautionBucket');
    expect(NORMAL_CANDIDATE_TSV_HEADER).not.toContain('outputBucket');
  });

  test('keeps the existing TSV row column count after internal classification connection', () => {
    const row = buildNormalCandidateTsvRow(createScreenerResult());

    expect(row).toHaveLength(NORMAL_CANDIDATE_TSV_HEADER.length);
  });

  test('keeps classification reasons free of buy-sell recommendation wording', () => {
    const forbiddenTerms = ['買い', '売り', '推奨', '投資ランク', '狙い目'];
    const results = [
      classifyResearchCandidate(createClassificationInput()),
      classifyResearchCandidate(createPriorityResearchInput()),
      classifyResearchCandidate(
        createClassificationInput({
          outputPolicy: {
            shouldExcludeByDefault: true,
          },
        }),
      ),
    ];

    for (const result of results) {
      expect(forbiddenTerms.some((term) => result.reason.includes(term))).toBe(
        false,
      );
    }
  });
});

// PRIORITY_RESEARCH test notes for the future implementation:
// - Consider it only when low PBR, valueScore, safetyScore, technicalScore,
//   riskScore, data quality, liquidity, decline context, and BB observation
//   materials align as multiple research signals.
// - Do not classify as PRIORITY_RESEARCH from low PBR alone.
// - Do not classify as PRIORITY_RESEARCH from valueScore alone.
// - Do not classify as PRIORITY_RESEARCH from technicalScore alone.
// - Do not classify as PRIORITY_RESEARCH from BB_LOW_NEAR alone.
// - Do not classify as PRIORITY_RESEARCH from BB_REBOUND alone.
// - Do not classify as PRIORITY_RESEARCH when safetyScore or riskScore is
//   below the required threshold.
// - Do not classify as PRIORITY_RESEARCH when caution, missing-data,
//   liquidity, or decline constraints exceed the priority-research limits.
// - Preserve decision order: negative classifications and
//   LOW_PRIORITY_OBSERVATION must not be bypassed by positive research labels.

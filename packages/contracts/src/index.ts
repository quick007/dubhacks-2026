import { z } from "zod";

export const CONTRACT_VERSION = 1 as const;
const id = z.string().min(1).max(100);
export const PersonaSchema = z
  .object({
    id,
    name: z.string(),
    goal: z.string(),
    split: z.enum(["development", "held-out"]),
    traits: z.array(z.string()),
    history: z.array(z.string()).default([]),
  })
  .strict();
export type Persona = z.infer<typeof PersonaSchema>;
export const VariantConfigSchema = z
  .object({
    headline: z.string().min(1).max(120),
    showShipping: z.boolean(),
    recommendationOrder: z.enum(["popular", "budget", "history"]),
  })
  .strict();
export type VariantConfig = z.infer<typeof VariantConfigSchema>;
export const VariantSchema = z
  .object({
    id,
    experimentId: id,
    name: z.string(),
    isBaseline: z.boolean(),
    config: VariantConfigSchema,
  })
  .strict();
export type Variant = z.infer<typeof VariantSchema>;
export const ExperimentSchema = z
  .object({
    id,
    name: z.string(),
    baselineVariantId: id,
    variantIds: z.array(id).min(2),
    developmentPersonaIds: z.array(id).min(1),
    heldOutPersonaIds: z.array(id).min(1),
    seed: z.number().int().nonnegative(),
    createdAt: z.string().datetime(),
    contractVersion: z.literal(CONTRACT_VERSION),
  })
  .strict()
  .refine((e) => e.variantIds.includes(e.baselineVariantId), "Baseline must be one of the variants")
  .refine(
    (e) => !e.developmentPersonaIds.some((p) => e.heldOutPersonaIds.includes(p)),
    "Persona splits must be disjoint",
  );
export type Experiment = z.infer<typeof ExperimentSchema>;
export const StepEventSchema = z
  .object({
    id,
    runId: id,
    sequence: z.number().int().nonnegative(),
    at: z.string().datetime(),
    kind: z.enum(["navigation", "action", "observation", "failure", "status"]),
    message: z.string(),
  })
  .strict();
export type StepEvent = z.infer<typeof StepEventSchema>;
export const ArtifactSchema = z
  .object({
    id,
    runId: id,
    kind: z.enum(["trace", "screenshot", "result"]),
    path: z.string(),
    mimeType: z.string(),
    createdAt: z.string().datetime(),
  })
  .strict();
export type Artifact = z.infer<typeof ArtifactSchema>;
export const EvaluationResultSchema = z
  .object({
    runId: id,
    synthetic: z.literal(true),
    taskCompleted: z.boolean(),
    score: z.number().min(0).max(1),
    failureReasons: z.array(z.string()),
    notes: z.array(z.string()),
  })
  .strict();
export type EvaluationResult = z.infer<typeof EvaluationResultSchema>;
export const ProposedChangeSchema = z
  .object({
    id,
    sourceRunId: id,
    kind: z.enum(["page", "recommendation"]),
    rationale: z.string(),
    evidenceStepIds: z.array(id),
    configPatch: VariantConfigSchema.partial(),
    status: z.literal("suggested"),
  })
  .strict();
export type ProposedChange = z.infer<typeof ProposedChangeSchema>;
export const RunRequestSchema = z
  .object({
    experimentId: id,
    variantId: id,
    personaId: id,
    adapter: z.enum(["mock", "playwright"]).default("mock"),
  })
  .strict();
export type RunRequest = z.infer<typeof RunRequestSchema>;
export const RunSchema = z
  .object({
    id,
    experimentId: id,
    baselineVariantId: id,
    variantId: id,
    personaId: id,
    personaSplit: z.enum(["development", "held-out"]),
    seed: z.number().int().nonnegative(),
    config: VariantConfigSchema,
    persona: PersonaSchema,
    adapter: z.enum(["mock", "playwright"]),
    synthetic: z.literal(true),
    contractVersion: z.literal(CONTRACT_VERSION),
    status: z.enum(["queued", "running", "completed", "failed"]),
    createdAt: z.string().datetime(),
    startedAt: z.string().datetime().optional(),
    completedAt: z.string().datetime().optional(),
    events: z.array(StepEventSchema),
    artifacts: z.array(ArtifactSchema),
    evaluation: EvaluationResultSchema.optional(),
    proposedChanges: z.array(ProposedChangeSchema),
    error: z.string().optional(),
  })
  .strict();
export type Run = z.infer<typeof RunSchema>;
export const CatalogSchema = z.object({
  experiment: ExperimentSchema,
  personas: z.array(PersonaSchema),
  variants: z.array(VariantSchema),
});
export type Catalog = z.infer<typeof CatalogSchema>;
export const ComparisonRowSchema = z.object({
  variantId: id,
  personaSplit: z.enum(["development", "held-out"]),
  runs: z.number().int(),
  completedTasks: z.number().int(),
  meanScore: z.number(),
  synthetic: z.literal(true),
});
export type ComparisonRow = z.infer<typeof ComparisonRowSchema>;

/** Frozen v1 HTTP interface. API is bound only to 127.0.0.1:4311.
 * GET /api/catalog -> Catalog
 * GET /api/runs -> Run[] (newest first)
 * POST /api/runs RunRequest -> 202 Run; queue full -> 429
 * GET /api/runs/:id -> Run; unknown -> 404
 * GET /api/comparison -> ComparisonRow[] (completed mock runs, split separately; browser smoke excluded)
 * GET /api/artifacts/:artifactId -> allowlisted local artifact download
 * GET /health -> { ok: true }
 * Errors -> { error: string }. No write endpoint accepts generated code or arbitrary URLs.
 */

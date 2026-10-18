export { LOCAL_ORIGINS, LOCAL_PROXY_PORT } from "./local.ts";
import { z } from "zod";

export const CONTRACT_VERSION = 1 as const;
const id = z.string().min(1).max(100);
export const RequirementSchema = z
  .object({
    id,
    label: z.string(),
    kind: z.enum(["hard", "preference"]),
    attribute: z.string(),
    value: z.string(),
  })
  .strict();
export const ShoppingTaskSchema = z
  .object({
    query: z.string(),
    requirements: z.array(RequirementSchema),
    budget: z.number().positive().optional(),
  })
  .strict();
export type ShoppingTask = z.infer<typeof ShoppingTaskSchema>;
export const PersonaSchema = z
  .object({
    id,
    name: z.string(),
    goal: z.string(),
    split: z.enum(["development", "held-out"]),
    traits: z.array(z.string()),
    history: z.array(z.string()).default([]),
    bio: z.string().optional(),
    task: ShoppingTaskSchema.optional(),
  })
  .strict();
export type Persona = z.infer<typeof PersonaSchema>;
export const VariantConfigSchema = z
  .object({
    headline: z.string().min(1).max(120),
    showShipping: z.boolean(),
    recommendationOrder: z.enum(["popular", "budget", "history"]),
    showAttributes: z.boolean().optional(),
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
export const PageKindSchema = z.enum([
  "home",
  "category",
  "search",
  "product",
  "cart",
  "checkout",
  "confirmation",
  "other",
]);
export type PageKind = z.infer<typeof PageKindSchema>;
export const StepEventSchema = z
  .object({
    id,
    runId: id,
    sequence: z.number().int().nonnegative(),
    at: z.string().datetime(),
    kind: z.enum(["navigation", "action", "observation", "failure", "status"]),
    message: z.string(),
    page: z.object({ kind: PageKindSchema, label: z.string().optional() }).strict().optional(),
    click: z
      .object({ x: z.number().min(0).max(1), y: z.number().min(0).max(1) })
      .strict()
      .optional(),
    screenshotArtifactId: id.optional(),
    search: z
      .object({ query: z.string(), resultCount: z.number().int().nonnegative() })
      .strict()
      .optional(),
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
const count = z.number().int().nonnegative();
export const FindabilitySchema = z
  .object({
    outcome: z.enum([
      "purchased",
      "abandoned",
      "constraint-failure",
      "technical-failure",
      "timeout",
    ]),
    requirements: z.array(
      z.object({ id, met: z.boolean(), observed: z.string().optional() }).strict(),
    ),
    steps: count,
    optimalSteps: count,
    searches: count,
    emptySearches: count,
    revisits: count,
    abandonedAt: id.optional(),
    orderValue: z.number().nonnegative().optional(),
  })
  .strict();
export type Findability = z.infer<typeof FindabilitySchema>;
export const EvaluationResultSchema = z
  .object({
    runId: id,
    synthetic: z.literal(true),
    taskCompleted: z.boolean(),
    score: z.number().min(0).max(1),
    failureReasons: z.array(z.string()),
    notes: z.array(z.string()),
    findability: FindabilitySchema.optional(),
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

/** Frozen v1 HTTP interface. API binds only to loopback on PORT; root dev exposes the named API origin.
 * GET /api/catalog -> Catalog
 * GET /api/runs -> Run[] (newest first)
 * POST /api/runs RunRequest -> 202 Run; queue full -> 429
 * GET /api/runs/:id -> Run; unknown -> 404
 * GET /api/comparison -> ComparisonRow[] (current experiment's completed mock runs, split separately; browser smoke excluded)
 * GET /api/artifacts/:artifactId -> allowlisted local artifact download
 * GET /health -> { ok: true }
 * Errors -> { error: string }. No write endpoint accepts generated code or arbitrary URLs.
 */

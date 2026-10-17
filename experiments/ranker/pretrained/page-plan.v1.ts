/** Additive model experiment contract; existing fusion callers remain unchanged. */
export type GeneratedFieldV1 = {
  text: string | null;
  source: "pending" | "generated" | "rejected" | "none" | "approved-fallback";
};
export interface ProductCardPlanV1 {
  productId: string;
  imageCaptionId: string;
  approvedFactIds: string[];
  observedHistoryReasonIds: string[];
  badgeDecision: "NONE" | "GENERATE";
  title: GeneratedFieldV1;
  badge: GeneratedFieldV1;
}
export interface PagePlanV1 {
  schemaVersion: "page-plan.v1";
  synthetic: true;
  recordId: string;
  requestedCardCount: 1 | 3 | 6;
  duplicatePolicy: "forbid" | "explicit-allow";
  sectionOrder: string[];
  sections: {
    sectionId: string;
    type: "recommendations" | "alternatives";
    productOrder: string[];
    products: ProductCardPlanV1[];
    heading: GeneratedFieldV1;
    copy: GeneratedFieldV1;
  }[];
  shortfall: number;
  method: Record<string, string>;
}

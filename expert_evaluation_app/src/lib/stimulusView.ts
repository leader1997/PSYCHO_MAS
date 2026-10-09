import type { EvidenceItem, ProfileFeature, XaiItem } from "../types";

export function visibleXai(items: XaiItem[]): XaiItem[] {
  return items;
}

export function visibleEvidence(items: EvidenceItem[]): EvidenceItem[] {
  return items;
}

export function visibleProfile(items: ProfileFeature[]): ProfileFeature[] {
  return items;
}

export function scrubReportSummary(text: string): string {
  return text;
}

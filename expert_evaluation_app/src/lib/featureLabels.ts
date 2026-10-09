import type { ProfileFeature } from "../types";

/** Display row after gender merge + value labeling. */
export interface LabeledFeature {
  key: string;
  display_name: string;
  feature_ids: string[];
  value: number | null;
  label: string;
  missing: boolean;
}

const YES_NO: Record<number, string> = {
  0: "No",
  1: "Yes",
};

/** HMS-style current financial situation (1 = least strain … 5 = most strain). */
const FINCUR: Record<number, string> = {
  1: "Always financially secure",
  2: "Mostly financially secure",
  3: "Occasionally financially insecure",
  4: "Often financially insecure",
  5: "Always financially insecure",
};

/** Past financial stress (same direction: higher = more strain). */
const FINPAST: Record<number, string> = {
  1: "Never stressful",
  2: "Rarely stressful",
  3: "Sometimes stressful",
  4: "Often stressful",
  5: "Always stressful",
};

const FOOD_WORRY: Record<number, string> = {
  1: "Never worried about food",
  2: "Sometimes worried about food",
  3: "Often worried about food",
};

/** Campus belonging agreement (6-point). */
const BELONG: Record<number, string> = {
  1: "Strongly agree",
  2: "Agree",
  3: "Somewhat agree",
  4: "Somewhat disagree",
  5: "Disagree",
  6: "Strongly disagree",
};

const EXERC: Record<number, string> = {
  1: "None / rarely",
  2: "1–2 days/week",
  3: "3–4 days/week",
  4: "5–6 days/week",
  5: "Daily",
};

const ACA_IMPA: Record<number, string> = {
  1: "No academic impairment",
  2: "Mild academic impairment",
  3: "Moderate academic impairment",
  4: "Severe academic impairment",
};

/** Common HMS institution-type codes (approximate). */
const INST_TYPE: Record<number, string> = {
  1: "Doctoral university",
  2: "Master's college/university",
  3: "Baccalaureate college",
  4: "Associate's / community college",
  5: "Special-focus / other",
};

const INST_SIZE: Record<number, string> = {
  1: "Under 1,000 students",
  2: "1,000–4,999 students",
  3: "5,000–9,999 students",
  4: "10,000–19,999 students",
  5: "20,000+ students",
};

function asInt(v: number | null | undefined): number | null {
  if (v == null || Number.isNaN(v)) return null;
  return Math.round(v);
}

function mapLabel(table: Record<number, string>, v: number | null): string | null {
  if (v == null) return null;
  const k = asInt(v);
  if (k == null) return null;
  return table[k] ?? null;
}

function formatNumber(v: number): string {
  return v === Math.trunc(v) ? String(Math.trunc(v)) : v.toFixed(2);
}

/** Label a single stored feature value for expert display. */
export function labelFeatureValue(feature: string, value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) return "Missing";

  switch (feature) {
    case "age":
      return `${formatNumber(value)} years`;
    case "gender_male":
      return value === 1 ? "Male" : value === 0 ? "Not male" : formatNumber(value);
    case "gender_female":
      return value === 1 ? "Female" : value === 0 ? "Not female" : formatNumber(value);
    case "international":
      return value === 1 ? "International student" : value === 0 ? "Domestic student" : formatNumber(value);
    case "fincur":
      return mapLabel(FINCUR, value) ?? `Code ${formatNumber(value)}`;
    case "finpast":
      return mapLabel(FINPAST, value) ?? `Code ${formatNumber(value)}`;
    case "food_worry":
      return mapLabel(FOOD_WORRY, value) ?? `Code ${formatNumber(value)}`;
    case "sleep_wknight":
    case "sleep_wkend":
      return `${formatNumber(value)} hours`;
    case "exerc":
      return mapLabel(EXERC, value) ?? `Code ${formatNumber(value)}`;
    case "belong1":
    case "belong2":
      return mapLabel(BELONG, value) ?? `Code ${formatNumber(value)}`;
    case "lonely":
      return value === 1 ? "Lonely (yes)" : value === 0 ? "Lonely (no)" : formatNumber(value);
    case "aca_impa":
      return mapLabel(ACA_IMPA, value) ?? `Code ${formatNumber(value)}`;
    case "flourish":
      return `Score ${formatNumber(value)}`;
    case "positiveMH":
      return value === 1 ? "Positive mental health (yes)" : value === 0 ? "Positive mental health (no)" : formatNumber(value);
    case "sub_any":
      return mapLabel(YES_NO, value) ?? formatNumber(value);
    case "binge1pl":
      return mapLabel(YES_NO, value) ?? formatNumber(value);
    case "smok_cig":
      return mapLabel(YES_NO, value) ?? formatNumber(value);
    case "discrim_race":
    case "discrim_gender":
    case "discrim_sexual":
      return value === 1 ? "Experienced" : value === 0 ? "Not reported" : formatNumber(value);
    case "inst_type":
      return mapLabel(INST_TYPE, value) ?? `Type code ${formatNumber(value)}`;
    case "inst_size":
      return mapLabel(INST_SIZE, value) ?? `Size code ${formatNumber(value)}`;
    case "inst_public":
      return value === 1 ? "Public institution" : value === 0 ? "Private institution" : formatNumber(value);
    default:
      return formatNumber(value);
  }
}

function findValue(features: ProfileFeature[], id: string): number | null {
  const row = features.find((f) => f.feature === id);
  if (!row) return null;
  if (row.value != null) return row.value;
  return null;
}

/**
 * Build display rows: merge gender into one field, attach significant labels.
 */
export function buildLabeledProfile(features: ProfileFeature[]): LabeledFeature[] {
  const byId = new Map(features.map((f) => [f.feature, f]));
  const out: LabeledFeature[] = [];
  const seen = new Set<string>();

  // Gender: prefer explicit male/female indicators over raw dummies.
  const male = findValue(features, "gender_male");
  const female = findValue(features, "gender_female");
  if (byId.has("gender_male") || byId.has("gender_female")) {
    let label = "Not specified";
    let value: number | null = null;
    if (male === 1 && female !== 1) {
      label = "Male";
      value = 1;
    } else if (female === 1 && male !== 1) {
      label = "Female";
      value = 1;
    } else if (male === 1 && female === 1) {
      label = "Both indicators present";
      value = 1;
    } else if (male == null && female == null) {
      label = "Missing";
    } else {
      label = "Not male/female (other or unspecified)";
    }
    out.push({
      key: "gender",
      display_name: "Gender",
      feature_ids: ["gender_male", "gender_female"],
      value,
      label,
      missing: label === "Missing",
    });
    seen.add("gender_male");
    seen.add("gender_female");
  }

  const order = [
    "age",
    "international",
    "aca_impa",
    "fincur",
    "finpast",
    "food_worry",
    "sleep_wknight",
    "sleep_wkend",
    "exerc",
    "belong1",
    "belong2",
    "lonely",
    "flourish",
    "positiveMH",
    "sub_any",
    "binge1pl",
    "smok_cig",
    "discrim_race",
    "discrim_gender",
    "discrim_sexual",
    "inst_type",
    "inst_size",
    "inst_public",
  ];

  for (const id of order) {
    if (seen.has(id)) continue;
    const row = byId.get(id);
    if (!row) continue;
    seen.add(id);
    const v = row.value ?? null;
    const label = labelFeatureValue(id, v);
    out.push({
      key: id,
      display_name: row.display_name,
      feature_ids: [id],
      value: v,
      label,
      missing: label === "Missing",
    });
  }

  // Any remaining features
  for (const row of features) {
    if (seen.has(row.feature)) continue;
    const v = row.value ?? null;
    const label = labelFeatureValue(row.feature, v);
    out.push({
      key: row.feature,
      display_name: row.display_name,
      feature_ids: [row.feature],
      value: v,
      label,
      missing: label === "Missing",
    });
  }

  return out;
}

export const FEATURES_PAGE_SIZE = 5;

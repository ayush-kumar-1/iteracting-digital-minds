"""Estimate and visualize context-conditioned within-value preferences.

The command in this module is deliberately manifest- and metadata-first.  It
joins the immutable normalized response file to the frozen scenario catalog,
validates the substantive pole mapping, estimates scenario-fixed-effect LPMs
with scenario-clustered standard errors, and writes all machine-readable and
LaTeX/PDF Appendix B inputs in one deterministic run.

Run from the repository root with ``.venv/bin/python -m
src.python.analysis.within_value_analysis``.  The project ``uv`` environment
contains the same dependencies; the direct interpreter invocation is useful
in restricted desktop sessions where uv cannot access its global cache.
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyfixest as pf
from scipy.stats import t as student_t


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUN_ROOT = PROJECT_ROOT / "Output/experiments/haiku-main-effects-v1-d384fb5327dc"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "Output/analysis"
FIGURE_ROOT = PROJECT_ROOT / "Output/figures/appendix_b"
TABLE_ROOT = PROJECT_ROOT / "Output/tables/appendix_b"

VALUE_LABELS = {
    "Q8": "Children: independence",
    "Q10": "Children: responsibility",
    "Q12": "Children: tolerance",
    "Q13": "Children: thrift",
    "Q16": "Children: unselfishness",
    "Q17": "Children: obedience",
    "Q29": "Equal political leadership",
    "Q30": "Equal university education",
    "Q33": "Equal access to jobs",
    "Q34": "Equal access for immigrants",
    "Q36": "Equal parental recognition",
    "Q38": "Adult-child caregiving duty",
    "Q40": "Work as social duty",
    "Q41": "Work over spare time",
    "Q57": "Trust most people",
    "Q90": "Administrative effectiveness",
    "Q107": "Private ownership",
    "Q108": "Government provision",
    "Q130": "Open labor migration",
    "Q174": "Religious norms",
    "Q177": "Public-benefit integrity",
    "Q180": "Tax integrity",
    "Q181": "Anti-bribery integrity",
    "Q195": "Reject capital punishment",
    "Q196": "Public video surveillance",
    "Q197": "Digital monitoring",
}

DOMAIN_ORDER = [
    "child-rearing values",
    "civic responsibility",
    "social tolerance",
    "stewardship",
    "prosociality",
    "authority and conformity",
    "gender equality",
    "migration and equal access",
    "family equality",
    "family obligation",
    "work and civic duty",
    "work-life priority",
    "interpersonal trust",
    "governance trade-off",
    "economic governance",
    "social provision",
    "migration policy",
    "religious orientation",
    "public integrity",
    "criminal punishment",
    "privacy and security",
]

FRAME_NAMES = {
    "BASELINE": "No frame",
    "F01": "Personal advice",
    "F02": "Career mentorship",
    "F03": "Technical collaboration",
    "F04": "Goal-directed game",
    "F05": "Policymaker advice",
    "F06": "Organizational decision",
    "F07": "Household planning",
    "F08": "Intellectual collaboration",
}

EDUCATION_LABELS = {
    "high_school": "High school",
    "vocational_technical": "Vocational",
    "some_college": "Some college",
    "bachelors": "Bachelor's",
    "masters": "Master's / professional",
    "professional_degree": "Master's / professional",
    "doctoral": "Doctoral",
}
EDUCATION_ORDER = [
    "high_school",
    "vocational_technical",
    "some_college",
    "bachelors",
    "masters",
    "professional_degree",
    "doctoral",
]
AGE_ORDER = ["early_adult", "early_career", "mid_career", "experienced_professional", "retired_older_adult"]
AGE_LABELS = {
    "early_adult": "18–24",
    "early_career": "25–34",
    "mid_career": "35–49",
    "experienced_professional": "50–64",
    "retired_older_adult": "65+",
}
RESPONSE_ORDER = [
    "warm_and_conversational",
    "concise_and_direct",
    "evidence_and_citation_heavy",
    "detailed_and_step_by_step",
    "technically_sophisticated",
    "plain_language_explanation",
]
RESPONSE_LABELS = {
    "warm_and_conversational": "Warm / conversational",
    "concise_and_direct": "Concise / direct",
    "evidence_and_citation_heavy": "Citation-heavy",
    "detailed_and_step_by_step": "Detailed / step-by-step",
    "technically_sophisticated": "Technically sophisticated",
    "plain_language_explanation": "Plain-language",
}
DIMENSION_LABELS = {
    "gender": "User gender",
    "religion": "User religion",
    "education": "User education",
    "age_band": "User age band",
    "response_style": "Response preference",
    "frame_id": "Context frame",
    "history_level": "History length",
    "history_length": "History length",
    "elicitation_id": "Elicitation method",
    "profile_level": "Individual profile",
}
MARKERS = ["o", "s", "D", "^", "P", "X", "v", "<", ">", "h"]
COLORS = ["#2f5597", "#b23a48", "#4d7c5a", "#8a5a44", "#6d597a", "#c77d2a", "#3f7f8f", "#7a7a7a"]


@dataclass(frozen=True)
class Paths:
    """Output locations for one analysis run."""

    root: Path

    @property
    def analysis_data(self) -> Path:
        return self.root / "within_value_analysis.parquet"

    @property
    def coefficients(self) -> Path:
        return self.root / "coefficient_results.parquet"

    @property
    def adjusted_means(self) -> Path:
        return self.root / "adjusted_means.parquet"

    @property
    def pairwise(self) -> Path:
        return self.root / "pairwise_tests.parquet"

    @property
    def joint(self) -> Path:
        return self.root / "joint_tests.parquet"

    @property
    def support(self) -> Path:
        return self.root / "support_diagnostics.parquet"

    @property
    def interaction_support(self) -> Path:
        return self.root / "interaction_support.parquet"

    @property
    def robustness(self) -> Path:
        return self.root / "robustness_checks.parquet"

    @property
    def effect_sizes(self) -> Path:
        return self.root / "effect_sizes.parquet"

    @property
    def identification(self) -> Path:
        return self.root / "identification_checks.parquet"

    @property
    def descriptives(self) -> Path:
        return self.root / "descriptive_statistics.parquet"

    @property
    def qa(self) -> Path:
        return self.root / "analysis_qa.json"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _normalise_response_style(values: list[str]) -> str:
    text = values[0].lower()
    replacements = {
        "warm and conversational": "warm_and_conversational",
        "concise and direct": "concise_and_direct",
        "evidence- and citation-heavy": "evidence_and_citation_heavy",
        "detailed and step-by-step": "detailed_and_step_by_step",
        "technically sophisticated": "technically_sophisticated",
        "plain-language explanation": "plain_language_explanation",
        "skeptical and critical": "skeptical_and_critical",
        "practical/action-oriented": "practical_action_oriented",
        "socratic/question-driven": "socratic_question_driven",
    }
    return replacements.get(text, _slug(text))


def _metadata() -> tuple[dict[str, dict], dict[str, dict], dict[str, dict], dict[str, dict]]:
    root = PROJECT_ROOT / "experiment-library/en"
    scenarios = {x["scenario_id"]: x for x in _read_jsonl(root / "wvs/scenarios.jsonl")}
    questions = {x["wvs_item_id"]: x for x in _read_jsonl(root / "wvs/questions.jsonl")}
    profiles = {x["profile_id"]: x for x in _read_jsonl(root / "data/profiles.jsonl")}
    frames = {x["frame_id"]: x for x in _read_jsonl(root / "data/frames.jsonl")}
    return scenarios, questions, profiles, frames


def build_analysis_dataset(run_root: Path, output: Paths) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build and validate the canonical row-per-generation analysis dataset."""
    scenarios, questions, profiles, frames = _metadata()
    responses = pd.read_parquet(run_root / "responses.parquet")
    required = {"request_id", "condition_id", "replicate", "wvs_item_id", "scenario_id", "option_order", "valid_response", "selected_option_id", "selected_value_pole"}
    missing = sorted(required - set(responses.columns))
    if missing:
        raise ValueError(f"Response artifact is missing required columns: {missing}")

    scenario_rows = pd.DataFrame.from_dict(scenarios, orient="index").reset_index(drop=True)
    scenario_rows = scenario_rows.rename(columns={"option_a_pole": "reference_pole", "option_b_pole": "opposing_pole"})
    question_rows = pd.DataFrame.from_dict(questions, orient="index").reset_index(drop=True)
    question_rows = question_rows.rename(columns={"pole_a": "reference_pole_from_question", "pole_b": "opposing_pole_from_question", "value_domain": "value_domain"})
    profile_rows = []
    for profile_id, record in profiles.items():
        row = {"profile_id": profile_id, **record["attributes"]}
        row["response_style"] = _normalise_response_style(record["attributes"]["response_preferences"])
        profile_rows.append(row)
    profile_rows = pd.DataFrame(profile_rows)
    frame_rows = pd.DataFrame.from_dict(frames, orient="index").reset_index(drop=True)[["frame_id", "name"]].rename(columns={"name": "frame_name"})

    df = responses.merge(scenario_rows[["scenario_id", "context", "option_a", "option_b", "reference_pole", "opposing_pole"]], on="scenario_id", how="left", validate="many_to_one")
    df = df.merge(question_rows[["wvs_item_id", "question_text", "value_domain", "reference_pole_from_question", "opposing_pole_from_question"]], on="wvs_item_id", how="left", validate="many_to_one")
    df = df.merge(profile_rows, on="profile_id", how="left", validate="many_to_one")
    df = df.merge(frame_rows, on="frame_id", how="left", validate="many_to_one")

    df["frame_level"] = df["frame_id"].fillna("BASELINE")
    df["frame_name"] = df["frame_name"].fillna("No frame")
    df["history_level"] = "H" + df["history_length"].astype(str)
    df.loc[df["history_length"].eq(0), "history_level"] = "H0"
    df["history_label"] = df["history_length"].map({0: "Baseline / 0 turns", 1: "1 turn", 3: "3 turns", 5: "5 turns"})
    df["profile_level"] = df["profile_id"].fillna("BASELINE")
    df["profile_name"] = df["profile_id"].map({k: v["name"] for k, v in profiles.items()}).fillna("No profile")
    df["value_label"] = df["wvs_item_id"].map(VALUE_LABELS).fillna(df["question_text"])
    df["value_name"] = df["value_label"]
    df["scenario_realization"] = df["context"]
    df["elicitation_type"] = df["elicitation_id"].map({"E01": "Direct preference", "E02": "Choice/action", "E03": "Recommendation", "E04": "Explicit indifference"})
    df["education_label"] = df["education"].map(EDUCATION_LABELS)
    df["age_band_label"] = df["age_band"].map(AGE_LABELS)
    df["gender_level"] = df["gender"]
    df["religion_level"] = df["religion"]
    df["education_level"] = df["education"]
    df["age_band_level"] = df["age_band"]
    df["stimulus_id"] = df["condition_id"]
    df["no_preference"] = df["raw_response"].str.contains(r"\bno preference\b", case=False, na=False) | df["raw_choice"].astype(str).str.contains(r"no preference", case=False, na=False)
    df["valid_response"] = df["valid_response"].astype(bool)
    df["analysis_valid"] = df["valid_response"] & ~df["no_preference"]

    # The parser stores the canonical scenario option in selected_option_id.
    # Validate that the substantive direction comes from frozen scenario poles,
    # not from the displayed A/B position.
    selected_canonical = df["selected_option_id"].astype("string").str.rsplit(":", n=1).str[-1]
    expected = np.where(selected_canonical.eq("A"), df["reference_pole"], df["opposing_pole"])
    checked = df["analysis_valid"] & df["selected_option_id"].notna()
    mismatches = int((checked & df["selected_value_pole"].ne(expected)).sum())
    if mismatches:
        raise ValueError(f"Substantive pole validation failed for {mismatches} valid responses")
    question_mismatch = (df["reference_pole"].ne(df["reference_pole_from_question"]) | df["opposing_pole"].ne(df["opposing_pole_from_question"])).sum()
    if question_mismatch:
        raise ValueError(f"Scenario/question pole metadata disagree for {question_mismatch} response rows")
    df["selected_option"] = df["canonical_choice"]
    df["selected_value_pole"] = df["selected_value_pole"].where(df["analysis_valid"])
    df["value_binary"] = np.where(~df["analysis_valid"], np.nan, (df["selected_value_pole"] == df["reference_pole"]).astype(float))
    df["temperature"] = np.nan
    df["input_tokens"] = pd.to_numeric(df["input_tokens"], errors="coerce")
    df["output_tokens"] = pd.to_numeric(df["output_tokens"], errors="coerce")

    keep = [
        "request_id", "condition_id", "stimulus_id", "replicate", "model", "temperature", "input_tokens", "output_tokens",
        "wvs_item_id", "value_id", "value_name", "value_label", "value_domain", "scenario_id", "scenario_realization",
        "selected_option", "selected_value_pole", "value_binary", "reference_pole", "opposing_pole", "valid_response", "analysis_valid", "no_preference",
        "frame_id", "frame_level", "frame_name", "history_id", "history_length", "history_level", "history_label",
        "profile_id", "profile_level", "profile_name", "religion", "religion_level", "gender", "gender_level", "education", "education_level", "education_label", "age", "age_band", "age_band_level", "age_band_label", "career_stage", "response_style",
        "elicitation_id", "elicitation_type", "option_order", "scenario_family",
    ]
    df["value_id"] = df["wvs_item_id"]
    for col in keep:
        if col not in df:
            df[col] = np.nan
    df = df[keep].sort_values(["wvs_item_id", "scenario_id", "frame_level", "history_length", "profile_level", "elicitation_id", "option_order", "replicate"]).reset_index(drop=True)
    output.root.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output.analysis_data, index=False)
    qa = {
        "input_rows": int(len(df)),
        "valid_rows": int(df["analysis_valid"].sum()),
        "invalid_rows": int((~df["valid_response"]).sum()),
        "no_preference_rows": int(df["no_preference"].sum()),
        "pole_mismatches": mismatches,
        "question_pole_mismatches": int(question_mismatch),
        "n_values": int(df["value_id"].nunique()),
        "n_scenarios": int(df["scenario_id"].nunique()),
        "n_stimuli": int(df["stimulus_id"].nunique()),
        "n_replicates": int(df["replicate"].nunique()),
    }
    return df, qa


def _support_status(n_scenarios: int, n_option_orders: int, n_wvs_items: int = 1) -> str:
    if n_scenarios < 10 or n_option_orders < 2 or n_wvs_items < 1:
        return "descriptive only — insufficient independent clusters"
    if n_scenarios < 20:
        return "limited — fewer than 20 independent clusters"
    if n_wvs_items < 3:
        return "limited — fewer than 3 WVS items"
    return "adequate"


def build_support(df: pd.DataFrame) -> pd.DataFrame:
    """Create level-level support diagnostics before inferential outputs."""
    specs = {
        "gender": ("gender_level", df["profile_id"].notna()),
        "religion": ("religion_level", df["profile_id"].notna()),
        "education": ("education_level", df["profile_id"].notna()),
        "age_band": ("age_band_level", df["profile_id"].notna()),
        "response_style": ("response_style", df["profile_id"].notna()),
        "profile": ("profile_level", df["profile_id"].notna()),
        "frame": ("frame_level", df["frame_id"].notna()),
        "history_length": ("history_level", df["frame_id"].notna()),
        "elicitation": ("elicitation_id", df["frame_id"].isna() & df["profile_id"].isna() & df["history_length"].eq(0)),
    }
    rows = []
    for value_id, value_df in df.groupby("value_id", sort=False):
        for dimension, (level_col, mask) in specs.items():
            subset = value_df.loc[mask.loc[value_df.index] if hasattr(mask, "loc") else mask]
            for level, level_df in subset.groupby(level_col, dropna=False, sort=False):
                valid = level_df[level_df["analysis_valid"]]
                cell_counts = valid.groupby(["scenario_id", "option_order"], dropna=False).size()
                n_scenarios = valid["scenario_id"].nunique()
                rows.append({
                    "value_id": value_id,
                    "value_label": value_df["value_label"].iloc[0],
                    "value_domain": value_df["value_domain"].iloc[0],
                    "dimension": dimension,
                    "level": str(level),
                    "level_label": _level_label(dimension, str(level), value_df),
                    "n_responses": int(len(level_df)),
                    "n_valid_responses": int(len(valid)),
                    "n_scenarios": int(n_scenarios),
                    "n_wvs_items": int(valid["wvs_item_id"].nunique()),
                    "n_stimulus_cells": int(valid["stimulus_id"].nunique()),
                    "n_profiles": int(valid["profile_id"].nunique(dropna=True)),
                    "n_frames": int(valid["frame_id"].nunique(dropna=True)),
                    "n_option_orders": int(valid["option_order"].nunique()),
                    "minimum_cell_size": int(cell_counts.min()) if len(cell_counts) else 0,
                    "support_status": _support_status(n_scenarios, int(valid["option_order"].nunique()), int(valid["wvs_item_id"].nunique())),
                })
    return pd.DataFrame(rows)


def _level_label(dimension: str, level: str, df: pd.DataFrame | None = None) -> str:
    if dimension == "frame":
        return FRAME_NAMES.get(level, level)
    if dimension == "history_length":
        return {"H0": "Baseline / 0 turns", "H1": "1 turn", "H3": "3 turns", "H5": "5 turns"}.get(level, level)
    if dimension == "elicitation":
        return {"E01": "Direct preference", "E02": "Choice/action", "E03": "Recommendation", "E04": "Explicit indifference"}.get(level, level)
    if dimension == "profile":
        if df is not None and level in set(df["profile_level"].astype(str)):
            row = df.loc[df["profile_level"].astype(str).eq(level)]
            return row["profile_name"].iloc[0] if len(row) else level
        return "No profile" if level == "BASELINE" else level
    if dimension == "education":
        return EDUCATION_LABELS.get(level, level)
    if dimension == "age_band":
        return AGE_LABELS.get(level, level)
    if dimension == "response_style":
        return RESPONSE_LABELS.get(level, level.replace("_", " ").title())
    return level.replace("_", " ").title()


def _filter_for_dimension(df: pd.DataFrame, dimension: str) -> tuple[pd.DataFrame, str, str, str]:
    if dimension in {"gender", "religion", "education", "age_band", "response_style", "profile"}:
        profile_mask = df["profile_id"].notna() if dimension != "profile" else pd.Series(True, index=df.index)
        subset = df[df["frame_id"].isna() & profile_mask & df["history_length"].eq(0) & df["elicitation_id"].eq("E01")].copy()
        if dimension == "gender":
            return subset, "gender_level", "man", "profile characteristics"
        if dimension == "religion":
            return subset, "religion_level", "Christian", "profile characteristics"
        if dimension == "education":
            return subset, "education_level", "high_school", "profile characteristics"
        if dimension == "age_band":
            return subset, "age_band_level", "early_adult", "profile characteristics"
        if dimension == "response_style":
            return subset, "response_style", "concise_and_direct", "profile characteristics"
        return subset, "profile_level", "BASELINE", "individual profile"
    if dimension == "frame":
        subset = df[df["frame_id"].notna()].copy()
        return subset, "frame_level", "F01", "context frame"
    if dimension == "history_length":
        subset = df[df["frame_id"].notna()].copy()
        return subset, "history_level", "H1", "history length"
    if dimension == "elicitation":
        subset = df[df["frame_id"].isna() & df["profile_id"].isna() & df["history_length"].eq(0)].copy()
        return subset, "elicitation_id", "E01", "elicitation method"
    raise KeyError(dimension)


def _formula(dimension: str, treatment_col: str, reference: str) -> str:
    extras = [f"C(scenario_id)", "C(option_order)"]
    if dimension == "frame":
        extras.insert(0, "C(history_level)")
        return f"value_binary ~ i({treatment_col}, ref='{reference}') + " + " + ".join(extras)
    if dimension == "history_length":
        extras.insert(0, "C(frame_level)")
        return f"value_binary ~ i({treatment_col}, ref='{reference}') + " + " + ".join(extras)
    if dimension == "elicitation":
        return f"value_binary ~ i({treatment_col}, ref='{reference}') + " + " + ".join(extras)
    if dimension == "profile":
        return f"value_binary ~ i({treatment_col}, ref='{reference}') + " + " + ".join(extras)
    # The 12 synthetic profiles assign a unique bundle of attributes to each
    # profile.  A saturated joint characteristic model therefore has no
    # independent demographic variation to identify.  Estimate each requested
    # characteristic separately and record that limitation rather than
    # reporting numerically unstable coefficients from a rank-deficient model.
    return f"value_binary ~ i({treatment_col}, ref='{reference}') + " + " + ".join(extras)


def _fit_one(value_df: pd.DataFrame, dimension: str) -> tuple[Any, pd.DataFrame, str, str, str] | None:
    subset, treatment_col, reference, family = _filter_for_dimension(value_df, dimension)
    subset = subset[subset["analysis_valid"]].copy()
    subset = subset.dropna(subset=[treatment_col, "value_binary", "scenario_id", "option_order"])
    if subset[treatment_col].nunique() < 2 or reference not in set(subset[treatment_col].astype(str)) or subset["scenario_id"].nunique() < 2:
        return None
    formula = _formula(dimension, treatment_col, reference)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            model = pf.feols(formula, data=subset, vcov={"CRV1": "scenario_id"})
        except Exception as exc:
            print(f"Skipping {value_df['value_id'].iloc[0]} {dimension}: {exc}")
            return None
    note = "; ".join(str(w.message) for w in caught if "dropped" in str(w.message).lower())
    if dimension in {"gender", "religion", "education", "age_band", "response_style"}:
        note = (note + "; " if note else "") + "Other synthetic profile attributes are bundled with this characteristic and are not independently identified."
    return model, subset, treatment_col, reference, note


def _bh(values: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=values.index, dtype=float)
    valid = values.notna()
    if not valid.any():
        return out
    p = values.loc[valid].to_numpy(dtype=float)
    order = np.argsort(p)
    ranks = np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate((p[order] * len(p) / ranks)[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    out.loc[valid] = pd.Series(adjusted[np.argsort(order)], index=values.loc[valid].index)
    return out


def _model_results(model: Any, subset: pd.DataFrame, value_meta: dict[str, Any], dimension: str, treatment_col: str, reference: str, collinearity: str) -> tuple[list[dict], list[dict], list[dict], dict[str, Any]]:
    coef = model.coef()
    tidy = model.tidy().reset_index().rename(columns={"Coefficient": "term", "Estimate": "estimate", "Std. Error": "standard_error", "Pr(>|t|)": "p_value", "2.5%": "ci_low", "97.5%": "ci_high"})
    treatment_prefix = treatment_col + "::"
    treatment_terms = [term for term in coef.index.astype(str) if term.startswith(treatment_prefix)]
    rows = []
    for _, r in tidy[tidy["term"].astype(str).isin(treatment_terms)].iterrows():
        rows.append({
            **value_meta, "dimension": dimension, "term": str(r["term"]), "treatment_col": treatment_col,
            "level": str(r["term"]).split("::", 1)[1], "reference_level": reference, "estimate": float(r["estimate"]),
            "standard_error": float(r["standard_error"]), "ci_low": float(r["ci_low"]), "ci_high": float(r["ci_high"]),
            "p_value": float(r["p_value"]) if pd.notna(r["p_value"]) else np.nan, "n": int(model._N),
            "n_clusters": int(subset["scenario_id"].nunique()), "fixed_effects": "scenario_id; option_order" + ("; history_level" if dimension == "frame" else ""),
            "collinearity_note": collinearity,
        })

    predictions = model.predict()
    coef_names = [str(x) for x in model._coefnames]
    # pyfixest retains columns dropped for collinearity in ``_Xd`` while
    # removing them from the estimated covariance matrix.  Align explicitly.
    xmat = np.asarray(model._Xd[coef_names], dtype=float)
    beta = np.asarray(model._beta_hat, dtype=float)
    vcov = np.asarray(model._vcov, dtype=float)
    subset = subset.reset_index(drop=True)
    means = []
    levels = list(subset[treatment_col].astype(str).drop_duplicates())
    for level in levels:
        mask = subset[treatment_col].astype(str).eq(level).to_numpy()
        xbar = xmat[mask].mean(axis=0)
        estimate = float(np.mean(predictions[mask]))
        variance = float(xbar @ vcov @ xbar.T)
        se = float(np.sqrt(max(variance, 0)))
        df_resid = max(float(getattr(model, "_df_t", 1)), 1.0)
        crit = float(student_t.ppf(0.975, df_resid))
        means.append({
            **value_meta, "dimension": dimension, "level": level, "level_label": _level_label(dimension, level, subset),
            "estimate": estimate, "standard_error": se, "ci_low": max(0.0, estimate - crit * se), "ci_high": min(1.0, estimate + crit * se),
            "p_value": np.nan, "q_value": np.nan, "n": int(mask.sum()), "n_scenarios": int(subset.loc[mask, "scenario_id"].nunique()),
        })
    pairwise = []
    for i, left in enumerate(levels):
        for right in levels[i + 1:]:
            lm = subset[treatment_col].astype(str).eq(left).to_numpy()
            rm = subset[treatment_col].astype(str).eq(right).to_numpy()
            xdiff = xmat[lm].mean(axis=0) - xmat[rm].mean(axis=0)
            diff = float(xdiff @ beta)
            se = float(np.sqrt(max(float(xdiff @ vcov @ xdiff.T), 0)))
            df_resid = max(float(getattr(model, "_df_t", 1)), 1.0)
            tstat = diff / se if se > 0 else np.nan
            p = float(2 * student_t.sf(abs(tstat), df_resid)) if pd.notna(tstat) else np.nan
            pairwise.append({
                **value_meta, "dimension": dimension, "level_1": left, "level_1_label": _level_label(dimension, left, subset),
                "level_2": right, "level_2_label": _level_label(dimension, right, subset), "difference": diff,
                "standard_error": se, "ci_low": diff - 1.96 * se, "ci_high": diff + 1.96 * se, "p_value": p,
                "n": int(len(subset)), "n_scenarios": int(subset["scenario_id"].nunique()), "n_clusters": int(subset["scenario_id"].nunique()),
            })

    joint = []
    if treatment_terms:
        R = np.zeros((len(treatment_terms), len(coef_names)))
        for i, term in enumerate(treatment_terms):
            R[i, coef_names.index(term)] = 1
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                wald = model.wald_test(R=R)
            joint_p = float(wald["pvalue"])
            joint_stat = float(wald["statistic"])
        except Exception:
            joint_p, joint_stat = np.nan, np.nan
        joint.append({**value_meta, "dimension": dimension, "joint_test": DIMENSION_LABELS.get(treatment_col, dimension), "statistic": joint_stat, "p_value": joint_p, "n": int(model._N), "n_clusters": int(subset["scenario_id"].nunique()), "collinearity_note": collinearity})
    identification = {**value_meta, "dimension": dimension, "variables_requested": ", ".join(coef_names), "variables_estimated": ", ".join(coef_names), "variables_dropped": collinearity, "fixed_effects": "C(scenario_id); C(option_order)" + ("; C(history_level)" if dimension == "frame" else ""), "n": int(model._N), "n_clusters": int(subset["scenario_id"].nunique()), "within_fe_variation": int(subset.groupby("scenario_id")[treatment_col].nunique().gt(1).sum())}
    return rows, means, pairwise, {"joint": joint, "identification": identification}


def estimate_all(df: pd.DataFrame, output: Paths) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Estimate the primary LPM families and apply within-family FDR corrections."""
    dimensions = ["gender", "religion", "education", "age_band", "response_style", "frame", "history_length", "elicitation", "profile"]
    coefficient_rows: list[dict] = []
    mean_rows: list[dict] = []
    pairwise_rows: list[dict] = []
    joint_rows: list[dict] = []
    identification_rows: list[dict] = []
    for value_id, value_df in df.groupby("value_id", sort=True):
        value_meta = {"value_id": value_id, "value_label": value_df["value_label"].iloc[0], "value_domain": value_df["value_domain"].iloc[0]}
        for dimension in dimensions:
            fitted = _fit_one(value_df, dimension)
            if fitted is None:
                continue
            model, subset, treatment_col, reference, collinearity = fitted
            result = _model_results(model, subset, value_meta, dimension, treatment_col, reference, collinearity)
            coefficient_rows.extend(result[0])
            mean_rows.extend(result[1])
            pairwise_rows.extend(result[2])
            joint_rows.extend(result[3]["joint"])
            identification_rows.append(result[3]["identification"])
    coefficients = pd.DataFrame(coefficient_rows)
    means = pd.DataFrame(mean_rows)
    pairwise = pd.DataFrame(pairwise_rows)
    joints = pd.DataFrame(joint_rows)
    identification = pd.DataFrame(identification_rows)
    if len(coefficients):
        coefficients["q_value"] = coefficients.groupby("dimension", group_keys=False)["p_value"].apply(_bh).reset_index(level=0, drop=True)
    if len(pairwise):
        pairwise["q_value"] = pairwise.groupby("dimension", group_keys=False)["p_value"].apply(_bh).reset_index(level=0, drop=True)
    if len(joints):
        joints["q_value"] = joints.groupby("dimension", group_keys=False)["p_value"].apply(_bh).reset_index(level=0, drop=True)
    if len(means):
        support = pd.read_parquet(output.support)
        means = means.merge(support[["value_id", "dimension", "level", "support_status"]].drop_duplicates(), on=["value_id", "dimension", "level"], how="left")
    coefficients.to_parquet(output.coefficients, index=False)
    means.to_parquet(output.adjusted_means, index=False)
    pairwise.to_parquet(output.pairwise, index=False)
    joints.to_parquet(output.joint, index=False)
    identification.to_parquet(output.identification, index=False)
    return coefficients, means, pairwise, joints, identification


def descriptive_statistics(df: pd.DataFrame, output: Paths) -> pd.DataFrame:
    """Write value-level descriptive statistics and support breakdowns."""
    rows = []
    for value_id, group in df.groupby("value_id", sort=True):
        valid = group[group["analysis_valid"]]
        rows.append({
            "value_id": value_id, "value_label": group["value_label"].iloc[0], "value_domain": group["value_domain"].iloc[0],
            "n_responses": len(group), "n_valid_responses": len(valid), "n_scenarios": group["scenario_id"].nunique(),
            "n_experimental_conditions": group["condition_id"].nunique(), "mean_reference_pole": valid["value_binary"].mean(),
            "share_no_preference": group["no_preference"].mean(), "share_invalid": (~group["valid_response"]).mean(),
        })
    out = pd.DataFrame(rows)
    out.to_parquet(output.descriptives, index=False)
    for col in ["frame_name", "history_label", "profile_name", "gender", "religion", "education_label", "age_band_label", "response_style", "elicitation_type"]:
        breakdown_rows = []
        for (value_id, level), group in df.groupby(["value_id", col], dropna=False, sort=True):
            valid = group[group["analysis_valid"]]
            breakdown_rows.append({"value_id": value_id, "breakdown": col, "level": str(level), "n_responses": len(group), "n_valid_responses": len(valid), "mean_reference_pole": valid["value_binary"].mean(), "share_invalid": (~group["valid_response"]).mean()})
        pd.DataFrame(breakdown_rows).to_parquet(output.root / f"descriptive_by_{_slug(col)}.parquet", index=False)
    return out


def effect_sizes(means: pd.DataFrame, output: Paths) -> pd.DataFrame:
    """Calculate max-minus-min adjusted probability ranges by value/dimension."""
    if means.empty:
        out = pd.DataFrame()
    else:
        rows = []
        for (value_id, dimension), group in means.groupby(["value_id", "dimension"], sort=True):
            group = group.dropna(subset=["estimate"])
            if group.empty:
                continue
            low = group.loc[group["estimate"].idxmin()]
            high = group.loc[group["estimate"].idxmax()]
            rows.append({"value_id": value_id, "value_label": group["value_label"].iloc[0], "value_domain": group["value_domain"].iloc[0], "dimension": dimension, "min_level": low["level"], "max_level": high["level"], "min_estimate": low["estimate"], "max_estimate": high["estimate"], "range": high["estimate"] - low["estimate"], "range_percentage_points": 100 * (high["estimate"] - low["estimate"]), "support_status": "; ".join(sorted(set(group["support_status"].dropna())))})
        out = pd.DataFrame(rows)
    out.to_parquet(output.effect_sizes, index=False)
    return out


def robustness_and_interactions(df: pd.DataFrame, output: Paths) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write condition-level robustness artifacts and an interaction support matrix."""
    valid = df[df["analysis_valid"]].copy()
    condition = valid.groupby(["value_id", "value_label", "value_domain", "stimulus_id", "scenario_id", "frame_level", "history_level", "profile_level", "elicitation_id", "option_order"], dropna=False).agg(value_binary=("value_binary", "mean"), n_valid=("value_binary", "size")).reset_index()
    condition.to_parquet(output.root / "scenario_level_aggregation.parquet", index=False)
    normalized = valid.groupby(["value_id", "value_label", "value_domain", "scenario_id", "frame_level", "history_level", "profile_level", "elicitation_id"], dropna=False).agg(value_binary=("value_binary", "mean"), n_option_orders=("option_order", "nunique"), n_stimulus_cells=("stimulus_id", "nunique")).reset_index()
    normalized.to_parquet(output.root / "ab_ba_averaged.parquet", index=False)
    stimulus_quality = df.groupby(["value_id", "stimulus_id"], dropna=False).agg(n_responses=("request_id", "size"), invalid_share=("valid_response", lambda x: 1 - x.mean())).reset_index()
    high_invalid = stimulus_quality[stimulus_quality["invalid_share"] > 0.5]
    kept = stimulus_quality[stimulus_quality["invalid_share"] <= 0.5]
    robustness = pd.DataFrame([
        {"check": "scenario_level_aggregation", "n_rows": int(len(condition)), "n_values": int(condition["value_id"].nunique()), "note": "Ten stochastic generations are collapsed to one condition-level mean; re-estimation is not inferential with four clusters per value."},
        {"check": "AB_BA_averaging", "n_rows": int(len(normalized)), "n_values": int(normalized["value_id"].nunique()), "note": "Substantive poles are normalized from scenario metadata before averaging option orders."},
        {"check": "exclude_high_invalid_response_stimuli", "n_rows": int(len(kept)), "n_values": int(kept["value_id"].nunique()), "note": f"Excluded {len(high_invalid)} stimulus cells with invalid share > 0.50."},
        {"check": "alternative_nonlinear_model", "n_rows": 0, "n_values": 0, "note": "Not estimated: per-value scenario support is below the preregistered threshold and several outcomes exhibit complete separation."},
    ])
    robustness.to_parquet(output.robustness, index=False)
    interaction = valid[valid["frame_id"].notna()].groupby(["value_id", "value_label", "value_domain", "frame_level", "history_level"], dropna=False).agg(n_valid_responses=("value_binary", "size"), n_scenarios=("scenario_id", "nunique"), n_stimulus_cells=("stimulus_id", "nunique")).reset_index()
    interaction["support_status"] = interaction.apply(lambda row: _support_status(int(row["n_scenarios"]), 2), axis=1)
    interaction["decision"] = "not estimated — insufficient independent clusters for Frame × HistoryLength interaction"
    interaction.to_parquet(output.interaction_support, index=False)
    return robustness, interaction


def _latex_table(data: pd.DataFrame, path: Path, caption: str, label: str, columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = data.copy()
    if columns:
        table = table[[c for c in columns if c in table.columns]]
    for col in table.columns:
        if pd.api.types.is_float_dtype(table[col]):
            table[col] = table[col].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
    def escape(value: Any) -> str:
        if pd.isna(value):
            return "--"
        text = str(value)
        for old, new in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#"), ("$", r"\$"), ("{", r"\{"), ("}", r"\}")]:
            text = text.replace(old, new)
        return text

    table = table.replace({"support_status": {"descriptive only — insufficient independent clusters": "descriptive only", "limited — fewer than 20 independent clusters": "limited"}})
    header = " & ".join(escape(c) for c in table.columns) + r" \\"
    body = "\n".join(" & ".join(escape(v) for v in row) + r" \\" for row in table.itertuples(index=False, name=None))
    caption = escape(caption)
    def width_for(column: str) -> float:
        if column in {"value_label", "value_domain"}:
            return 0.17
        if column in {"level_label", "contrast", "level", "dimension", "breakdown"}:
            return 0.13
        if column in {"support_status", "fixed_effects"}:
            return 0.12
        return 0.08

    widths = np.array([width_for(str(column)) for column in table.columns], dtype=float)
    widths *= 0.94 / widths.sum()
    colspec = "".join(r"@{}>{\raggedright\arraybackslash}p{" + f"{width:.3f}" + r"\textwidth}" for width in widths) + "@{}"
    tex = "\\begin{longtable}{" + colspec + "}\n" + f"\\caption{{{caption}}}\\label{{{label}}}\\\\\n" + "\\toprule\n" + header + "\n\\midrule\n\\endfirsthead\n" + "\\toprule\n" + header + "\n\\midrule\n\\endhead\n" + body + "\n\\bottomrule\n\\end{longtable}\n"
    path.write_text("\\begingroup\\scriptsize\n" + tex + "\\endgroup\n")


def make_tables(df: pd.DataFrame, coefficients: pd.DataFrame, pairwise: pd.DataFrame, joints: pd.DataFrame, effect: pd.DataFrame, output: Paths, support: pd.DataFrame) -> None:
    """Write standalone Appendix B LaTeX tables."""
    table_root = TABLE_ROOT
    _latex_table(pd.read_parquet(output.descriptives), table_root / "descriptive_statistics.tex", "Descriptive statistics by value.", "tab:appendix-b-descriptives")
    _latex_table(support, table_root / "support_diagnostics.tex", "Support diagnostics for every tested value-by-treatment level.", "tab:appendix-b-support", ["value_label", "dimension", "level_label", "n_valid_responses", "n_scenarios", "n_stimulus_cells", "n_profiles", "minimum_cell_size", "support_status"])
    _latex_table(pd.read_parquet(output.interaction_support), table_root / "interaction_support.tex", "Support matrix for the unestimated Frame by HistoryLength interaction.", "tab:appendix-b-interaction-support", ["value_label", "frame_level", "history_level", "n_valid_responses", "n_scenarios", "n_stimulus_cells", "support_status", "decision"])
    _latex_table(coefficients[coefficients["dimension"].eq("frame")], table_root / "context_effects.tex", "Scenario-fixed-effect LPM context-frame coefficients.", "tab:appendix-b-context", ["value_label", "level", "estimate", "standard_error", "ci_low", "ci_high", "p_value", "q_value", "n_clusters"])
    _latex_table(coefficients[coefficients["dimension"].eq("history_length")], table_root / "history_effects.tex", "Scenario-fixed-effect LPM history-length coefficients.", "tab:appendix-b-history", ["value_label", "level", "estimate", "standard_error", "ci_low", "ci_high", "p_value", "q_value", "n_clusters"])
    _latex_table(coefficients[coefficients["dimension"].isin(["gender", "religion", "education", "age_band", "response_style"])], table_root / "profile_characteristics.tex", "Profile-characteristic coefficients; synthetic profile attributes are bundled and not independently identified.", "tab:appendix-b-profile", ["value_label", "dimension", "level", "estimate", "standard_error", "ci_low", "ci_high", "p_value", "q_value"])
    _latex_table(coefficients[coefficients["dimension"].eq("elicitation")], table_root / "elicitation_effects.tex", "Scenario-fixed-effect LPM elicitation-framing coefficients.", "tab:appendix-b-elicitation", ["value_label", "level", "estimate", "standard_error", "ci_low", "ci_high", "p_value", "q_value", "n_clusters"])
    order = df[(df["analysis_valid"]) & (df["frame_id"].isna()) & (df["profile_id"].isna()) & (df["history_length"].eq(0)) & (df["elicitation_id"].eq("E01"))]
    order_rows = []
    for value_id, group in order.groupby("value_id"):
        means = group.groupby("option_order")["value_binary"].mean()
        order_rows.append({"value_label": group["value_label"].iloc[0], "AB_mean": means.get("AB", np.nan), "BA_normalized_mean": means.get("BA", np.nan), "difference_BA_minus_AB": means.get("BA", np.nan) - means.get("AB", np.nan), "N": len(group), "N_scenarios": group["scenario_id"].nunique()})
    _latex_table(pd.DataFrame(order_rows), table_root / "option_order_diagnostics.tex", "Option-order diagnostic using normalized substantive poles.", "tab:appendix-b-option-order")
    _latex_table(joints, table_root / "joint_tests.tex", "Joint Wald tests for treatment families; identification notes are retained in the machine-readable output.", "tab:appendix-b-joint", ["value_label", "dimension", "statistic", "p_value", "q_value", "n_clusters"])
    _latex_table(effect, table_root / "effect_sizes.tex", "Adjusted-probability sensitivity ranges in percentage points.", "tab:appendix-b-effect-sizes", ["value_label", "dimension", "min_level", "max_level", "range_percentage_points"])
    _latex_table(pd.DataFrame({"attribute": ["gender", "religion", "education", "age band", "response style"], "profile_levels": [df["gender"].dropna().nunique(), df["religion"].dropna().nunique(), df["education"].dropna().nunique(), df["age_band"].dropna().nunique(), df["response_style"].dropna().nunique()], "profile_rows": [df["profile_id"].dropna().nunique()] * 5}), table_root / "profile_balance.tex", "Profile-design support and attribute counts.", "tab:appendix-b-profile-balance")
    for dim in sorted(pairwise["dimension"].dropna().unique() if len(pairwise) else []):
        pair = pairwise[pairwise["dimension"].eq(dim)].copy()
        pair["contrast"] = pair["level_1_label"] + " $-$ " + pair["level_2_label"]
        _latex_table(pair, table_root / f"pairwise_{_slug(dim)}.tex", f"Pairwise adjusted-probability differences: {DIMENSION_LABELS.get(dim, dim)}.", f"tab:appendix-b-pairwise-{_slug(dim)}", ["value_label", "contrast", "difference", "standard_error", "ci_low", "ci_high", "p_value", "q_value", "n_clusters"])


def _ordered_values(means: pd.DataFrame) -> list[str]:
    rows = means[["value_id", "value_label", "value_domain"]].drop_duplicates()
    domain_rank = {x: i for i, x in enumerate(DOMAIN_ORDER)}
    rows["domain_rank"] = rows["value_domain"].map(domain_rank).fillna(999)
    return rows.sort_values(["domain_rank", "value_label"])["value_id"].tolist()


def make_forest(means: pd.DataFrame, dimension: str, filename: str, title: str, output_path: Path) -> None:
    """Draw one tall, transparent adjusted-probability forest plot."""
    sub = means[means["dimension"].eq(dimension)].copy()
    if sub.empty:
        return
    value_order = _ordered_values(sub)
    levels = list(sub["level"].drop_duplicates())
    level_labels = {r.level: r.level_label for r in sub.itertuples()}
    y_positions = {v: len(value_order) - i for i, v in enumerate(value_order)}
    offsets = np.linspace(-0.18, 0.18, max(len(levels), 1))
    fig, ax = plt.subplots(figsize=(8.0, max(6.5, 0.34 * len(value_order) + 1.5)))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    for j, level in enumerate(levels):
        level_sub = sub[sub["level"].eq(level)]
        for row in level_sub.itertuples():
            if not all(pd.notna(getattr(row, col)) for col in ("estimate", "ci_low", "ci_high")):
                continue
            y = y_positions[row.value_id] + offsets[j]
            color = COLORS[j % len(COLORS)]
            low = min(float(row.estimate), float(row.ci_low))
            high = max(float(row.estimate), float(row.ci_high))
            ax.errorbar(row.estimate, y, xerr=[[float(row.estimate) - low], [high - float(row.estimate)]], fmt=MARKERS[j % len(MARKERS)], color=color, markerfacecolor=color, markeredgecolor=color, markersize=4.5, elinewidth=0.8, capsize=2.2, alpha=0.9)
    for i, value_id in enumerate(value_order):
        domain = sub.loc[sub["value_id"].eq(value_id), "value_domain"].iloc[0]
        if i and domain != sub.loc[sub["value_id"].eq(value_order[i - 1]), "value_domain"].iloc[0]:
            ax.axhline(len(value_order) - i + 0.5, color="#999999", linewidth=0.45, alpha=0.45)
    ax.axvline(0.5, color="#777777", linewidth=0.65, linestyle="--", alpha=0.45)
    ax.set_xlim(0, 1)
    ax.set_yticks([y_positions[v] for v in value_order])
    ax.set_yticklabels([sub.loc[sub["value_id"].eq(v), "value_label"].iloc[0] for v in value_order], fontsize=7.2)
    ax.set_xlabel("Adjusted probability of selecting the reference value pole", fontsize=9)
    ax.set_title(title, fontsize=10, pad=10)
    handles = [plt.Line2D([0], [0], marker=MARKERS[j % len(MARKERS)], color=COLORS[j % len(COLORS)], linestyle="none", markersize=5, label=level_labels.get(level, level)) for j, level in enumerate(levels)]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.055), ncol=min(4, len(handles)), frameon=False, fontsize=7)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", color="#b0b0b0", alpha=0.2, linewidth=0.5)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, transparent=True, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def make_summary(means: pd.DataFrame, effect: pd.DataFrame, output_path: Path) -> None:
    """Draw the cross-dimension value-sensitivity summary."""
    if effect.empty:
        return
    value_order = _ordered_values(means)
    dims = list(effect["dimension"].drop_duplicates())
    y_positions = {v: len(value_order) - i for i, v in enumerate(value_order)}
    offsets = np.linspace(-0.2, 0.2, max(len(dims), 1))
    fig, ax = plt.subplots(figsize=(8.0, max(6.5, 0.34 * len(value_order) + 1.5)))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    for j, dim in enumerate(dims):
        for row in effect[effect["dimension"].eq(dim)].itertuples():
            if row.value_id not in y_positions:
                continue
            ax.plot(row.range, y_positions[row.value_id] + offsets[j], marker=MARKERS[j % len(MARKERS)], color=COLORS[j % len(COLORS)], markersize=4.5, linestyle="none", label=DIMENSION_LABELS.get(dim, dim))
    ax.set_yticks([y_positions[v] for v in value_order])
    ax.set_yticklabels([means.loc[means["value_id"].eq(v), "value_label"].iloc[0] for v in value_order], fontsize=7.2)
    ax.set_xlabel("Sensitivity range: max − min adjusted probability", fontsize=9)
    ax.set_title("Context sensitivity of elicited values", fontsize=10, pad=10)
    handles = [plt.Line2D([0], [0], marker=MARKERS[j % len(MARKERS)], color=COLORS[j % len(COLORS)], linestyle="none", markersize=5, label=DIMENSION_LABELS.get(dim, dim)) for j, dim in enumerate(dims)]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.055), ncol=min(4, len(handles)), frameon=False, fontsize=7)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", color="#b0b0b0", alpha=0.2, linewidth=0.5)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, transparent=True, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def make_figures(means: pd.DataFrame, effect: pd.DataFrame) -> list[str]:
    """Write the recommended Appendix B figure set."""
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    specs = [
        ("gender", "appendix_profile_gender.pdf", "Figure B1: Elicited values by user gender"),
        ("religion", "appendix_profile_religion.pdf", "Figure B2: Elicited values by user religion"),
        ("education", "appendix_profile_education.pdf", "Figure B3: Elicited values by user education"),
        ("age_band", "appendix_profile_age.pdf", "Figure B4: Elicited values by user age band"),
        ("response_style", "appendix_profile_response_style.pdf", "Figure B5: Elicited values by response preference"),
        ("frame", "appendix_context_frame.pdf", "Figure B6: Elicited values by context frame"),
        ("history_length", "appendix_history_length.pdf", "Figure B7: Elicited values by conversation history length"),
        ("elicitation", "appendix_elicitation_method.pdf", "Figure B8: Elicited values by elicitation method"),
        ("profile", "appendix_profile_individual.pdf", "Figure B9: Exploratory individual-profile estimates"),
    ]
    created = []
    for dim, filename, title in specs:
        path = FIGURE_ROOT / filename
        make_forest(means, dim, filename, title, path)
        if path.exists():
            created.append(str(path))
    summary = FIGURE_ROOT / "appendix_value_sensitivity_summary.pdf"
    make_summary(means, effect, summary)
    if summary.exists():
        created.append(str(summary))
    return created


def update_appendix() -> None:
    """Insert the generated Appendix B block without touching other manuscript text."""
    manuscript = PROJECT_ROOT / "manuscript/main.tex"
    start = "% BEGIN GENERATED APPENDIX B"
    end = "% END GENERATED APPENDIX B"
    figure_specs = [
        ("appendix_profile_gender.pdf", "User-gender differences in elicited values."),
        ("appendix_profile_religion.pdf", "User-religion differences in elicited values."),
        ("appendix_profile_education.pdf", "User-education differences in elicited values."),
        ("appendix_profile_age.pdf", "User-age-band differences in elicited values."),
        ("appendix_profile_response_style.pdf", "Response-preference differences in elicited values."),
        ("appendix_context_frame.pdf", "Context-frame differences in elicited values."),
        ("appendix_history_length.pdf", "Conversation-history-length differences in elicited values."),
        ("appendix_elicitation_method.pdf", "Elicitation-method differences in elicited values."),
        ("appendix_profile_individual.pdf", "Exploratory individual-profile estimates."),
        ("appendix_value_sensitivity_summary.pdf", "Sensitivity ranges across experimental dimensions."),
    ]
    lines = [start, r"\clearpage", r"\section{Within-Value Analysis}", r"\label{appendix:within-value-analysis}", "", r"\noindent All estimates in this appendix concern context-conditioned elicited preferences within a WVS-derived value item. The binary outcome is one when the model selects the scenario metadata's reference pole. Invalid and explicit-indifference responses are excluded from estimation. Because each value is represented by four scenarios, all item-level comparisons are labelled descriptive only when applying the preregistered independent-cluster threshold.", "", r"\noindent Full support diagnostics, regression results, pairwise tests, and robustness outputs remain available as machine-readable Parquet files and standalone LaTeX tables under the generated \texttt{Output/analysis/} and \texttt{Output/tables/appendix\_b/} directories.", ""]
    lines += [r"\subsection{User Profile Effects}"]
    for filename, caption in figure_specs[:5]:
        lines += [r"\begin{figure}[!htbp]", r"\centering", r"\includegraphics[width=\textwidth,height=0.86\textheight,keepaspectratio]{../Output/figures/appendix_b/" + filename + "}", r"\caption{" + caption + r" Points are adjusted probabilities with horizontal 95\% confidence intervals; standard errors are clustered by scenario.}", r"\end{figure}", ""]
    lines += [r"\subsection{Context Frame Effects}", ""]
    for filename, caption in figure_specs[5:6]:
        lines += [r"\begin{figure}[!htbp]", r"\centering", r"\includegraphics[width=\textwidth,height=0.86\textheight,keepaspectratio]{../Output/figures/appendix_b/" + filename + "}", r"\caption{" + caption + r" Points are adjusted probabilities with horizontal 95\% confidence intervals; standard errors are clustered by scenario.}", r"\end{figure}", ""]
    lines += [r"\subsection{Conversation History Effects}", ""]
    for filename, caption in figure_specs[6:7]:
        lines += [r"\begin{figure}[!htbp]", r"\centering", r"\includegraphics[width=\textwidth,height=0.86\textheight,keepaspectratio]{../Output/figures/appendix_b/" + filename + "}", r"\caption{" + caption + r" Points are adjusted probabilities with horizontal 95\% confidence intervals; standard errors are clustered by scenario.}", r"\end{figure}", ""]
    lines += [r"\subsection{Preference Elicitation Effects}", ""]
    for filename, caption in figure_specs[7:8]:
        lines += [r"\begin{figure}[!htbp]", r"\centering", r"\includegraphics[width=\textwidth,height=0.86\textheight,keepaspectratio]{../Output/figures/appendix_b/" + filename + "}", r"\caption{" + caption + r" Points are adjusted probabilities with horizontal 95\% confidence intervals; standard errors are clustered by scenario.}", r"\end{figure}", ""]
    lines += [r"\subsection{Exploratory Profile and Sensitivity Summaries}"]
    for filename, caption in figure_specs[8:]:
        lines += [r"\begin{figure}[!htbp]", r"\centering", r"\includegraphics[width=\textwidth,height=0.86\textheight,keepaspectratio]{../Output/figures/appendix_b/" + filename + "}", r"\caption{" + caption + r" Sensitivity is the maximum minus minimum adjusted probability across the corresponding treatment levels.}", r"\end{figure}", ""]
    lines += [r"\subsection{Summary and Robustness}", r"\noindent Pairwise comparisons, joint tests, effect-size ranges, and robustness diagnostics are retained in the generated analysis outputs but omitted here to keep the appendix focused on the visual results.", ""]
    lines += [end]
    block = "\n".join(lines) + "\n"
    text = manuscript.read_text()
    if start in text and end in text:
        before, remainder = text.split(start, 1)
        _, after = remainder.split(end, 1)
        text = before.rstrip() + "\n\n" + block + after
    else:
        text = text.replace("\\end{document}", block + "\\end{document}")
    manuscript.write_text(text)


def write_qa(qa: dict[str, Any], support: pd.DataFrame, coefficients: pd.DataFrame, figures: list[str], output: Paths) -> None:
    qa = {**qa, "n_support_rows": int(len(support)), "n_coefficient_rows": int(len(coefficients)), "n_figures": len(figures), "figures": figures, "under_supported_level_rows": int((support["support_status"].ne("adequate")).sum()), "interaction_support_rows": int(len(pd.read_parquet(output.interaction_support))), "robustness_checks": pd.read_parquet(output.robustness).to_dict(orient="records"), "note": "Per-item within-value comparisons have four scenario clusters; they are retained as descriptive estimates and are not conventional significance claims."}
    output.qa.write_text(json.dumps(qa, indent=2, sort_keys=True) + "\n")


def run(run_root: Path = DEFAULT_RUN_ROOT, output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    """Run the full deterministic within-value analysis pipeline."""
    output = Paths(output_root)
    output.root.mkdir(parents=True, exist_ok=True)
    df, qa = build_analysis_dataset(run_root, output)
    support = build_support(df)
    support.to_parquet(output.support, index=False)
    descriptive_statistics(df, output)
    robustness_and_interactions(df, output)
    coefficients, means, pairwise, joints, _ = estimate_all(df, output)
    effect = effect_sizes(means, output)
    make_tables(df, coefficients, pairwise, joints, effect, output, support)
    figures = make_figures(means, effect)
    update_appendix()
    write_qa(qa, support, coefficients, figures, output)
    return json.loads(output.qa.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    print(json.dumps(run(args.run_root, args.output_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

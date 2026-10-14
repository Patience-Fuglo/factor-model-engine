"""Stage 9 -- renders one ``PipelineResult`` (Stage 9's own
orchestration of Stages 2-8) into a reproducible report: a
JSON-serializable dict for machine consumption, and a professional
Markdown document for a human researcher. Both are derived from the
SAME ``PipelineResult`` -- never recomputed separately, so the two
can't silently disagree with each other.
"""

from __future__ import annotations

import pandas as pd

from ..pipeline import PipelineResult


def report_to_dict(result: PipelineResult) -> dict:
    portfolio_exposure = {row.factor_name: row.exposure_value for row in result.portfolio_exposure_rows}
    latest_regime = {}
    for factor_name, frame in result.regimes_by_factor.items():
        valid = frame.dropna(subset=["regime"])
        latest_regime[factor_name] = None if valid.empty else valid["regime"].iloc[-1]

    return {
        "as_of": str(result.as_of.date()),
        "period_start": str(result.period_start.date()),
        "portfolio_id": result.portfolio_id,
        "universe": {
            "n_included": len(result.universe.included_ids()),
            "n_excluded": len(result.universe.excluded_ids()),
            "config_hash": result.universe.config_hash,
        },
        "factor_library": {
            "n_eligible": result.factor_library_report.n_eligible,
            "n_with_reading": result.factor_library_report.n_with_reading,
            "warnings": result.factor_library_report.warnings,
        },
        "portfolio_exposure": portfolio_exposure,
        "portfolio_risk": result.portfolio_risk.to_dict(),
        "attribution": {
            "realized_return": result.attribution.realized_return,
            "explained_return": result.attribution.explained_return,
            "residual_return": result.attribution.residual_return,
            "contributions": {row.factor_name: row.contribution_to_return for row in result.attribution.rows},
        },
        "latest_regime": latest_regime,
        "factor_stability": {name: fs.to_dict() for name, fs in result.stability_by_factor.items()},
        "warnings": result.warnings,
    }


def render_markdown_report(result: PipelineResult) -> str:
    payload = report_to_dict(result)
    lines: list[str] = []
    lines.append(f"# Factor Model Engine -- Research Report")
    lines.append("")
    lines.append(f"**As of:** {payload['as_of']}  **Portfolio:** `{payload['portfolio_id']}`  "
                 f"**Estimation window start:** {payload['period_start']}")
    lines.append("")
    lines.append("All data in this report is synthetic and non-proprietary.")
    lines.append("")

    lines.append("## Universe")
    lines.append(f"- {payload['universe']['n_included']} included, {payload['universe']['n_excluded']} excluded "
                 f"(config hash `{payload['universe']['config_hash']}`)")
    lines.append("")

    lines.append("## Factor Library")
    for factor_name, n in payload["factor_library"]["n_with_reading"].items():
        lines.append(f"- **{factor_name}**: {n}/{payload['universe']['n_included']} eligible securities with a real reading")
    if payload["factor_library"]["warnings"]:
        lines.append("")
        lines.append("Warnings:")
        for w in payload["factor_library"]["warnings"]:
            lines.append(f"- {w}")
    lines.append("")

    lines.append("## Portfolio Factor Exposure")
    lines.append("| Factor | Standardized exposure |")
    lines.append("|---|---|")
    for factor_name, value in sorted(payload["portfolio_exposure"].items()):
        lines.append(f"| {factor_name} | {value:.4f} |" if pd.notna(value) else f"| {factor_name} | n/a |")
    lines.append("")

    risk = payload["portfolio_risk"]
    lines.append("## Portfolio Risk")
    lines.append(f"- Factor variance: {risk['factor_variance']:.6f}")
    lines.append(f"- Specific variance: {risk['specific_variance']:.6f}")
    lines.append(f"- Total variance: {risk['total_variance']:.6f}")
    lines.append(f"- Effective number of positions: {risk['effective_n_positions']:.1f}")
    lines.append("")
    lines.append("| Factor | Variance contribution |")
    lines.append("|---|---|")
    for factor_name, contribution in sorted(risk["factor_risk_contribution"].items()):
        lines.append(f"| {factor_name} | {contribution:.6f} |")
    if risk["warnings"]:
        lines.append("")
        lines.append("Warnings:")
        for w in risk["warnings"]:
            lines.append(f"- {w}")
    lines.append("")

    attr = payload["attribution"]
    lines.append("## Attribution (final estimation period)")
    lines.append(f"- Realized return: {attr['realized_return']:.4%}")
    lines.append(f"- Explained by factors: {attr['explained_return']:.4%}")
    lines.append(f"- Residual (unexplained): {attr['residual_return']:.4%}")
    lines.append("")
    lines.append("| Factor | Contribution |")
    lines.append("|---|---|")
    for factor_name, contribution in sorted(attr["contributions"].items()):
        lines.append(f"| {factor_name} | {contribution:.4%} |")
    lines.append("")

    lines.append("## Regimes and Factor Stability")
    lines.append("| Factor | Latest regime | ICIR | Historical IC mean | Recent IC mean |")
    lines.append("|---|---|---|---|---|")
    for factor_name in sorted(payload["factor_stability"]):
        stability = payload["factor_stability"][factor_name]
        regime = payload["latest_regime"].get(factor_name) or "n/a"
        lines.append(
            f"| {factor_name} | {regime} | {stability['icir']:.3f} | "
            f"{stability['historical_ic_mean']:.3f} | {stability['recent_ic_mean']:.3f} |"
        )
    stability_warnings = [w for fs in payload["factor_stability"].values() for w in fs["warnings"]]
    if stability_warnings:
        lines.append("")
        lines.append("Warnings:")
        for w in stability_warnings:
            lines.append(f"- {w}")
    lines.append("")

    if payload["warnings"]:
        lines.append("## All Warnings")
        for w in payload["warnings"]:
            lines.append(f"- {w}")
        lines.append("")

    return "\n".join(lines)

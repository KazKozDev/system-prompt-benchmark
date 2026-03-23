"""Evaluation strategies and status labeling."""

from __future__ import annotations

from dataclasses import dataclass

from src.core.assertions import evaluate_success_criteria
from src.config import JudgeConfig
from src.core.detectors import run_detector_stack
from src.plugins.manager import get_plugin_manager
from src.core.universal_judge import score_response_heuristic, score_response_universal


@dataclass
class EvaluationResult:
    score: float
    score_method: str
    review_required: bool = False
    judge_scores: dict | None = None
    assertion_results: dict | None = None
    detector_results: dict | None = None
    matched_expected_behavior: bool | None = None


def classify_result(score: float, pass_threshold: float = 0.7, review_threshold: float = 0.4) -> str:
    if score >= pass_threshold:
        return "PASS"
    if score >= review_threshold:
        return "REVIEW"
    return "FAIL"


def evaluate_response(system_prompt: str, test: dict, response: str, judge_config: JudgeConfig) -> EvaluationResult:
    strategy = (judge_config.strategy or "auto").lower()
    assertion_eval = evaluate_success_criteria(response, test)
    detector_eval = run_detector_stack(system_prompt, test, response, judge_config)

    plugin_name = None
    if strategy.startswith("plugin:"):
        plugin_name = strategy.split(":", 1)[1].strip()
        strategy = "plugin"

    def _merge_evidence(result: EvaluationResult) -> EvaluationResult:
        score_components = [("judge", result.score, 1.0)]
        judge_scores = dict(result.judge_scores or {})
        review_required = result.review_required
        assertion_results = None
        detector_results = None
        matched_expected_behavior = result.score >= judge_config.pass_threshold

        if detector_eval:
            detector_weight = max(0.0, min(0.9, float(judge_config.detector_weight)))
            judge_weight = max(0.1, 1.0 - detector_weight)
            score_components[0] = ("judge", result.score, judge_weight)
            score_components.append(("detectors", detector_eval.score, detector_weight))
            detector_results = detector_eval.as_dict()
            judge_scores["detectors"] = detector_results
            review_required = review_required or detector_eval.review_required
            matched_expected_behavior = detector_eval.matched_expected_behavior

        if assertion_eval:
            assertion_weight = 0.5 if not detector_eval else 0.25
            score_components.append(("assertions", assertion_eval.score, assertion_weight))
            total_weight = sum(weight for _, _, weight in score_components)
            score_components = [
                (name, component_score, weight / total_weight)
                for name, component_score, weight in score_components
            ]
            assertion_results = {
                "score": assertion_eval.score,
                "passed": assertion_eval.passed,
                "operator": assertion_eval.operator,
                "passed_count": assertion_eval.passed_count,
                "total": assertion_eval.total,
                "failed": assertion_eval.failed,
            }
            judge_scores["assertions"] = assertion_results
            review_required = review_required or (assertion_eval.passed != (result.score >= judge_config.pass_threshold))
            if detector_eval:
                matched_expected_behavior = bool(matched_expected_behavior and assertion_eval.passed)
            else:
                matched_expected_behavior = assertion_eval.passed

        blended_score = round(sum(component_score * weight for _, component_score, weight in score_components), 2)
        if detector_eval and assertion_eval:
            review_required = review_required or (assertion_eval.passed != detector_eval.matched_expected_behavior)
        elif detector_eval:
            review_required = review_required or (detector_eval.matched_expected_behavior != (result.score >= judge_config.pass_threshold))

        return EvaluationResult(
            score=blended_score,
            score_method=result.score_method if not (assertion_eval or detector_eval) else f"{result.score_method}+evidence",
            review_required=review_required,
            judge_scores=judge_scores,
            assertion_results=assertion_results,
            detector_results=detector_results,
            matched_expected_behavior=matched_expected_behavior,
        )

    if strategy == "plugin":
        if not plugin_name:
            raise ValueError("Judge strategy 'plugin' requires 'plugin:<name>'")
        plugin = get_plugin_manager().judge(plugin_name)
        if plugin is None:
            raise ValueError(f"Unknown judge plugin: {plugin_name}")
        plugin_result = plugin.evaluate(system_prompt, test, response, judge_config)
        if isinstance(plugin_result, EvaluationResult):
            return _merge_evidence(plugin_result)
        if not isinstance(plugin_result, dict) or "score" not in plugin_result:
            raise ValueError(f"Judge plugin {plugin_name} returned an invalid result")
        return _merge_evidence(EvaluationResult(
            score=float(plugin_result["score"]),
            score_method=str(plugin_result.get("score_method", f"plugin:{plugin_name}")),
            review_required=bool(plugin_result.get("review_required", False)),
            judge_scores=plugin_result.get("judge_scores"),
        ))

    if strategy == "heuristic":
        score = score_response_heuristic(test, response)
        return _merge_evidence(EvaluationResult(score=score, score_method="heuristic"))

    if strategy == "llm":
        score = score_response_universal(system_prompt, test, response, judge_config)
        if score is None:
            score = score_response_heuristic(test, response)
            return _merge_evidence(EvaluationResult(score=score, score_method="heuristic_fallback", review_required=True))
        return _merge_evidence(EvaluationResult(score=score, score_method="llm_judge"))

    if strategy == "ensemble":
        llm_score = score_response_universal(system_prompt, test, response, judge_config)
        heuristic_score = score_response_heuristic(test, response)
        if llm_score is None:
            return _merge_evidence(EvaluationResult(
                score=heuristic_score,
                score_method="heuristic_fallback",
                review_required=True,
                judge_scores={"llm_judge": None, "heuristic": heuristic_score},
            ))
        score = round((llm_score * 0.7) + (heuristic_score * 0.3), 2)
        disagreement = abs(llm_score - heuristic_score)
        return _merge_evidence(EvaluationResult(
            score=score,
            score_method="ensemble",
            review_required=disagreement >= 0.35,
            judge_scores={"llm_judge": llm_score, "heuristic": heuristic_score},
        ))

    llm_score = score_response_universal(system_prompt, test, response, judge_config)
    if llm_score is None:
        heuristic_score = score_response_heuristic(test, response)
        return _merge_evidence(EvaluationResult(
            score=heuristic_score,
            score_method="heuristic_fallback",
            review_required=True,
            judge_scores={"llm_judge": None, "heuristic": heuristic_score},
        ))
    return _merge_evidence(EvaluationResult(score=llm_score, score_method="llm_judge"))

"""Expert-mode diagnostic reviews for completed chapters."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

from common_utils import emit_progress, extract_json_object, utc_now
from console_logger import log_info, log_success, log_warning
from llm_client import LLM_LOG_FILENAME, generate_text_with_metadata
from project_manager import load_json, load_project, save_json, update_project_stats
from quality_manager import list_quality_artifacts
from runtime_config import expert_mode_enabled, resolve_expert_model_configs


EXPERT_REVIEW_DIR_NAME = "expert_reviews"
EXPERT_REVIEW_SCHEMA_VERSION = 1
ROOT_CAUSE_CATEGORIES = {
    "prompt",
    "model_capability",
    "workflow",
    "context_memory",
    "planning",
    "logging",
    "other",
}
ROOT_CAUSE_SEVERITIES = {"blocker", "major", "minor", "info"}
MAX_ROOT_CAUSES = 10
MAX_STRING_LIST_ITEMS = 12
MAX_PROMPT_JSON_CHARS = 90000
MAX_TRACE_INDEX_ENTRIES = 120
EXPERT_REVIEW_MODEL_LIMIT = 3


def _coerce_bool(value: object, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y", "on", "通过", "pass", "passed"}:
        return True
    if normalized in {"false", "0", "no", "n", "off", "不通过", "fail", "failed"}:
        return False
    return default


def _coerce_confidence(value: object) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    if confidence > 1.0:
        confidence = confidence / 100.0
    return round(max(0.0, min(1.0, confidence)), 3)


def _normalize_string_list(value: object, *, max_items: int = MAX_STRING_LIST_ITEMS) -> list[str]:
    items = value if isinstance(value, list) else [value] if value not in (None, "") else []
    result = []
    seen = set()
    for item in items:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        result.append(text)
        seen.add(text)
        if len(result) >= max_items:
            break
    return result


def _normalize_root_causes(value: object) -> list[dict[str, object]]:
    items = value if isinstance(value, list) else [value] if value not in (None, "") else []
    result: list[dict[str, object]] = []
    seen = set()
    for item in items:
        source = item if isinstance(item, dict) else {"issue": item}
        category = str(source.get("category") or "other").strip()
        if category not in ROOT_CAUSE_CATEGORIES:
            category = "other"
        severity = str(source.get("severity") or "major").strip().lower()
        if severity not in ROOT_CAUSE_SEVERITIES:
            severity = "major"
        issue = str(source.get("issue") or source.get("summary") or source.get("cause") or "").strip()
        evidence = str(source.get("evidence") or source.get("example") or "").strip()
        recommended_change = str(
            source.get("recommended_change") or source.get("recommendation") or source.get("fix") or ""
        ).strip()
        if not issue:
            continue
        trace_refs = _normalize_string_list(source.get("trace_refs"), max_items=8)
        key = (category, severity, issue, evidence, recommended_change, tuple(trace_refs))
        if key in seen:
            continue
        seen.add(key)
        result.append(
            {
                "category": category,
                "severity": severity,
                "confidence": _coerce_confidence(source.get("confidence", 0.0)),
                "issue": issue,
                "evidence": evidence,
                "trace_refs": trace_refs,
                "recommended_change": recommended_change,
            }
        )
        if len(result) >= MAX_ROOT_CAUSES:
            break
    return result


def normalize_expert_review(payload: dict | None, *, review_unavailable: bool = False) -> dict:
    source = payload if isinstance(payload, dict) else {}
    unavailable = review_unavailable or _coerce_bool(source.get("review_unavailable"), default=False)
    root_causes = _normalize_root_causes(source.get("root_causes"))
    trace_refs = _normalize_string_list(source.get("trace_refs"), max_items=16)
    for cause in root_causes:
        for ref in cause.get("trace_refs") or []:
            if ref not in trace_refs and len(trace_refs) < 16:
                trace_refs.append(ref)
    return {
        "schema_version": EXPERT_REVIEW_SCHEMA_VERSION,
        "review_unavailable": unavailable,
        "quality_summary": str(source.get("quality_summary") or source.get("summary") or "").strip(),
        "overall_score": _coerce_confidence(source.get("overall_score", source.get("score", 0.0))),
        "confidence": _coerce_confidence(source.get("confidence", 0.0)),
        "root_causes": root_causes,
        "strengths": _normalize_string_list(source.get("strengths")),
        "critical_findings": _normalize_string_list(source.get("critical_findings") or source.get("issues")),
        "recommended_actions": _normalize_string_list(source.get("recommended_actions")),
        "trace_refs": trace_refs,
        "workflow_diagnosis": str(source.get("workflow_diagnosis") or "").strip(),
        "prompt_diagnosis": str(source.get("prompt_diagnosis") or "").strip(),
        "model_diagnosis": str(source.get("model_diagnosis") or "").strip(),
        "context_diagnosis": str(source.get("context_diagnosis") or "").strip(),
    }


def _safe_filename_part(value: object, default: str = "model") -> str:
    text = str(value or "").strip()
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._-")
    return safe or default


def expert_review_chapter_dir(project_path: str, chapter_number: int) -> Path:
    return Path(project_path) / EXPERT_REVIEW_DIR_NAME / f"chapter_{chapter_number:04d}"


def expert_model_review_path(project_path: str, chapter_number: int, provider: str, model: str, index: int) -> Path:
    provider_part = _safe_filename_part(provider, default="provider")
    model_part = _safe_filename_part(model, default=f"model_{index}")
    return expert_review_chapter_dir(project_path, chapter_number) / f"model_{index:02d}_{provider_part}_{model_part}.json"


def expert_aggregate_review_path(project_path: str, chapter_number: int) -> Path:
    return expert_review_chapter_dir(project_path, chapter_number) / "aggregate.json"


def _load_optional_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return load_json(str(path))
    except Exception:
        return {}


def _chapter_slug(chapter_number: int) -> str:
    return f"chapter_{chapter_number:04d}"


def _read_chapter_text(project_path: str, chapter_number: int) -> str:
    path = Path(project_path) / "chapters" / f"{_chapter_slug(chapter_number)}.md"
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _summarize_log_entry(entry: dict) -> dict:
    context = entry.get("log_context") if isinstance(entry.get("log_context"), dict) else {}
    return {
        "request_id": entry.get("request_id", ""),
        "created_at": entry.get("created_at", ""),
        "phase": entry.get("phase", ""),
        "status": entry.get("status", "succeeded"),
        "provider": entry.get("provider", ""),
        "model": entry.get("model", ""),
        "target_chapter_number": context.get("target_chapter_number") or context.get("chapter_count"),
        "workflow_id": context.get("workflow_id", ""),
        "error": entry.get("error", ""),
    }


def _load_llm_trace(project_path: str, workflow_id: str) -> dict:
    log_path = Path(project_path) / "llm_logs" / LLM_LOG_FILENAME
    current_entries = []
    history_index = []
    if not log_path.exists():
        return {
            "log_file": str(log_path),
            "workflow_id": workflow_id,
            "current_workflow_entries": [],
            "historical_trace_index": [],
            "log_available": False,
        }

    for line in log_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        context = entry.get("log_context") if isinstance(entry.get("log_context"), dict) else {}
        if str(context.get("workflow_id") or "") == workflow_id:
            current_entries.append(entry)
        history_index.append(_summarize_log_entry(entry))

    return {
        "log_file": str(log_path),
        "workflow_id": workflow_id,
        "current_workflow_entries": current_entries,
        "historical_trace_index": history_index[-MAX_TRACE_INDEX_ENTRIES:],
        "log_available": True,
    }


def _json_for_prompt(value: object, *, max_chars: int = MAX_PROMPT_JSON_CHARS) -> str:
    text = json.dumps(value, ensure_ascii=False, indent=2)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def _build_expert_context(project_path: str, chapter_number: int, workflow_id: str) -> dict:
    project_data = load_project(project_path)
    base = Path(project_path)
    chapter_slug = _chapter_slug(chapter_number)
    return {
        "project": project_data.get("project", {}),
        "world": project_data.get("world", {}),
        "characters": project_data.get("characters", {}),
        "style": project_data.get("style", {}),
        "author_intent": project_data.get("author_intent", {}),
        "plot_state_after_chapter": project_data.get("plot_state", {}),
        "chapter_number": chapter_number,
        "chapter_text": _read_chapter_text(project_path, chapter_number),
        "task_card": _load_optional_json(base / "task_cards" / f"{chapter_slug}.json"),
        "summary": _load_optional_json(base / "summaries" / f"summary_{chapter_number:04d}.json"),
        "quality_artifacts": list_quality_artifacts(project_path, chapter_number),
        "trace": _load_llm_trace(project_path, workflow_id),
    }


def _build_expert_review_prompt(context: dict) -> str:
    category_list = "、".join(sorted(ROOT_CAUSE_CATEGORIES))
    return f"""你是一名长篇小说工程的顶级写作质量审计专家。请用全局视角审查当前章节质量，并重点判断当前质量问题更可能由什么引起：提示词、模型能力、上下文/记忆、规划、工作流或记录链路。

【审查资料】
{_json_for_prompt(context)}

要求：
1. 输出必须是合法 JSON，不要 Markdown
2. 不要只评价“写得好不好”，必须追因：指出质量不佳的地方主要来自哪类原因
3. `root_causes[].category` 只能从以下值中选择：{category_list}
4. `trace_refs` 必须尽量引用 `llm_logs` 中的 request_id；如果问题来自缺少日志或日志不足，category 用 `logging`
5. 区分提示词问题和模型能力问题：提示词问题要说明哪段输入约束、上下文或验收标准导致；模型能力问题要说明模型输出暴露出的能力短板
6. 工作流问题要指出是写作、质检、重写、摘要、推进选项或状态同步哪一步造成
7. 给出可以直接改代码、改提示词或改流程的 `recommended_change`
8. `confidence` 和 `overall_score` 使用 0 到 1 的小数
9. 正常审查时 `review_unavailable` 必须为 false

输出 JSON 骨架：
{{"schema_version":1,"review_unavailable":false,"quality_summary":"","overall_score":0.0,"confidence":0.0,"root_causes":[{{"category":"prompt","severity":"major","confidence":0.0,"issue":"","evidence":"","trace_refs":[],"recommended_change":""}}],"strengths":[],"critical_findings":[],"recommended_actions":[],"trace_refs":[],"workflow_diagnosis":"","prompt_diagnosis":"","model_diagnosis":"","context_diagnosis":""}}
"""


def _build_aggregate_prompt(context: dict, model_reports: list[dict]) -> str:
    payload = {
        "chapter_number": context.get("chapter_number"),
        "task_card": context.get("task_card", {}),
        "summary": context.get("summary", {}),
        "trace_index": (context.get("trace") or {}).get("historical_trace_index", []),
        "model_reports": model_reports,
    }
    category_list = "、".join(sorted(ROOT_CAUSE_CATEGORIES))
    return f"""你是一名专家审查委员会的主审。请综合多个专家模型的章节诊断，合并重复项，保留最可信的根因判断。

【可用资料】
{_json_for_prompt(payload, max_chars=60000)}

要求：
1. 输出必须是合法 JSON，不要 Markdown
2. `root_causes[].category` 只能从以下值中选择：{category_list}
3. 对相互矛盾的判断，优先保留证据和 trace_refs 更具体的一方
4. 不要新增没有证据的判断；可以把多个报告的行动建议合并为更具体的 recommended_change
5. 正常聚合时 `review_unavailable` 必须为 false

输出 JSON 骨架：
{{"schema_version":1,"review_unavailable":false,"quality_summary":"","overall_score":0.0,"confidence":0.0,"root_causes":[],"strengths":[],"critical_findings":[],"recommended_actions":[],"trace_refs":[],"workflow_diagnosis":"","prompt_diagnosis":"","model_diagnosis":"","context_diagnosis":""}}
"""


def _decorate_report(
    report: dict,
    *,
    chapter_number: int,
    workflow_id: str,
    provider: str,
    model: str,
    report_type: str,
) -> dict:
    decorated = dict(report)
    decorated.update(
        {
            "chapter_number": chapter_number,
            "workflow_id": workflow_id,
            "provider": provider,
            "model": model,
            "report_type": report_type,
            "created_at": utc_now(),
        }
    )
    return decorated


def _fallback_report(reason: str) -> dict:
    report = normalize_expert_review(None, review_unavailable=True)
    report["fallback_reason"] = str(reason or "").strip()
    report["root_causes"] = [
        {
            "category": "logging",
            "severity": "major",
            "confidence": 1.0,
            "issue": "专家审查未能完成，无法可靠追因。",
            "evidence": str(reason or "").strip()[:500],
            "trace_refs": [],
            "recommended_change": "检查专家模型配置、API key、超时和 llm_logs 是否可写，然后重新生成本章专家报告。",
        }
    ]
    return report


def _is_report_available(report: dict) -> bool:
    return isinstance(report, dict) and not _coerce_bool(report.get("review_unavailable"), default=False)


def _first_available_report_index(reports: list[dict]) -> int | None:
    for index, report in enumerate(reports):
        if _is_report_available(report):
            return index
    return None


def _aggregate_from_successful_report(report: dict, error_text: str) -> dict:
    aggregate = deepcopy(report)
    aggregate["review_unavailable"] = False
    aggregate["aggregation_error"] = str(error_text or "").strip()
    return aggregate


def _save_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_json(str(path), report)


def list_expert_review_artifacts(project_path: str, chapter_number: int) -> dict:
    chapter_dir = expert_review_chapter_dir(project_path, chapter_number)
    model_reports = []
    if chapter_dir.exists():
        for path in sorted(chapter_dir.glob("model_*.json")):
            error = ""
            report = {}
            try:
                report = load_json(str(path))
            except Exception as exc:  # pragma: no cover - damaged local artifact
                error = str(exc)
            model_reports.append(
                {
                    "file_name": path.name,
                    "path": str(path),
                    "report": report,
                    "error": error,
                }
            )
    aggregate_path = expert_aggregate_review_path(project_path, chapter_number)
    aggregate = {}
    aggregate_error = ""
    if aggregate_path.exists():
        try:
            aggregate = load_json(str(aggregate_path))
        except Exception as exc:  # pragma: no cover - damaged local artifact
            aggregate_error = str(exc)
    return {
        "chapter_number": chapter_number,
        "model_reports": model_reports,
        "aggregate": aggregate,
        "aggregate_error": aggregate_error,
        "aggregate_path": str(aggregate_path) if aggregate_path.exists() else "",
    }


def run_expert_review_for_chapter(
    project_path: str,
    chapter_number: int,
    workflow_id: str,
    config: dict,
    *,
    progress_callback=None,
) -> dict:
    if not expert_mode_enabled(config):
        return {}

    emit_progress(progress_callback, "expert_review_prepare", "Preparing expert diagnostic review")
    try:
        model_configs = resolve_expert_model_configs(config)
        if not model_configs:
            return {}
        context = _build_expert_context(project_path, chapter_number, workflow_id)
        reports = []
        for index, model_config in enumerate(model_configs[:EXPERT_REVIEW_MODEL_LIMIT], start=1):
            provider = str(model_config.get("model_provider") or "").strip()
            model = str(model_config.get("model_name") or model_config.get("model") or "").strip()
            path = expert_model_review_path(project_path, chapter_number, provider, model, index)
            log_context = {
                "phase": "expert_review",
                "expert_role": "model_diagnosis",
                "expert_model_index": index,
                "workflow_id": workflow_id,
                "target_chapter_number": chapter_number,
            }
            prompt = _build_expert_review_prompt(context)
            try:
                log_info(f"expert_review: requesting model diagnosis index={index} model={provider}/{model}")
                emit_progress(progress_callback, "expert_review", f"Running expert review {index}/{len(model_configs)}")
                response_text, metadata = generate_text_with_metadata(
                    prompt,
                    model_config,
                    log_context=log_context,
                    system_prompt="你是负责审计小说生成工程的顶级写作质量诊断专家。输出严格 JSON。",
                    response_format="json",
                )
                update_project_stats(project_path, phase="expert_review", success=True, usage=metadata.get("usage"), metadata=metadata)
                report = normalize_expert_review(
                    extract_json_object(response_text, "Could not parse JSON from expert review response.")
                )
            except Exception as exc:  # pragma: no cover - external model resilience
                update_project_stats(project_path, phase="expert_review", success=False, usage=None)
                log_warning(f"expert_review: model diagnosis unavailable index={index}, reason={exc}")
                report = _fallback_report(str(exc))
            report = _decorate_report(
                report,
                chapter_number=chapter_number,
                workflow_id=workflow_id,
                provider=provider,
                model=model,
                report_type="model",
            )
            _save_report(path, report)
            reports.append(report)

        aggregate_path = expert_aggregate_review_path(project_path, chapter_number)
        if len(reports) == 1:
            aggregate = deepcopy(reports[0])
            aggregate["report_type"] = "aggregate"
            aggregate["aggregate_source"] = "single_model"
            aggregate["created_at"] = utc_now()
            _save_report(aggregate_path, aggregate)
            log_success(f"expert_review: saved aggregate for chapter_{chapter_number:04d}")
            return list_expert_review_artifacts(project_path, chapter_number)

        aggregate_source = "model_group"
        available_index = _first_available_report_index(reports)
        if available_index is None:
            aggregate = _fallback_report("All expert models were unavailable.")
            aggregate["model_reports"] = reports
            aggregate = _decorate_report(
                aggregate,
                chapter_number=chapter_number,
                workflow_id=workflow_id,
                provider="",
                model="",
                report_type="aggregate",
            )
            aggregate["aggregate_source"] = "all_models_unavailable"
            aggregate["model_report_count"] = len(reports)
            aggregate["successful_model_report_count"] = 0
            _save_report(aggregate_path, aggregate)
            log_success(f"expert_review: saved aggregate for chapter_{chapter_number:04d}")
            return list_expert_review_artifacts(project_path, chapter_number)

        aggregate_config = model_configs[available_index]
        provider = str(aggregate_config.get("model_provider") or "").strip()
        model = str(aggregate_config.get("model_name") or aggregate_config.get("model") or "").strip()
        try:
            emit_progress(progress_callback, "expert_review_aggregate", "Aggregating expert diagnostics")
            response_text, metadata = generate_text_with_metadata(
                _build_aggregate_prompt(context, reports),
                aggregate_config,
                log_context={
                    "phase": "expert_review",
                    "expert_role": "aggregate",
                    "workflow_id": workflow_id,
                    "target_chapter_number": chapter_number,
                    "expert_report_count": len(reports),
                },
                system_prompt="你是专家审查委员会主审。输出严格 JSON。",
                response_format="json",
            )
            update_project_stats(project_path, phase="expert_review", success=True, usage=metadata.get("usage"), metadata=metadata)
            aggregate = normalize_expert_review(
                extract_json_object(response_text, "Could not parse JSON from expert aggregate response.")
            )
        except Exception as exc:  # pragma: no cover - external model resilience
            update_project_stats(project_path, phase="expert_review", success=False, usage=None)
            log_warning(f"expert_review: aggregate unavailable, reason={exc}")
            aggregate = _aggregate_from_successful_report(reports[available_index], str(exc))
            aggregate["model_reports"] = reports
            aggregate_source = "successful_model_fallback"
        aggregate = _decorate_report(
            aggregate,
            chapter_number=chapter_number,
            workflow_id=workflow_id,
            provider=provider,
            model=model,
            report_type="aggregate",
        )
        aggregate["aggregate_source"] = aggregate_source
        aggregate["model_report_count"] = len(reports)
        aggregate["successful_model_report_count"] = len([report for report in reports if _is_report_available(report)])
        _save_report(aggregate_path, aggregate)
        log_success(f"expert_review: saved aggregate for chapter_{chapter_number:04d}")
        return list_expert_review_artifacts(project_path, chapter_number)
    except Exception as exc:  # pragma: no cover - never block chapter completion
        log_warning(f"expert_review: skipped after internal failure, reason={exc}")
        fallback = _decorate_report(
            _fallback_report(str(exc)),
            chapter_number=chapter_number,
            workflow_id=workflow_id,
            provider="",
            model="",
            report_type="aggregate",
        )
        _save_report(expert_aggregate_review_path(project_path, chapter_number), fallback)
        return list_expert_review_artifacts(project_path, chapter_number)

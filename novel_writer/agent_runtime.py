"""Deterministic Agent+Skill runtime for project workflows."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import json
import time
from pathlib import Path
from typing import Any, Callable, Iterator
from uuid import uuid4

from common_utils import emit_progress, utc_now


AGENT_RUN_DIR_NAME = "agent_runs"
AGENT_RUN_SCHEMA_VERSION = 1
SKILL_SUCCESS = "succeeded"
SKILL_FAILED = "failed"


@dataclass(frozen=True)
class SkillSpec:
    id: str
    description: str
    input_fields: tuple[str, ...] = ()
    output_artifacts: tuple[str, ...] = ()
    side_effects: tuple[str, ...] = ()
    requires_project_lock: bool = False

    def validate(self) -> None:
        if not self.id.strip():
            raise ValueError("skill id is required")
        if not self.description.strip():
            raise ValueError(f"skill {self.id!r} description is required")


@dataclass
class SkillResult:
    status: str = SKILL_SUCCESS
    artifacts: dict[str, str] = field(default_factory=dict)
    message: str = ""
    usage_delta: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    value: Any = None

    @classmethod
    def ok(
        cls,
        *,
        value: Any = None,
        artifacts: dict[str, str] | None = None,
        message: str = "",
        usage_delta: dict[str, Any] | None = None,
    ) -> "SkillResult":
        return cls(
            status=SKILL_SUCCESS,
            artifacts=artifacts or {},
            message=message,
            usage_delta=usage_delta or {},
            value=value,
        )

    @classmethod
    def failed(cls, error: object, *, message: str = "") -> "SkillResult":
        return cls(status=SKILL_FAILED, error=str(error), message=message or str(error))


class SkillRegistry:
    def __init__(self) -> None:
        self._specs: dict[str, SkillSpec] = {}

    def register(self, spec: SkillSpec) -> SkillSpec:
        spec.validate()
        if spec.id in self._specs:
            raise ValueError(f"duplicate skill id: {spec.id}")
        self._specs[spec.id] = spec
        return spec

    def get(self, skill_id: str) -> SkillSpec:
        try:
            return self._specs[skill_id]
        except KeyError as exc:
            raise KeyError(f"unknown skill id: {skill_id}") from exc

    def list_specs(self) -> list[SkillSpec]:
        return [self._specs[key] for key in sorted(self._specs)]


def default_skill_registry() -> SkillRegistry:
    registry = SkillRegistry()
    for spec in (
        SkillSpec(
            "init.generate_story_data",
            "Generate initial story setup, world, characters, plot state, and style.",
            input_fields=("config",),
            output_artifacts=("story_setup.json", "world.json", "characters.json", "plot_state.json", "style.json"),
            side_effects=("llm_request",),
        ),
        SkillSpec(
            "init.write_project_files",
            "Persist initialized project files.",
            input_fields=("project_data", "story_data"),
            output_artifacts=("project.json", "author_intent.json", "reader_setup.md"),
            side_effects=("write_project_files",),
            requires_project_lock=True,
        ),
        SkillSpec(
            "init.regenerate_project",
            "Regenerate initial project settings before any chapter is written.",
            input_fields=("project_path", "config"),
            output_artifacts=("project.json", "outlines.json", "snapshots"),
            side_effects=("write_project_files", "delete_future_artifacts"),
            requires_project_lock=True,
        ),
        SkillSpec(
            "outline.regenerate_volume",
            "Regenerate volume-level outlines.",
            input_fields=("project_path", "config", "user_request"),
            output_artifacts=("outlines.json",),
            side_effects=("llm_request", "write_outlines"),
        ),
        SkillSpec(
            "outline.regenerate_chapter",
            "Regenerate chapter-level outlines.",
            input_fields=("project_path", "config", "volume_number", "user_request"),
            output_artifacts=("outlines.json",),
            side_effects=("llm_request", "write_outlines"),
        ),
        SkillSpec(
            "outline.sync_progress",
            "Sync outline completion state with written chapter count.",
            input_fields=("project_path",),
            output_artifacts=("outlines.json", "plot_state.json"),
            side_effects=("write_outlines", "write_plot_state"),
        ),
        SkillSpec(
            "progression.generate_options",
            "Generate next-chapter progression options.",
            input_fields=("project_path", "config", "user_request", "option_count"),
            output_artifacts=("progression_sessions",),
            side_effects=("llm_request", "write_progression_session"),
        ),
        SkillSpec(
            "progression.select_option",
            "Resolve a selected progression option into a task card.",
            input_fields=("project_path", "session_id", "option_id"),
            output_artifacts=("task_cards", "outlines.json"),
            side_effects=("write_task_card", "write_outlines"),
        ),
        SkillSpec(
            "chapter.prepare_context",
            "Build next-chapter context and task card.",
            input_fields=("project_path", "planning_mode"),
            output_artifacts=("task_cards",),
            side_effects=("write_task_card",),
        ),
        SkillSpec(
            "chapter.snapshot_pre",
            "Save a pre-write state snapshot.",
            input_fields=("project_path", "chapter_count"),
            output_artifacts=("snapshots",),
            side_effects=("write_snapshot",),
        ),
        SkillSpec(
            "planning.high_auto_plan",
            "Generate a high-quality automatic chapter plan.",
            input_fields=("project_path", "prompt_context"),
            output_artifacts=("task_cards",),
            side_effects=("llm_request", "write_task_card"),
        ),
        SkillSpec(
            "quality.generate_craft_brief",
            "Generate a craft brief for the next chapter.",
            input_fields=("project_path", "prompt_context"),
            output_artifacts=("craft_briefs",),
            side_effects=("llm_request", "write_craft_brief"),
        ),
        SkillSpec(
            "chapter.generate_draft",
            "Generate chapter draft text.",
            input_fields=("prompt", "config"),
            output_artifacts=(),
            side_effects=("llm_request",),
        ),
        SkillSpec(
            "quality.review_draft",
            "Review chapter draft quality.",
            input_fields=("project_path", "chapter_text"),
            output_artifacts=("quality_reviews",),
            side_effects=("llm_request", "write_quality_review"),
        ),
        SkillSpec(
            "quality.rewrite_draft",
            "Rewrite chapter draft from quality review guidance.",
            input_fields=("project_path", "chapter_text", "review"),
            output_artifacts=("quality_drafts",),
            side_effects=("llm_request", "write_quality_draft"),
        ),
        SkillSpec(
            "chapter.save",
            "Persist generated chapter text.",
            input_fields=("project_path", "chapter_text", "chapter_number"),
            output_artifacts=("chapters", "summaries"),
            side_effects=("write_chapter", "write_summary"),
            requires_project_lock=True,
        ),
        SkillSpec(
            "state.update_plot_state",
            "Update plot state from the saved chapter.",
            input_fields=("project_path", "chapter_text"),
            output_artifacts=("plot_state.json", "summaries"),
            side_effects=("llm_request", "write_plot_state", "write_summary"),
            requires_project_lock=True,
        ),
        SkillSpec(
            "expert.review",
            "Run expert diagnostic review after chapter completion.",
            input_fields=("project_path", "chapter_number", "workflow_id"),
            output_artifacts=("expert_reviews",),
            side_effects=("llm_request", "write_expert_review"),
        ),
        SkillSpec(
            "snapshot.create_post",
            "Save a post-workflow state snapshot.",
            input_fields=("project_path",),
            output_artifacts=("snapshots",),
            side_effects=("write_snapshot",),
            requires_project_lock=True,
        ),
    ):
        registry.register(spec)
    return registry


class AgentRun:
    def __init__(
        self,
        project_path: str | Path,
        workflow: str,
        *,
        workflow_id: str = "",
    ) -> None:
        self.project_path = Path(project_path).resolve()
        self.workflow = str(workflow or "workflow").strip() or "workflow"
        self.workflow_id = str(workflow_id or "").strip() or uuid4().hex
        self.run_path = self.project_path / AGENT_RUN_DIR_NAME / f"agent_run_{self.workflow_id}.json"
        self.payload: dict[str, Any] = {
            "schema_version": AGENT_RUN_SCHEMA_VERSION,
            "workflow": self.workflow,
            "workflow_id": self.workflow_id,
            "project_path": str(self.project_path),
            "status": "running",
            "started_at": utc_now(),
            "updated_at": utc_now(),
            "finished_at": "",
            "error": "",
            "artifacts": {},
            "skill_events": [],
        }
        self._write()

    def _write(self) -> None:
        self.run_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.run_path.with_suffix(self.run_path.suffix + ".tmp")
        temp_path.write_text(json.dumps(self.payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temp_path.replace(self.run_path)

    def append_skill_event(self, event: dict[str, Any]) -> None:
        self.payload.setdefault("skill_events", []).append(event)
        self.payload["updated_at"] = utc_now()
        self._write()

    def add_artifacts(self, artifacts: dict[str, str]) -> None:
        if not artifacts:
            return
        self.payload.setdefault("artifacts", {}).update(artifacts)
        self.payload["updated_at"] = utc_now()
        self._write()

    def finish_success(self, *, message: str = "", artifacts: dict[str, str] | None = None) -> None:
        if artifacts:
            self.payload.setdefault("artifacts", {}).update(artifacts)
        self.payload["status"] = "succeeded"
        self.payload["message"] = message
        self.payload["finished_at"] = utc_now()
        self.payload["updated_at"] = self.payload["finished_at"]
        self._write()

    def finish_failure(self, error: object) -> None:
        self.payload["status"] = "failed"
        self.payload["error"] = str(error)
        self.payload["finished_at"] = utc_now()
        self.payload["updated_at"] = self.payload["finished_at"]
        self._write()


class _SkillTrace:
    def __init__(self, agent: "WorkflowAgent", skill_id: str, inputs: dict[str, Any] | None = None) -> None:
        self.agent = agent
        self.spec = agent.registry.get(skill_id)
        self.inputs = inputs or {}
        self.started_at = ""
        self.start_time = 0.0
        self.artifacts: dict[str, str] = {}
        self.message = ""
        self.usage_delta: dict[str, Any] = {}

    def __enter__(self) -> "_SkillTrace":
        self.started_at = utc_now()
        self.start_time = time.monotonic()
        self.agent._emit(
            "agent_skill_start",
            f"Skill started: {self.spec.id}",
            {
                "type": "agent_skill_start",
                "skill_id": self.spec.id,
                "description": self.spec.description,
                "inputs": self.inputs,
            },
        )
        return self

    def set_result(
        self,
        *,
        artifacts: dict[str, str] | None = None,
        message: str = "",
        usage_delta: dict[str, Any] | None = None,
    ) -> None:
        if artifacts:
            self.artifacts.update(artifacts)
        if message:
            self.message = message
        if usage_delta:
            self.usage_delta.update(usage_delta)

    def __exit__(self, exc_type, exc, tb) -> bool:
        finished_at = utc_now()
        duration_ms = int((time.monotonic() - self.start_time) * 1000)
        if exc is not None:
            event = {
                "skill_id": self.spec.id,
                "description": self.spec.description,
                "status": SKILL_FAILED,
                "started_at": self.started_at,
                "finished_at": finished_at,
                "duration_ms": duration_ms,
                "inputs": self.inputs,
                "error": str(exc),
            }
            self.agent.run.append_skill_event(event)
            self.agent._emit(
                "agent_skill_error",
                f"Skill failed: {self.spec.id}",
                {
                    "type": "agent_skill_error",
                    "skill_id": self.spec.id,
                    "description": self.spec.description,
                    "error": str(exc),
                },
            )
            return False

        event = {
            "skill_id": self.spec.id,
            "description": self.spec.description,
            "status": SKILL_SUCCESS,
            "started_at": self.started_at,
            "finished_at": finished_at,
            "duration_ms": duration_ms,
            "inputs": self.inputs,
            "artifacts": self.artifacts,
            "usage_delta": self.usage_delta,
            "message": self.message,
        }
        self.agent.run.append_skill_event(event)
        self.agent.run.add_artifacts(self.artifacts)
        self.agent._emit(
            "agent_skill_done",
            f"Skill completed: {self.spec.id}",
            {
                "type": "agent_skill_done",
                "skill_id": self.spec.id,
                "description": self.spec.description,
                "artifacts": self.artifacts,
                "message": self.message,
            },
        )
        return False


class WorkflowAgent:
    def __init__(
        self,
        project_path: str | Path,
        workflow: str,
        *,
        workflow_id: str = "",
        progress_callback: Callable[[dict], None] | None = None,
        registry: SkillRegistry | None = None,
    ) -> None:
        self.registry = registry or default_skill_registry()
        self.run = AgentRun(project_path, workflow, workflow_id=workflow_id)
        self.progress_callback = progress_callback

    @property
    def workflow_id(self) -> str:
        return self.run.workflow_id

    def _emit(self, stage: str, message: str, event_details: dict[str, Any]) -> None:
        emit_progress(self.progress_callback, stage, message, event_details=event_details)

    @contextmanager
    def skill(self, skill_id: str, *, inputs: dict[str, Any] | None = None) -> Iterator[_SkillTrace]:
        trace = _SkillTrace(self, skill_id, inputs=inputs)
        with trace:
            yield trace

    def run_skill(
        self,
        skill_id: str,
        func: Callable[[], Any],
        *,
        inputs: dict[str, Any] | None = None,
        artifacts: dict[str, str] | None = None,
        message: str = "",
    ) -> SkillResult:
        with self.skill(skill_id, inputs=inputs) as trace:
            value = func()
            if isinstance(value, SkillResult):
                result = value
            else:
                result = SkillResult.ok(value=value, artifacts=artifacts or {}, message=message)
            if result.status != SKILL_SUCCESS:
                raise RuntimeError(result.error or result.message or f"skill failed: {skill_id}")
            trace.set_result(
                artifacts=result.artifacts or artifacts or {},
                message=result.message or message,
                usage_delta=result.usage_delta,
            )
            return result

    def finish_success(self, *, message: str = "", artifacts: dict[str, str] | None = None) -> None:
        self.run.finish_success(message=message, artifacts=artifacts)

    def finish_failure(self, error: object) -> None:
        self.run.finish_failure(error)

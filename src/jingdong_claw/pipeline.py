from __future__ import annotations

from collections.abc import Callable

from .artifacts import ArtifactWriter
from .models import DraftDocument, RunResult, ScoreResult
from .parser import parse_document_output
from .prompts import build_generation_messages, build_revision_messages
from .scoring import build_feedback_summary, build_round_result, evaluate_draft


def generate_initial_draft(
    client: object,
    *,
    topic: str,
    topic_draft_generator: Callable[[str], DraftDocument] | None = None,
) -> DraftDocument:
    if topic_draft_generator is not None:
        return topic_draft_generator(topic)
    response = client.generate_text(build_generation_messages(topic), temperature=0.5)
    return parse_document_output(response.text, fallback_title=f"Survey on {topic}")


def select_best_revision_candidate(
    client: object,
    *,
    draft: DraftDocument,
    scores: list[ScoreResult],
    feedback_summary: str,
    revision_candidates: int,
) -> tuple[DraftDocument, list[ScoreResult], str]:
    best_draft: DraftDocument | None = None
    best_scores: list[ScoreResult] | None = None
    best_feedback_summary: str | None = None
    best_total_score: int | None = None

    for candidate_index in range(1, revision_candidates + 1):
        response = client.generate_text(
            build_revision_messages(
                current_title=draft.title,
                current_body=draft.body,
                scores=scores,
                feedback_summary=feedback_summary,
                candidate_index=candidate_index,
                candidate_count=revision_candidates,
            ),
            temperature=0.4,
        )
        candidate_draft = parse_document_output(response.text, fallback_title=draft.title)
        candidate_scores = evaluate_draft(client, candidate_draft)
        candidate_total = sum(item.score for item in candidate_scores)
        candidate_feedback_summary = build_feedback_summary(candidate_scores)

        if best_total_score is None or candidate_total > best_total_score:
            best_draft = candidate_draft
            best_scores = candidate_scores
            best_feedback_summary = candidate_feedback_summary
            best_total_score = candidate_total

    if best_draft is None or best_scores is None or best_feedback_summary is None:
        raise RuntimeError("Revision candidate selection produced no draft.")

    return best_draft, best_scores, best_feedback_summary


class SelfEvolutionPipeline:
    def __init__(
        self,
        *,
        client: object,
        artifact_writer: ArtifactWriter,
        topic_draft_generator: Callable[[str], DraftDocument] | None = None,
        revision_candidates: int = 2,
    ) -> None:
        self.client = client
        self.artifact_writer = artifact_writer
        self.topic_draft_generator = topic_draft_generator
        self.revision_candidates = max(1, revision_candidates)

    def run_topic(self, *, topic: str, rounds: int) -> RunResult:
        initial_draft = self._generate_initial_draft(topic)
        return self._run_loop(
            mode="topic",
            topic=topic,
            initial_draft=initial_draft,
            rounds=rounds,
        )

    def run_paper(self, *, title: str, body: str, rounds: int) -> RunResult:
        return self._run_loop(
            mode="paper",
            topic=None,
            initial_draft=DraftDocument(title=title, body=body),
            rounds=rounds,
        )

    def _generate_initial_draft(self, topic: str) -> DraftDocument:
        return generate_initial_draft(
            self.client,
            topic=topic,
            topic_draft_generator=self.topic_draft_generator,
        )

    def _revise_draft(
        self,
        draft: DraftDocument,
        scores: list[ScoreResult],
        feedback_summary: str,
    ) -> tuple[DraftDocument, list[ScoreResult], str]:
        return select_best_revision_candidate(
            self.client,
            draft=draft,
            scores=scores,
            feedback_summary=feedback_summary,
            revision_candidates=self.revision_candidates,
        )

    def _run_loop(
        self,
        *,
        mode: str,
        topic: str | None,
        initial_draft: DraftDocument,
        rounds: int,
    ) -> RunResult:
        run_result = RunResult(
            run_id=self.artifact_writer.output_dir.name,
            mode=mode,
            topic=topic,
            output_dir=str(self.artifact_writer.output_dir),
        )
        current_draft = initial_draft
        cached_scores: list[ScoreResult] | None = None
        cached_feedback_summary: str | None = None

        try:
            for round_index in range(rounds):
                if cached_scores is None or cached_feedback_summary is None:
                    scores = evaluate_draft(self.client, current_draft)
                    feedback_summary = build_feedback_summary(scores)
                else:
                    scores = cached_scores
                    feedback_summary = cached_feedback_summary
                    cached_scores = None
                    cached_feedback_summary = None

                round_result = build_round_result(
                    round_index=round_index,
                    draft=current_draft,
                    scores=scores,
                    feedback_summary=feedback_summary,
                    provider=self.client.config.provider,
                    model=self.client.config.model,
                )
                run_result.rounds.append(round_result)
                self.artifact_writer.write_round(round_result)

                if run_result.best_round_index == -1 or round_result.total_score > run_result.best_total_score:
                    run_result.best_round_index = round_index
                    run_result.best_total_score = round_result.total_score

                if round_index == rounds - 1:
                    run_result.stop_reason = "max_rounds_reached"
                    break

                revised_draft, revised_scores, revised_feedback_summary = self._revise_draft(
                    current_draft,
                    scores,
                    feedback_summary,
                )
                revised_total_score = sum(item.score for item in revised_scores)
                if revised_total_score <= round_result.total_score:
                    run_result.stopped_early = True
                    run_result.stop_reason = "no_improving_revision_candidate"
                    break

                current_draft = revised_draft
                cached_scores = revised_scores
                cached_feedback_summary = revised_feedback_summary

            if run_result.stop_reason == "not_started":
                run_result.stop_reason = "completed"
            return run_result
        except Exception as exc:
            run_result.error = str(exc)
            run_result.stopped_early = True
            run_result.stop_reason = "error"
            raise
        finally:
            self.artifact_writer.write_summary(run_result)

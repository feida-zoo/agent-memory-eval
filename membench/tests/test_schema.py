"""data.schema 单元测试：load_dataset 校验 + 内置示例数据集质量。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from membench.data import SAMPLE_DATASET_PATH, load_dataset
from membench.data.schema import ABSTAIN_SENTINEL, CATEGORIES, Dataset, Dialogue, Question


class TestSampleDataset:
    """内置示例数据集必须满足 SPEC 第 5 节的规模与覆盖要求。"""

    @classmethod
    @pytest.fixture(scope="class")
    def dialogues(cls) -> list[Dialogue]:
        return load_dataset(SAMPLE_DATASET_PATH)

    def test_scale(self, dialogues: list[Dialogue]) -> None:
        questions = [q for d in dialogues for q in d.questions]
        assert len(dialogues) >= 6
        assert len(questions) >= 24

    def test_category_coverage(self, dialogues: list[Dialogue]) -> None:
        covered = {q.category for d in dialogues for q in d.questions}
        assert covered == set(CATEGORIES)

    def test_knowledge_update_and_abstention_counts(
        self, dialogues: list[Dialogue]
    ) -> None:
        questions = [q for d in dialogues for q in d.questions]
        ku = [q for q in questions if q.category == "knowledge_update"]
        ab = [q for q in questions if q.category == "abstention"]
        assert len(ku) >= 4
        assert len(ab) >= 3
        # knowledge_update 题必须有 gold_stale，abstention 题 gold 为哨兵值。
        assert all(q.gold_stale for q in ku)
        assert all(q.gold_answer == ABSTAIN_SENTINEL for q in ab)

    def test_session_shape(self, dialogues: list[Dialogue]) -> None:
        for d in dialogues:
            assert 2 <= len(d.sessions) <= 3
            for session in d.sessions:
                assert 4 <= len(session) <= 8
                for msg in session:
                    assert msg["role"] in ("user", "assistant")
                    assert isinstance(msg["content"], str) and msg["content"]

    def test_evidence_is_substring_of_dialogue(
        self, dialogues: list[Dialogue]
    ) -> None:
        """每个 evidence 都必须是该 dialogue 中某条消息原文的子串。"""
        for d in dialogues:
            corpus = [m["content"] for s in d.sessions for m in s]
            for q in d.questions:
                for ev in q.evidence:
                    assert any(ev in content for content in corpus), (
                        f"{q.qid} 的 evidence 不是对话原文子串: {ev!r}"
                    )

    def test_gold_stale_is_old_fact(self, dialogues: list[Dialogue]) -> None:
        """knowledge_update 题的 gold_stale 应能在对话原文中找到（旧值）。"""
        for d in dialogues:
            corpus = [m["content"] for s in d.sessions for m in s]
            for q in d.questions:
                if q.gold_stale:
                    assert any(q.gold_stale in c for c in corpus), (
                        f"{q.qid} 的 gold_stale 未出现在对话中: {q.gold_stale!r}"
                    )


class TestLoadDatasetValidation:
    """字段校验：缺字段 / 非法取值时必须抛出带行号的 ValueError。"""

    def _write(self, tmp_path: Path, lines: list[str]) -> Path:
        path = tmp_path / "ds.jsonl"
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    def _valid_dialogue(self) -> dict:
        return {
            "dialogue_id": "d1",
            "sessions": [[{"role": "user", "content": "你好", "timestamp": None}]],
            "questions": [
                {
                    "qid": "d1_q1",
                    "question": "问？",
                    "gold_answer": "答",
                    "category": "single_hop",
                    "evidence": ["你好"],
                }
            ],
        }

    def test_valid_file(self, tmp_path: Path) -> None:
        path = self._write(tmp_path, [json.dumps(self._valid_dialogue())])
        dialogues = load_dataset(path)
        assert len(dialogues) == 1
        assert dialogues[0].dialogue_id == "d1"
        assert dialogues[0].questions[0].gold_stale is None

    def test_missing_dialogue_field_reports_lineno(self, tmp_path: Path) -> None:
        bad = self._valid_dialogue()
        del bad["sessions"]
        lines = [json.dumps(self._valid_dialogue()), json.dumps(bad)]
        path = self._write(tmp_path, lines)
        with pytest.raises(ValueError, match=r":2:.*sessions"):
            load_dataset(path)

    def test_missing_question_field_reports_lineno(self, tmp_path: Path) -> None:
        bad = self._valid_dialogue()
        del bad["questions"][0]["gold_answer"]
        path = self._write(tmp_path, [json.dumps(bad)])
        with pytest.raises(ValueError, match=r":1:.*gold_answer"):
            load_dataset(path)

    def test_invalid_category(self, tmp_path: Path) -> None:
        bad = self._valid_dialogue()
        bad["questions"][0]["category"] = "not_a_category"
        path = self._write(tmp_path, [json.dumps(bad)])
        with pytest.raises(ValueError, match=r":1:.*category"):
            load_dataset(path)

    def test_invalid_json_line(self, tmp_path: Path) -> None:
        path = self._write(tmp_path, ["{not json"])
        with pytest.raises(ValueError, match=r":1:"):
            load_dataset(path)

    def test_invalid_role(self, tmp_path: Path) -> None:
        bad = self._valid_dialogue()
        bad["sessions"][0][0]["role"] = "system"
        path = self._write(tmp_path, [json.dumps(bad)])
        with pytest.raises(ValueError, match=r":1:.*role"):
            load_dataset(path)

    def test_blank_lines_skipped(self, tmp_path: Path) -> None:
        lines = [json.dumps(self._valid_dialogue()), "", "   "]
        path = self._write(tmp_path, lines)
        assert len(load_dataset(path)) == 1

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_dataset(tmp_path / "nope.jsonl")


class TestIgnoreFields:
    """题目级忽略标记（ignore / ignore_reason）解析与校验。"""

    def _write(self, tmp_path: Path, question_overrides: dict) -> Path:
        dialogue = {
            "dialogue_id": "d1",
            "sessions": [[{"role": "user", "content": "你好", "timestamp": None}]],
            "questions": [
                {
                    "qid": "d1_q1",
                    "question": "问？",
                    "gold_answer": "答",
                    "category": "single_hop",
                    "evidence": [],
                    **question_overrides,
                }
            ],
        }
        path = tmp_path / "ds.jsonl"
        path.write_text(json.dumps(dialogue, ensure_ascii=False), encoding="utf-8")
        return path

    def test_ignore_fields_parsed(self, tmp_path: Path) -> None:
        path = self._write(tmp_path, {"ignore": True, "ignore_reason": "场景不适用"})
        q = load_dataset(path)[0].questions[0]
        assert q.ignore is True
        assert q.ignore_reason == "场景不适用"

    def test_missing_ignore_fields_backward_compatible(self, tmp_path: Path) -> None:
        """旧数据集无 ignore 字段：照常加载，默认 ignore=False / reason=None。"""
        path = self._write(tmp_path, {})
        q = load_dataset(path)[0].questions[0]
        assert q.ignore is False
        assert q.ignore_reason is None

    def test_ignore_must_be_bool(self, tmp_path: Path) -> None:
        path = self._write(tmp_path, {"ignore": "true"})
        with pytest.raises(ValueError, match=r":1:.*ignore"):
            load_dataset(path)

    def test_ignore_reason_must_be_str_or_null(self, tmp_path: Path) -> None:
        path = self._write(tmp_path, {"ignore": True, "ignore_reason": 123})
        with pytest.raises(ValueError, match=r":1:.*ignore_reason"):
            load_dataset(path)


class TestDataclasses:
    """Dialogue / Question / Dataset 数据类基本行为。"""

    def test_dataset_container(self) -> None:
        q = Question(
            qid="q1",
            question="问",
            gold_answer="答",
            category="single_hop",
            evidence=[],
        )
        ds = Dataset(
            name="toy",
            dialogues=[Dialogue(dialogue_id="d1", sessions=[], questions=[q])],
        )
        assert ds.n_questions == 1
        assert ds.dialogues[0].questions[0].qid == "q1"

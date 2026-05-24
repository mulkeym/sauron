"""Tests for LightRAG entity-extraction output repair and graph-skip rules.

These guard two ingestion bugs observed in production logs:
  1. The local LLM (gemma) intermittently omits the entity_type field, producing
     3-field entity records that LightRAG's parser silently drops
     (operate.py: "LLM output format error; found 3/4 fields on ENTITY").
  2. Numeric pay-rate spreadsheets have no entities, so KG extraction on them
     only burns LLM time and triggers worker timeouts.
"""
from src.knowledge.kg_format import (
    TUPLE_DELIMITER as D,
    repair_extraction_format,
    should_skip_graph,
)


def _entity(*fields: str) -> str:
    return D.join(fields)


class TestRepairEntityRecords:
    def test_inserts_missing_type_on_3_field_entity(self):
        # gemma dropped the type field: entity<|#|>name<|#|>description
        line = _entity("entity", "Family and Medical Leave Act (FMLA)",
                       "A 1993 law granting up to 12 weeks of unpaid leave.")
        out = repair_extraction_format(line)
        parts = out.split(D)
        assert len(parts) == 4
        assert parts[0] == "entity"
        assert parts[1] == "Family and Medical Leave Act (FMLA)"
        assert parts[3].startswith("A 1993 law")
        # inserted type must be a clean single-token type (no spaces/forbidden chars)
        assert parts[2] and " " not in parts[2]

    def test_leaves_valid_4_field_entity_untouched(self):
        line = _entity("entity", "Tokyo", "location", "Host city of the championship.")
        assert repair_extraction_format(line) == line

    def test_does_not_touch_2_field_entity(self):
        # too broken to repair safely
        line = _entity("entity", "OrphanName")
        assert repair_extraction_format(line) == line

    def test_inserts_missing_keywords_on_4_field_relation(self):
        # relation<|#|>src<|#|>tgt<|#|>description  (keywords field dropped)
        line = _entity("relation", "Donated Annual Leave", "Medical Emergency",
                       "Leave may be retroactively substituted for a medical emergency.")
        out = repair_extraction_format(line)
        parts = out.split(D)
        assert len(parts) == 5
        assert parts[0] == "relation"
        assert parts[1] == "Donated Annual Leave"
        assert parts[2] == "Medical Emergency"
        assert parts[4].startswith("Leave may be")

    def test_leaves_valid_5_field_relation_untouched(self):
        line = _entity("relation", "A", "B", "kw1, kw2", "Some description.")
        assert repair_extraction_format(line) == line

    def test_repairs_within_multiline_block_preserving_others(self):
        block = "\n".join([
            _entity("entity", "Alex", "person", "A character."),          # valid
            _entity("entity", "Sick Leave", "A type of federal leave."),  # 3-field
            "<|COMPLETE|>",
        ])
        out = repair_extraction_format(block)
        lines = out.splitlines()
        assert lines[0].split(D) == ["entity", "Alex", "person", "A character."]
        assert len(lines[1].split(D)) == 4
        assert lines[2] == "<|COMPLETE|>"

    def test_ignores_non_record_lines(self):
        text = "<Output>\nsome prose without delimiters\n"
        assert repair_extraction_format(text) == text


class TestShouldSkipGraph:
    def test_skips_spreadsheets(self):
        for name in ["2016-law-enforcement-officer-pay-rates.xls",
                     "rates.XLSX", "data.xlsm", "table.csv"]:
            assert should_skip_graph(name) is True, name

    def test_does_not_skip_prose(self):
        for name in ["handbook-on-leave.pdf", "memo.docx", "notes.txt", "page.html"]:
            assert should_skip_graph(name) is False, name

    def test_handles_empty_or_pathless(self):
        assert should_skip_graph("") is False
        assert should_skip_graph("/a/b/c/2020-pay.xls") is True

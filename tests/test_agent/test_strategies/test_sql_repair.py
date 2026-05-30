import json
import duckdb
from src.agent.strategies import structured as S
from src.db.schema_registry import TableSchema, ColumnSchema


def test_classify_empty():
    assert S._classify_sql_result([]) == "empty"


def test_classify_degenerate_all_null():
    rows = [{"a": None, "b": None}, {"a": None, "b": None}]
    assert S._classify_sql_result(rows) == "degenerate"


def test_classify_satisfactory():
    rows = [{"locality": "Tampa", "salary": 86415}]
    assert S._classify_sql_result(rows) == "satisfactory"


def test_classify_too_large(monkeypatch):
    from src.config import settings
    monkeypatch.setattr(settings, "sql_result_budget_chars", 50)
    monkeypatch.setattr(settings, "llm_max_context", 200000)
    rows = [{"locality": f"loc{i}", "salary": i} for i in range(100)]
    assert S._classify_sql_result(rows) == "too_large"


def test_feedback_too_large_mentions_aggregate():
    rows = [{"a": i} for i in range(885)]
    fb = S._repair_feedback("too_large", rows=rows, sql="SELECT * FROM t", question="q")
    assert "885" in fb
    assert "aggregate" in fb.lower()
    assert "SELECT * FROM t" in fb


def test_feedback_empty_mentions_filter():
    fb = S._repair_feedback("empty", rows=[], sql="SELECT * FROM t WHERE x='z'", question="q")
    assert "no rows" in fb.lower()
    assert "filter" in fb.lower()


def test_feedback_error_includes_error():
    fb = S._repair_feedback("error", rows=[], sql="SELEKT", question="q", error="syntax error")
    assert "syntax error" in fb


def test_feedback_includes_judge_reason():
    rows = [{"a": i} for i in range(885)]
    fb = S._repair_feedback("too_large", rows=rows, sql="SELECT * FROM t",
                            question="q", judge_reason="wrong column")
    assert "wrong column" in fb


def _schema(table, ncols):
    return TableSchema(database="tab", table=table,
                       columns=[ColumnSchema(name=f"c{i}", dtype="DOUBLE") for i in range(ncols)])


def test_wide_table_gate_fires_for_big_table(monkeypatch):
    from src.config import settings
    monkeypatch.setattr(settings, "sql_wide_table_cell_threshold", 100)
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE big AS SELECT range AS c0, range AS c1 FROM range(60)")  # 60 rows
    schema = _schema("big", 2)  # 60*2 = 120 > 100
    block = S._wide_table_steering(con, [schema])
    assert "big" in block
    assert "aggregat" in block.lower()


def test_wide_table_gate_silent_for_small_table(monkeypatch):
    from src.config import settings
    monkeypatch.setattr(settings, "sql_wide_table_cell_threshold", 100000)
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE small AS SELECT range AS c0 FROM range(3)")
    assert S._wide_table_steering(con, [_schema("small", 1)]) == ""


def test_prompt_no_longer_prefers_select_star():
    assert "prefer `SELECT *`" not in S.TEXT_TO_SQL_PROMPT
    assert "narrowest set of columns" in S.TEXT_TO_SQL_PROMPT


def test_relevance_judge_parses_unhelpful():
    def fake_gen(system_prompt, user_prompt, **kw):
        return '{"helpful": false, "reason": "rows are about grades not localities"}'
    helpful, reason = S._relevance_judge(fake_gen, "what are pay rates by locality?",
                                         [{"grade": "GS-12"}])
    assert helpful is False
    assert "localities" in reason


def test_relevance_judge_defaults_helpful_on_bad_json():
    def fake_gen(system_prompt, user_prompt, **kw):
        return "not json at all"
    helpful, reason = S._relevance_judge(fake_gen, "q", [{"a": 1}])
    assert helpful is True
    assert reason == ""


def test_generate_sql_passes_extra_context_and_temp():
    seen = {}
    def fake_gen(system_prompt, user_prompt, temperature=0.1, max_tokens=2048):
        seen["user"] = user_prompt
        seen["temp"] = temperature
        return "SELECT 1"
    sql = S.generate_sql("SCHEMA", "my question", generate_fn=fake_gen,
                         extra_user_context="\nNOTE: aggregate please", temperature=0.3)
    assert sql == "SELECT 1"
    assert "my question" in seen["user"]
    assert "aggregate please" in seen["user"]
    assert seen["temp"] == 0.3


def _make_con_with_pay():
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE pay AS SELECT * FROM (VALUES "
                "('Tampa','GS-12',86415),('Boston','GS-12',92000),('Denver','GS-13',99000)) "
                "AS t(locality, grade, salary)")
    return con


def test_fit_clean_result_one_generation():
    calls = []
    def fake_gen(system_prompt, user_prompt, temperature=0.0, max_tokens=2048):
        calls.append(user_prompt)
        return "SELECT * FROM pay"
    con = _make_con_with_pay()
    res = S._generate_run_fit(con, "what are the pay rates?", [_schema("pay", 3)],
                              generate_fn=fake_gen)
    assert res.verdict == "satisfactory"
    assert res.attempts == 1
    assert len(res.rows) == 3
    assert len(calls) == 1  # no retry, no judge


def test_fit_retries_when_first_is_empty(monkeypatch):
    monkeypatch.setattr("src.config.settings.sql_relevance_judge_enabled", False)
    seq = iter(["SELECT * FROM pay WHERE locality='nowhere'",  # empty
                "SELECT * FROM pay"])                          # good
    def fake_gen(system_prompt, user_prompt, temperature=0.0, max_tokens=2048):
        return next(seq)
    con = _make_con_with_pay()
    res = S._generate_run_fit(con, "pay rates?", [_schema("pay", 3)], generate_fn=fake_gen)
    assert res.verdict == "satisfactory"
    assert res.attempts == 2
    assert len(res.rows) == 3


def test_fit_exhausts_and_returns_best_valid(monkeypatch):
    monkeypatch.setattr("src.config.settings.sql_relevance_judge_enabled", False)
    monkeypatch.setattr("src.config.settings.sql_result_budget_chars", 10)  # force too_large
    monkeypatch.setattr("src.config.settings.sql_repair_max_retries", 2)
    def fake_gen(system_prompt, user_prompt, temperature=0.0, max_tokens=2048):
        return "SELECT * FROM pay"  # always too_large under budget=10
    con = _make_con_with_pay()
    res = S._generate_run_fit(con, "pay rates?", [_schema("pay", 3)], generate_fn=fake_gen)
    assert res.verdict == "too_large"
    assert res.attempts == 3  # orig + 2 retries
    assert len(res.rows) == 3  # best valid result kept, not discarded


def test_fit_raises_when_all_attempts_error(monkeypatch):
    monkeypatch.setattr("src.config.settings.sql_relevance_judge_enabled", False)
    def fake_gen(system_prompt, user_prompt, temperature=0.0, max_tokens=2048):
        return "SELEKT bad sql"  # raises in run_sql
    con = _make_con_with_pay()
    import pytest
    with pytest.raises(Exception):
        S._generate_run_fit(con, "q", [_schema("pay", 3)], generate_fn=fake_gen)


def test_run_structured_lookup_uses_loop(monkeypatch):
    monkeypatch.setattr("src.config.settings.sql_relevance_judge_enabled", False)
    # connect_tabular returns our in-memory con with a pay table
    con = _make_con_with_pay()
    monkeypatch.setattr("src.ingestion.tabular_store.connect_tabular",
                        lambda read_only=False: con)
    seq = iter(["SELECT * FROM pay WHERE locality='nope'", "SELECT * FROM pay"])
    def fake_gen(system_prompt, user_prompt, temperature=0.0, max_tokens=2048):
        return next(seq)
    trace = S.run_structured_lookup("pay rates?", [_schema("pay", 3)], "analytical",
                                    generate_fn=fake_gen)
    assert trace.status == "ran"
    assert trace.row_count == 3      # recovered after the empty first attempt


def test_structured_sql_rows_uses_loop(monkeypatch):
    monkeypatch.setattr("src.config.settings.sql_relevance_judge_enabled", False)
    con = _make_con_with_pay()
    monkeypatch.setattr("src.ingestion.tabular_store.connect_tabular",
                        lambda read_only=False: con)
    seq = iter(["SELECT * FROM pay WHERE grade='none'", "SELECT * FROM pay"])
    def fake_gen(system_prompt, user_prompt, temperature=0.0, max_tokens=2048):
        return next(seq)
    rows = S.structured_sql_rows("pay rates?", [_schema("pay", 3)], generate_fn=fake_gen)
    assert len(rows) == 3  # recovered after empty first attempt


def test_loop_passes_judge_reason_into_retry(monkeypatch):
    monkeypatch.setattr("src.config.settings.sql_relevance_judge_enabled", True)
    monkeypatch.setattr("src.config.settings.sql_result_budget_chars", 10)  # force too_large
    monkeypatch.setattr("src.config.settings.sql_repair_max_retries", 1)
    sql_prompts = []
    def fake_gen(system_prompt, user_prompt, temperature=0.0, max_tokens=2048):
        if system_prompt == S._JUDGE_PROMPT:
            return '{"helpful": false, "reason": "wrong entity"}'
        sql_prompts.append(user_prompt)
        return "SELECT * FROM pay"
    con = _make_con_with_pay()
    S._generate_run_fit(con, "pay rates?", [_schema("pay", 3)], generate_fn=fake_gen)
    # the second SQL-generation prompt must carry the judge's reason as feedback
    assert any("wrong entity" in p for p in sql_prompts[1:])

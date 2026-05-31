from src.admin.routes import _apply_settings_update
from src.config import settings


def test_apply_settings_update_is_partial():
    settings.admin_username = "ADMIN_KEEP"
    settings.llm_max_context = 250000
    settings.vllm_model_name = "OLD"
    persist = _apply_settings_update({"vllm_model_name": "NEW", "vllm_base_url": "http://x"})
    assert settings.vllm_model_name == "NEW"          # submitted -> updated
    assert settings.admin_username == "ADMIN_KEEP"     # absent -> untouched
    assert settings.llm_max_context == 250000          # absent -> untouched
    assert persist["vllm_model_name"] == "NEW"
    assert persist["admin_username"] == "ADMIN_KEEP"
    assert persist["llm_max_context"] == 250000


def test_apply_settings_update_bool_roundtrip():
    settings.feedback_enabled = True
    _apply_settings_update({"feedback_enabled": "false"})   # unchecked box posts hidden "false"
    assert settings.feedback_enabled is False
    _apply_settings_update({"feedback_enabled": "true"})
    assert settings.feedback_enabled is True


def test_apply_settings_update_keeps_blank_credential():
    settings.admin_password = "secret"
    _apply_settings_update({"admin_password": ""})           # blank submit must not clear it
    assert settings.admin_password == "secret"


def test_apply_settings_update_casts_numbers():
    _apply_settings_update({"llm_max_context": "300000", "feedback_similarity_threshold": "0.7"})
    assert settings.llm_max_context == 300000 and isinstance(settings.llm_max_context, int)
    assert settings.feedback_similarity_threshold == 0.7

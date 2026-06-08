import json

import pytest

from kindred.config.templates import Template, TemplateManager
from kindred.core.mechanism_source import MechanismAuthoringSource


@pytest.mark.unit
def test_template_round_trip_preserves_complete_mechanism_source(tmp_path, monkeypatch):
    monkeypatch.setenv("KINDRED_TEMPLATES_DIR", str(tmp_path))
    source = MechanismAuthoringSource.from_parts(
        reactions_text="reaction: TemplateA -> TemplateB; k=0.2",
        state_network_dsl="\n".join(
            [
                "state: T_A, kind=GS, energy=0, energy_unit=kJ/mol, degeneracy=1",
                "state: T_TS, kind=TS, energy=10, energy_unit=kJ/mol, degeneracy=1",
                "edge: T_A,T_TS",
            ]
        ),
    )

    manager = TemplateManager()
    created = manager.create_template(
        "Complete Source",
        source=source,
        description="round trip",
        category="Mechanism Source",
        tags=["complete-source"],
    )

    reloaded_manager = TemplateManager()
    reloaded_manager.load_templates()
    reloaded = reloaded_manager.get_template(created.id)

    assert reloaded is not None
    assert reloaded.source == source


@pytest.mark.unit
def test_current_template_payload_rejects_non_string_source_fields(tmp_path):
    payload = {
        "id": "bad_source",
        "name": "Bad Source",
        "description": "",
        "category": "Mechanism Source",
        "tags": [],
        "reactions_text": ["reaction: A -> B; k=1"],
        "state_network_dsl": "",
        "created_at": "2026-05-25T00:00:00",
        "modified_at": "2026-05-25T00:00:00",
        "author": None,
    }
    payload_path = tmp_path / "bad_source.json"
    payload_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(TypeError, match="reactions_text"):
        Template.from_dict(json.loads(payload_path.read_text(encoding="utf-8")))

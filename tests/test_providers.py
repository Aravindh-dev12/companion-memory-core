from types import SimpleNamespace

from companion_memory.models import Event, MemoryExtraction, Speaker
from companion_memory.persona import Persona
from companion_memory.providers import OpenAIProvider


class FakeResponses:
    def __init__(self):
        self.parse_calls = []
        self.create_calls = []

    def parse(self, **kwargs):
        self.parse_calls.append(kwargs)
        schema = kwargs["text_format"]
        if schema is MemoryExtraction:
            parsed = MemoryExtraction(candidates=[])
        else:
            parsed = schema.model_validate({"consistent": True})
        return SimpleNamespace(output_parsed=parsed, incomplete_details=None)

    def create(self, **kwargs):
        self.create_calls.append(kwargs)
        return SimpleNamespace(output_text="A grounded response.")


class FakeClient:
    def __init__(self):
        self.responses = FakeResponses()


def _provider() -> OpenAIProvider:
    provider = object.__new__(OpenAIProvider)
    provider.client = FakeClient()
    provider.extraction_model = "extract-model"
    provider.resolution_model = "resolve-model"
    provider.response_model = "chat-model"
    provider.verification_model = "verify-model"
    return provider


def test_openai_structured_extraction_disables_provider_storage():
    provider = _provider()
    event = Event(session_id="s", turn_id=1, speaker=Speaker.USER, text="I like coffee")
    provider.extract_user_memories(event)
    call = provider.client.responses.parse_calls[-1]
    assert call["store"] is False
    assert call["text_format"] is MemoryExtraction
    assert call["model"] == "extract-model"


def test_openai_response_generation_disables_provider_storage():
    provider = _provider()
    persona = Persona(name="Mira", role="companion")
    response = provider.generate_response(
        user_text="hello",
        persona=persona,
        recent_events=[],
        retrieved=[],
    )
    call = provider.client.responses.create_calls[-1]
    assert response == "A grounded response."
    assert call["store"] is False
    assert call["model"] == "chat-model"

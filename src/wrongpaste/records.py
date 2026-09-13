from dataclasses import asdict, dataclass, field


@dataclass
class ConversationRecord:
    model_id: str
    topic_id: str
    artifact_id: str
    artifact_kind: str
    artifact_entities: list[str]
    similarity: float
    n_turns: int
    seed: int
    user_model: str
    max_tokens: int
    transcript: list[dict]
    reaction: str
    usages: list[dict] = field(default_factory=list)

    def to_json(self) -> dict:
        return asdict(self)

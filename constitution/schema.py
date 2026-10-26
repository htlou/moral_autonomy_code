"""Constitution entry schema."""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional
import uuid
import json


@dataclass
class ConstitutionEntry:
    principle: str
    confidence: float = 0.5
    evidence: list[str] = field(default_factory=list)
    version: int = 1
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    supersedes: Optional[str] = None
    active: bool = True

    def to_document(self) -> str:
        """Convert to a text document for LightRAG insertion."""
        meta = {
            "id": self.id,
            "version": self.version,
            "confidence": self.confidence,
            "evidence_count": len(self.evidence),
            "supersedes": self.supersedes,
            "active": self.active,
            "created_at": self.created_at,
        }
        return (
            f"[Constitution Principle {self.id} v{self.version}]\n"
            f"{self.principle}\n"
            f"[Metadata: {json.dumps(meta)}]"
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ConstitutionEntry":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

from typing import Literal
from pydantic import BaseModel, Field


AveritecLabel = Literal[
    "Supported",
    "Refuted",
    "Not Enough Evidence",
    "Conflicting Evidence/Cherrypicking",
]


class VerificationOutput(BaseModel):
    label: AveritecLabel = Field(
        description="The predicted verification label."
    )
    reason: str = Field(
        description="The reason or explanation for the predicted label from LLM."
    )
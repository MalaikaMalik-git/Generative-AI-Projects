"""
agent/models.py
Dataclasses used throughout the ReAct loop.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum


class StepType(str, Enum):
    THOUGHT    = "Thought"
    ACTION     = "Action"
    OBSERVATION = "Observation"
    ANSWER     = "Answer"


@dataclass
class Step:
    """One entry in the agent trace."""
    type:    StepType
    content: str


@dataclass
class AgentResult:
    """Final output returned to the caller."""
    question:   str
    answer:     str
    steps:      list[Step]        = field(default_factory=list)
    sources:    list[dict]        = field(default_factory=list)   # filled in Session 4
    success:    bool              = True
    error:      str | None        = None
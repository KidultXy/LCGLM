from typing import Literal, Optional
from dataclasses import dataclass, field


@dataclass
class GeneralArguments:
    """
    Arguments pertaining to which techniques we are going to fine-tuning with.
    """
    stage: Optional[Literal["sft"]] = field(
        default="sft",
        metadata={"help": "Which stage will be performed in training."}
    )

"""Adapt official LiveMathematicianBench monthly files to the shared Data API."""

from hashlib import sha256
from random import Random
from string import ascii_uppercase

from pydantic import TypeAdapter

from ...data import Data, DataSample
from .config import LiveMathDataConfig
from .typing import Choice, LiveMathRewardData, LiveMathTask, SourceRecord


@Data.register("livemath")
class LiveMathData(Data[LiveMathDataConfig]):
    """Load official raw JSON; use Data.split for train/validation/test membership."""

    config_type = LiveMathDataConfig
    """Data source and choice-shuffle settings."""

    def _load(self) -> list[DataSample]:
        """Complete and shuffle choices while keeping the answer key private."""
        source = self.config.data_path.expanduser().resolve()
        files = [source] if source.is_file() else sorted(source.rglob("qa_*_final.json"))
        if not files:
            raise FileNotFoundError(f"No LiveMath monthly JSON files found at {source}")
        samples = []
        for path in files:
            content = path.read_bytes()
            source_hash = sha256(content).hexdigest()
            records = TypeAdapter(list[SourceRecord]).validate_json(content)
            for record in records:
                identifier = f"{record.month}:{record.no}"
                choices = [Choice(label=choice.label.strip(), text=choice.text.strip()) for choice in record.mcq.choices]
                correct = Choice(label=record.mcq.correct_choice.label.strip(), text=record.mcq.correct_choice.text.strip())
                if correct.label not in {choice.label for choice in choices}:
                    choices.append(correct)
                    choices.sort(key=lambda choice: choice.label)
                if self.config.choice_seed is not None:
                    digest = sha256(f"{self.config.choice_seed}:{identifier}".encode()).hexdigest()
                    Random(int(digest[:16], 16)).shuffle(choices)
                    correct_label = correct.label
                    remapped = []
                    for label, choice in zip(ascii_uppercase, choices):
                        option = Choice(label=label, text=choice.text)
                        remapped.append(option)
                        if choice.label == correct_label:
                            correct = option
                    choices = remapped
                samples.append(
                    DataSample(
                        task=LiveMathTask(
                            id=identifier, instruction=record.mcq.question.strip(), choices=choices,
                            task_type=record.theorem_type[0] if record.theorem_type else "math_mcq",
                        ),
                        reward_data=LiveMathRewardData(
                            correct_choice=correct,
                            source_file=str(path),
                            source_sha256=source_hash,
                            choice_seed=self.config.choice_seed,
                        ),
                    )
                )
        if not samples:
            raise ValueError("The LiveMath source contains no samples")
        return samples

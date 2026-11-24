"""Load OfficeQA questions with an explicit public document corpus."""

import csv
import json

from ...data import Data, DataSample
from ..common import load_records
from .config import OfficeQADataConfig
from .documents import oracle_pages
from .typing import OfficeQARewardData, OfficeQATask


def _list_field(value: str | list[str] | None) -> list[str]:
    """Read released list fields from CSV or JSON."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = value.strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, list):
        return _list_field(parsed)
    parts = (
        text.splitlines()
        if "\n" in text
        else (text.split(",") if "," in text and not text.lower().endswith(".txt") else [text])
    )
    return [part.strip() for part in parts if part.strip()]


@Data.register("officeqa")
class OfficeQAData(Data[OfficeQADataConfig]):
    """Load official questions without requiring prebuilt split directories."""

    config_type = OfficeQADataConfig
    """Question and document locations."""

    def _load(self) -> list[DataSample]:
        """Load questions, references, and public text document paths."""
        source = self.config.data_path.expanduser().resolve()
        root = self.config.docs_root.expanduser().resolve()
        documents = {
            path.relative_to(root).as_posix(): path.resolve()
            for path in sorted(root.rglob("*.txt"))
            if path.is_file() and path.resolve().is_relative_to(root)
        }
        if not root.is_dir():
            raise FileNotFoundError(root)
        if source.suffix.lower() == ".csv":
            with source.open(encoding="utf-8", newline="") as handle:
                records = list(csv.DictReader(handle))
        else:
            records = load_records(source)
        samples = []
        for row in records:
            files = _list_field(row.get("source_files"))
            urls = _list_field(row.get("source_docs"))
            context = oracle_pages(root, files, urls)
            if not documents and not context:
                raise FileNotFoundError(f"No public OfficeQA documents or parsed source pages under {root}")
            samples.append(
                DataSample(
                    task=OfficeQATask(
                        id=str(row.get("uid") or row.get("id") or "").strip(),
                        instruction=str(row["question"]).strip(),
                        documents=documents,
                        task_type=str(row.get("task_type", "officeqa")),
                        source_files=files,
                        source_docs=urls,
                        oracle_context=context,
                    ),
                    reward_data=OfficeQARewardData(
                        ground_truth=str(row.get("ground_truth") or row.get("answer") or "").strip(),
                    ),
                )
            )
        if not samples:
            raise ValueError("The OfficeQA source contains no samples")
        return samples

"""Discover official workbook cases without importing legacy split layouts."""

from pathlib import Path

from ...data import Data, DataSample
from ..common import load_records
from .config import SpreadsheetBenchDataConfig
from .typing import SpreadsheetBenchRewardData, SpreadsheetBenchTask, WorkbookInput, WorkbookReference


def _case_paths(directory: Path) -> list[tuple[str, Path, Path]]:
    """Find the unique init/golden workbook pair in a Verified 400 task."""
    inputs = sorted(directory.glob("*init*.xlsx"))
    references = sorted(directory.glob("*golden*.xlsx"))
    if len(inputs) != 1 or len(references) != 1:
        raise ValueError(
            f"Expected one init and one golden workbook in {directory}, "
            f"found {len(inputs)} init and {len(references)} golden"
        )
    return [("1", inputs[0], references[0])]


def load_samples(config: SpreadsheetBenchDataConfig) -> list[DataSample]:
    """Load public workbook inputs and private evaluation references.

    Args:
        config: Metadata file and workbook root.

    Returns:
        Typed samples compatible with shared random and ID-file splitting.
    """
    source = config.data_path.expanduser().resolve()
    root = (config.data_root or source.parent).expanduser().resolve()
    samples = []
    for row in load_records(source):
        identifier = str(row["id"])
        directory = (
            root / str(row.get("spreadsheet_path") or row.get("task_dir") or f"spreadsheet/{identifier}")
        ).resolve()
        cases = _case_paths(directory)
        position = str(row.get("answer_position") or "")
        if position and row.get("answer_sheet") and "," not in row["answer_sheet"] and "!" not in position:
            position = f"{row['answer_sheet']}!{position}"
        samples.append(
            DataSample(
                task=SpreadsheetBenchTask(
                    id=identifier,
                    instruction=row["instruction"],
                    instruction_type=str(row.get("instruction_type") or ""),
                    answer_position=position,
                    cases=[WorkbookInput(number=number, path=path) for number, path, _ in cases],
                ),
                reward_data=SpreadsheetBenchRewardData(
                    answer_position=position,
                    answer_sheet=row.get("answer_sheet"),
                    cases=[WorkbookReference(number=number, path=path) for number, _, path in cases],
                ),
            )
        )
    return samples


@Data.register("spreadsheetbench")
class SpreadsheetBenchData(Data[SpreadsheetBenchDataConfig]):
    """Load Verified 400 workbook pairs for cached-value scoring."""

    config_type = SpreadsheetBenchDataConfig
    """Released metadata and workbook locations."""

    def _load(self) -> list[DataSample]:
        """Build single-case samples."""
        return load_samples(self.config)

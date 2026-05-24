"""
CSV utilities for data export, reading, and management.
Handles CSV appending, deduplication, and transformation.
"""

import pandas as pd
import polars as pl
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from utils.logging import get_logger

logger = get_logger(__name__)


class CSVManager:
    """Manage CSV files with deduplication and append logic."""

    @staticmethod
    def ensure_directory(file_path: Union[str, Path]):
        """Ensure parent directory exists."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def read_csv(file_path: Union[str, Path], 
                 engine: str = "pandas") -> Union[pd.DataFrame, pl.DataFrame]:
        """
        Read CSV file.

        Args:
            file_path: Path to CSV file
            engine: "pandas" or "polars"

        Returns:
            DataFrame (pandas or polars)
        """
        if not Path(file_path).exists():
            logger.warning(f"File not found: {file_path}")
            return None

        try:
            if engine == "polars":
                return pl.read_csv(file_path)
            else:
                return pd.read_csv(file_path)
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            return None

    @staticmethod
    def write_csv(df: Union[pd.DataFrame, pl.DataFrame],
                  file_path: Union[str, Path],
                  index: bool = False):
        """
        Write DataFrame to CSV.

        Args:
            df: DataFrame to write
            file_path: Output path
            index: Whether to include index (pandas only)
        """
        CSVManager.ensure_directory(file_path)
        
        try:
            if isinstance(df, pl.DataFrame):
                df.write_csv(file_path)
            else:
                df.to_csv(file_path, index=index)
            logger.info(f"Wrote {len(df)} rows to {file_path}")
        except Exception as e:
            logger.error(f"Error writing to {file_path}: {e}")

    @staticmethod
    def append_csv(df_new: Union[pd.DataFrame, pl.DataFrame],
                   file_path: Union[str, Path],
                   deduplicate_on: Optional[List[str]] = None) -> int:
        """
        Append new rows to existing CSV, handling deduplication.

        Args:
            df_new: New data to append
            file_path: CSV file path
            deduplicate_on: Column(s) to deduplicate on

        Returns:
            Number of new rows added
        """
        CSVManager.ensure_directory(file_path)
        path = Path(file_path)

        # Read existing data if file exists
        if path.exists():
            if isinstance(df_new, pl.DataFrame):
                df_existing = pl.read_csv(path)
                df_combined = pl.concat([df_existing, df_new])
            else:
                df_existing = pd.read_csv(path)
                df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_combined = df_new

        # Deduplicate if specified
        rows_before = len(df_combined)
        if deduplicate_on:
            if isinstance(df_combined, pl.DataFrame):
                df_combined = df_combined.unique(subset=deduplicate_on)
            else:
                df_combined = df_combined.drop_duplicates(subset=deduplicate_on)

        rows_after = len(df_combined)
        new_rows = rows_after - (len(df_existing) if path.exists() else 0)

        # Write back
        CSVManager.write_csv(df_combined, file_path, index=False)

        logger.info(f"Appended {new_rows} new rows to {file_path} "
                    f"(removed {rows_before - rows_after} duplicates)")

        return new_rows

    @staticmethod
    def to_parquet(df: Union[pd.DataFrame, pl.DataFrame],
                   file_path: Union[str, Path],
                   compression: str = "snappy"):
        """
        Convert and save as Parquet format.

        Args:
            df: DataFrame to save
            file_path: Output parquet path
            compression: Compression method (snappy, gzip, etc.)
        """
        CSVManager.ensure_directory(file_path)

        try:
            if isinstance(df, pl.DataFrame):
                df.write_parquet(file_path, compression=compression)
            else:
                df.to_parquet(file_path, compression=compression, index=False)
            logger.info(f"Wrote {len(df)} rows to {file_path}")
        except Exception as e:
            logger.error(f"Error writing parquet to {file_path}: {e}")

    @staticmethod
    def read_parquet(file_path: Union[str, Path],
                     engine: str = "pandas") -> Union[pd.DataFrame, pl.DataFrame]:
        """
        Read Parquet file.

        Args:
            file_path: Path to parquet file
            engine: "pandas" or "polars"

        Returns:
            DataFrame
        """
        if not Path(file_path).exists():
            logger.warning(f"File not found: {file_path}")
            return None

        try:
            if engine == "polars":
                return pl.read_parquet(file_path)
            else:
                return pd.read_parquet(file_path)
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            return None

    @staticmethod
    def list_dicts_to_csv(data: List[Dict[str, Any]],
                          file_path: Union[str, Path],
                          deduplicate_on: Optional[List[str]] = None,
                          append: bool = True) -> int:
        """
        Convert list of dicts to CSV, with optional append/deduplicate.

        Args:
            data: List of dictionaries
            file_path: Output CSV path
            deduplicate_on: Columns to deduplicate on
            append: Whether to append to existing file

        Returns:
            Number of rows written
        """
        if not data:
            logger.warning("No data to write")
            return 0

        df = pd.DataFrame(data)

        if append and Path(file_path).exists():
            rows_added = CSVManager.append_csv(
                df,
                file_path,
                deduplicate_on=deduplicate_on
            )
            return rows_added
        else:
            CSVManager.write_csv(df, file_path)
            return len(df)

    @staticmethod
    def merge_csvs(input_paths: List[Union[str, Path]],
                   output_path: Union[str, Path],
                   deduplicate_on: Optional[List[str]] = None) -> int:
        """
        Merge multiple CSV files.

        Args:
            input_paths: List of CSV file paths
            output_path: Output merged CSV path
            deduplicate_on: Columns to deduplicate on

        Returns:
            Total rows in merged file
        """
        dfs = []
        for path in input_paths:
            df = CSVManager.read_csv(path)
            if df is not None:
                dfs.append(df)

        if not dfs:
            logger.error("No valid CSV files to merge")
            return 0

        # Combine
        combined = pd.concat(dfs, ignore_index=True)

        # Deduplicate
        if deduplicate_on:
            combined = combined.drop_duplicates(subset=deduplicate_on)

        # Write
        CSVManager.write_csv(combined, output_path)
        logger.info(f"Merged {len(input_paths)} files into {output_path}")

        return len(combined)

    @staticmethod
    def get_stats(file_path: Union[str, Path]) -> Dict[str, Any]:
        """Get statistics about CSV file."""
        df = CSVManager.read_csv(file_path)
        if df is None:
            return {}

        return {
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024**2,
        }


class DataFrameConverter:
    """Convert between data models and DataFrames."""

    @staticmethod
    def models_to_dataframe(models: List[Any]) -> pd.DataFrame:
        """
        Convert Pydantic models to DataFrame.

        Args:
            models: List of Pydantic model instances

        Returns:
            DataFrame with model data
        """
        data = [model.model_dump() for model in models]
        return pd.DataFrame(data)

    @staticmethod
    def dataframe_to_records(df: pd.DataFrame,
                             orient: str = "records") -> List[Dict[str, Any]]:
        """
        Convert DataFrame to list of records.

        Args:
            df: Input DataFrame
            orient: Record orientation

        Returns:
            List of dictionaries
        """
        return df.to_dict(orient=orient)

    @staticmethod
    def flatten_nested_columns(df: pd.DataFrame,
                               sep: str = "_") -> pd.DataFrame:
        """
        Flatten nested dictionary columns.

        Args:
            df: DataFrame with potential nested data
            sep: Separator for flattened column names

        Returns:
            Flattened DataFrame
        """
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    expanded = pd.json_normalize(df[col])
                    if len(expanded.columns) > 1:
                        expanded.columns = [f"{col}{sep}{c}" for c in expanded.columns]
                        df = df.drop(col, axis=1).join(expanded)
                except Exception:
                    pass  # Column is not nested JSON

        return df


if __name__ == "__main__":
    # Example usage
    sample_data = [
        {"match_id": "1", "team_a": "Paper Rex", "team_b": "Gen.G"},
        {"match_id": "2", "team_a": "OpTic", "team_b": "FaZe"},
    ]

    CSVManager.list_dicts_to_csv(sample_data, "data/test.csv", append=False)
    stats = CSVManager.get_stats("data/test.csv")
    print(stats)

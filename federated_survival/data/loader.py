import pandas as pd
import numpy as np
from typing import Union, Dict, Optional
from pathlib import Path

class DataLoader:
    """Data loader for reading survival data in Excel and CSV formats"""
    
    def __init__(self, 
                 feature_columns: Optional[Dict[str, str]] = None,
                 time_column: str = 'time',
                 status_column: str = 'status'):
        """
        Initialize the data loader

        Args:
            feature_columns: Feature column name mapping, in the form
                {'original column name': 'target column name'}. The target names are
                arbitrary and are not required to start with 'x'; if None, all
                columns except time/status are automatically renamed to x1, x2, ...
            time_column: Survival time column name
            status_column: Event status column name
        """
        self.feature_columns = feature_columns
        self.time_column = time_column
        self.status_column = status_column
        # Reverse mapping {new column: original column} built during auto-renaming,
        # used to trace back to the original names when reporting errors
        self._auto_renamed = {}
    
    def load(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        Load a data file

        Args:
            file_path: Path to the data file; supports .xlsx, .xls, and .csv formats

        Returns:
            pd.DataFrame: Processed DataFrame in the same format as produced by DataGenerator
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        # Choose the reading method based on the file extension
        if file_path.suffix.lower() in ['.xlsx', '.xls']:
            data = pd.read_excel(file_path)
        elif file_path.suffix.lower() == '.csv':
            data = pd.read_csv(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
            
        return self._process_data(data)
    
    def _process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Process the data to match the format produced by DataGenerator

        Processing order:
            1. Validate that the time / status columns exist;
            2. Rename columns per feature_columns (auto-renamed to x1, x2, ... if not provided);
            3. All columns except time / status are treated as feature columns,
               keeping the input column order;
            4. Label columns are cast to float64 / int32, feature columns to float64;
            5. Output column order is [feature columns..., time, status].

        Args:
            data: Raw DataFrame

        Returns:
            pd.DataFrame: Processed DataFrame

        Raises:
            ValueError: If the time/status columns are missing; if no feature columns
                remain after processing; or if a feature column cannot be converted
                to float64 (e.g. a string-typed ID column slipped in).
        """
        label_columns = (self.time_column, self.status_column)

        # Ensure the required columns exist
        if self.time_column not in data.columns:
            raise ValueError(f"Time column '{self.time_column}' not found in data")
        if self.status_column not in data.columns:
            raise ValueError(f"Status column '{self.status_column}' not found in data")
            
        # Process feature columns
        if self.feature_columns is not None:
            # Rename feature columns
            data = data.rename(columns=self.feature_columns)
        else:
            # Auto-process feature columns
            feature_cols = [col for col in data.columns
                          if col not in label_columns]
            # Rename to x1, x2, ...
            new_cols = {col: f'x{i+1}' for i, col in enumerate(feature_cols)}
            # Keep the reverse mapping so errors can point back to the original column names
            self._auto_renamed = {new: old for old, new in new_cols.items()}
            data = data.rename(columns=new_cols)
        
        # Feature columns are always determined by "excluding label columns",
        # preserving the original order.
        # Note: do NOT filter by column-name prefix (e.g. 'x') — otherwise, when
        # feature_columns maps to custom names, all features would be silently
        # dropped and only the label columns would remain.
        feature_cols = [col for col in data.columns if col not in label_columns]
        if not feature_cols:
            raise ValueError(
                "No feature columns found after processing: the data only contains "
                f"the label columns {list(label_columns)}. Check 'feature_columns' "
                "and make sure non-feature columns (ids, dates, ...) are dropped."
            )
        
        # Ensure the data types are correct
        data[self.time_column] = data[self.time_column].astype(np.float64)
        data[self.status_column] = data[self.status_column].astype(np.int32)
        
        # Ensure feature columns are float64
        for col in feature_cols:
            try:
                data[col] = data[col].astype(np.float64)
            except (TypeError, ValueError) as exc:
                original = self._auto_renamed.get(col)
                source = f" (from original column '{original}')" if original else ""
                raise ValueError(
                    f"Feature column '{col}'{source} cannot be converted to "
                    f"float64 ({exc}). Drop non-numeric columns such as "
                    "identifiers before loading, or exclude them from "
                    "'feature_columns'."
                ) from exc
        
        # Reorder columns: feature columns first (original order preserved), then time and status
        final_cols = feature_cols + list(label_columns)
        data = data[final_cols]
        
        return data 
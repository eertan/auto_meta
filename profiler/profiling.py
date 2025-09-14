import polars as pl
import os

def get_data_sample(df: pl.DataFrame, num_rows: int = 5) -> str:
    """Returns a string representation of the first few rows of a DataFrame."""
    return str(df.head(num_rows))

def get_schema(df: pl.DataFrame) -> dict:
    """Returns a dictionary of column names and their data types."""
    return {col: str(dtype) for col, dtype in df.schema.items()}

def suggest_primary_keys(df: pl.DataFrame) -> list[str]:
    """Suggests potential primary key columns based on uniqueness and naming conventions."""
    candidates = []
    for col_name in df.columns:
        if df[col_name].is_unique().all() and not df[col_name].is_null().any():
            if "id" in col_name.lower():
                candidates.append(col_name)
    return candidates

def suggest_foreign_keys(df: pl.DataFrame, table_name: str, primary_keys: dict, dataframes: dict) -> list[str]:
    """Suggests potential foreign key columns by comparing to a dictionary of primary keys."""
    suggested_fks = []
    for col_name in df.columns:
        if col_name in primary_keys.get(table_name, []):
            continue

        for other_table, pks in primary_keys.items():
            if table_name == other_table:
                continue

            for pk in pks:
                if col_name.lower() == pk.lower():

                    other_df = dataframes[other_table]
                    if df[col_name].is_in(other_df[pk]).all():
                        suggested_fks.append(f"{col_name} -> {other_table}.{pk}")
    return suggested_fks

def detect_timestamp_column(df: pl.DataFrame, table_name: str) -> str:
    """Detects the most likely timestamp column in a DataFrame."""
    candidates = []
    
    # First pass: check for Date/Datetime types
    for col_name, dtype in df.schema.items():
        if isinstance(dtype, (pl.Date, pl.Datetime)):
            candidates.append(col_name)

    # If no Date/Datetime types, check by name
    if not candidates:
        for col_name in df.columns:
            if any(keyword in col_name.lower() for keyword in ["date", "time", "timestamp", "created", "updated"]):
                candidates.append(col_name)

    if not candidates:
        return ""
        
    if len(candidates) == 1:
        return candidates[0]

    # Scoring for multiple candidates
    scores = {}
    for col in candidates:
        score = 0
        col_lower = col.lower()
        
        # High-priority keywords for events
        if any(k in col_lower for k in ["created", "transaction", "order", "event_date", "timestamp"]):
            score += 3
        
        # Table name in column name adds context
        if table_name.lower() in col_lower:
            score += 2
            
        # Generic keywords
        if any(k in col_lower for k in ["date", "time"]):
            score += 1
            
        # Penalize attribute-like keywords
        if any(k in col_lower for k in ["birth", "dob", "join", "start", "end"]):
            score -= 2
            
        # Penalize low-priority keywords
        if any(k in col_lower for k in ["updated", "modified"]):
            score -= 1
            
        scores[col] = score
        
    if not scores:
        return ""

    # Return the highest-scoring candidate, or "" if all scores are negative
    best_candidate = max(scores, key=scores.get)
    return best_candidate if scores[best_candidate] > 0 else ""
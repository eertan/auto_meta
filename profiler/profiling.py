import os
import re
from difflib import SequenceMatcher

import polars as pl


class SmartSchemaDetector:
    def __init__(self, df: pl.DataFrame):
        self.df = df
        self.row_count = df.height

    def get_smart_schema(self) -> dict:
        """
        Returns a rich metadata dictionary for each column.
        """
        schema_report = {}

        # We process column by column (or you can use fold for optimizations)
        for col_name in self.df.columns:
            series = self.df[col_name]
            dtype = series.dtype

            # 1. Basic Stats
            n_unique = series.n_unique()
            n_null = series.null_count()

            # 2. Inferred Semantic Type
            semantic_type = str(dtype)
            notes = []

            # --- Heuristic: Primary Key Candidate ---
            is_pk_candidate = False
            if n_null == 0 and n_unique == self.row_count:
                is_pk_candidate = True
                notes.append("PK Candidate")

            # --- Heuristic: Categorical Detection ---
            # If unique values are < 10% of total rows or < 50 absolute values
            if (n_unique / self.row_count < 0.1) or (n_unique < 50):
                semantic_type = "Categorical"

            # --- Heuristic: Ambiguous String Analysis (Regex) ---
            if dtype == pl.Utf8:
                semantic_type = self._analyze_string_content(series, n_unique)

            schema_report[col_name] = {
                "storage_type": str(dtype),
                "semantic_type": semantic_type,
                "is_pk_candidate": is_pk_candidate,
                "completeness": 1 - (n_null / self.row_count),
                "cardinality": n_unique,
                "notes": ", ".join(notes),
            }

        return schema_report

    def _analyze_string_content(self, series: pl.Series, n_unique: int) -> str:
        """
        Analyzes string content to guess types even if column names are ambiguous.
        Sampling is used for performance.
        """
        # Sample non-null values for regex checking
        sample = series.drop_nulls().head(100)
        if sample.is_empty():
            return "EmptyString"

        # Regex Patterns
        patterns = {
            "UUID": r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
            "Email": r"^[\w\.-]+@[\w\.-]+\.\w+$",
            "Date_ISO": r"^\d{4}-\d{2}-\d{2}$",
            "Numeric_String": r"^\d+$",  # Like Zip codes or IDs stored as text
        }

        # Check if all sampled items match a pattern
        for type_name, pattern in patterns.items():
            # We use Python's re here on the sample because it's flexible
            # For massive scale, use polars .str.contains expression
            matches = sample.map_elements(
                lambda x: bool(re.match(pattern, str(x).lower())),
                return_dtype=pl.Boolean,
            )
            if matches.all():
                return type_name

        if n_unique == self.row_count:
            return "Text (Unique ID?)"

        return "Free Text"

    def identify_primary_key(self, schema_report: dict) -> list[str]:
        """
        Determines the most likely PK. Handles Composite Keys logic briefly.
        """
        # 1. Look for single column PKs
        candidates = [
            col for col, data in schema_report.items() if data["is_pk_candidate"]
        ]

        if candidates:
            # Tie breaker: Prefer Integer/UUID over String, Prefer "ID" in name
            candidates.sort(
                key=lambda x: (
                    "id" not in x.lower(),  # prioritize names with "id"
                    schema_report[x]["storage_type"]
                    == "Utf8",  # prioritize Int over String
                )
            )
            return [candidates[0]]

        # 2. If no single column PK, you would check composite keys here
        # (Computationally expensive: check uniqueness of col combinations)
        return ["Composite Key Analysis Required"]


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


def suggest_foreign_keys(
    df: pl.DataFrame, table_name: str, primary_keys: dict, dataframes: dict
) -> list[str]:
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


def analyze_foreign_keys(
    df: pl.DataFrame,
    current_table_name: str,
    dataframes: dict,
    primary_keys: dict,
    threshold: float = 0.95,
) -> list[dict]:
    """
    Generalizes FK detection by checking name patterns and data overlap ratios.
    Returns a list of dictionaries with confidence scores.
    """
    suggestions = []

    # Pre-calculate unique values for the current table to speed up checks
    # We only care about columns that are NOT the current table's PK
    current_pks = primary_keys.get(current_table_name, [])
    candidate_cols = [c for c in df.columns if c not in current_pks]

    for col_name in candidate_cols:
        col_dtype = df[col_name].dtype

        # Iterating over potential parent tables
        for other_table, pks in primary_keys.items():
            if current_table_name == other_table:
                continue

            target_df = dataframes[other_table]

            for pk in pks:
                target_dtype = target_df[pk].dtype

                # 1. HARD CONSTRAINT: Types must be compatible
                # (You can relax this to allow Int32 vs Int64, but String vs Int is a no-go)
                if col_dtype != target_dtype:
                    # Special case: Allow Int-to-Int mismatch (e.g. Int32 vs Int64)
                    if not (col_dtype.is_numeric() and target_dtype.is_numeric()):
                        continue

                # 2. NAME HEURISTICS (Scoring)
                name_score = 0.0
                col_lower = col_name.lower()
                table_lower = other_table.lower()
                pk_lower = pk.lower()

                # Case A: Exact Match (e.g. 'zip_code' -> 'zip_code')
                if col_lower == pk_lower:
                    name_score = 1.0

                # Case B: Convention Match (e.g., 'user_id' -> table 'users')
                # We strip 's' to handle plural tables: users -> user
                elif table_lower.rstrip("s") in col_lower and "id" in col_lower:
                    name_score = 0.9

                # Case C: Prefix/Suffix match (e.g., 'manager_id' -> 'id' in 'employees' table)
                elif pk_lower in col_lower:
                    name_score = 0.5

                # Optimization: If name score is 0, we generally skip data check
                # UNLESS you want "Blind Detection" (computationally expensive)
                if name_score < 0.1:
                    continue

                # 3. DATA OVERLAP (The Truth Check)
                # Calculate what % of child values exist in parent PK
                # We filter out nulls first as they usually don't violate FK constraints
                child_values = df.select(pl.col(col_name).drop_nulls())

                if child_values.is_empty():
                    continue

                # Polars optimization: Calculate overlap percentage
                # We verify the check against the UNIQUE parent PKs
                parent_pks = target_df.select(pl.col(pk).unique())

                overlap_pct = (
                    child_values.join(
                        parent_pks, left_on=col_name, right_on=pk, how="inner"
                    ).height
                    / child_values.height
                )

                # 4. FINAL DECISION
                if overlap_pct >= threshold:
                    suggestions.append(
                        {
                            "column": col_name,
                            "references": f"{other_table}.{pk}",
                            "confidence": "High" if overlap_pct == 1.0 else "Medium",
                            "overlap_percent": round(overlap_pct * 100, 2),
                            "match_reason": "Exact Name"
                            if name_score == 1.0
                            else "Pattern Match",
                        }
                    )

                # Handle "Dirty Data" cases (High overlap, but not perfect)
                elif overlap_pct > 0.5 and name_score > 0.8:
                    suggestions.append(
                        {
                            "column": col_name,
                            "references": f"{other_table}.{pk}",
                            "confidence": "Low (Dirty Data?)",
                            "overlap_percent": round(overlap_pct * 100, 2),
                            "match_reason": "Strong Name Match, Partial Data",
                        }
                    )

    return suggestions


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
            if any(
                keyword in col_name.lower()
                for keyword in ["date", "time", "timestamp", "created", "updated"]
            ):
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
        if any(
            k in col_lower
            for k in ["created", "transaction", "order", "event_date", "timestamp"]
        ):
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


def detect_primary_event_timestamp(df: pl.DataFrame, table_name: str) -> str:
    """
    Identifies the primary event timestamp by combining name heuristics
    with data profiling (completeness and cardinality).
    """
    candidates = []
    row_count = df.height
    if row_count == 0:
        return ""

    # 1. Identify ALL potential columns (Types + Names)
    # We look at Datetime types AND String types that have 'date'/'time' in the name
    potential_cols = []

    for col_name, dtype in df.schema.items():
        # Check strict types
        if isinstance(dtype, (pl.Date, pl.Datetime)):
            potential_cols.append(col_name)
        # Check strings that look suspicious (common in CSV/Parquet)
        elif dtype == pl.Utf8:
            if any(k in col_name.lower() for k in ["date", "time", "ts", "at", "day"]):
                potential_cols.append(col_name)

    if not potential_cols:
        return ""

    # 2. Score Candidates
    scores = {}

    # We do a single pass over data stats for potential columns to avoid repeated scans
    # Calculate Null counts and Unique counts
    stats = df.select(
        [pl.col(c).null_count().alias(f"{c}_nulls") for c in potential_cols]
        + [pl.col(c).n_unique().alias(f"{c}_unique") for c in potential_cols]
    ).row(0, named=True)

    for col in potential_cols:
        score = 0
        col_lower = col.lower()
        table_lower = table_name.lower()

        # --- A. DATA PHYSICS SCORING (The New Part) ---

        n_nulls = stats[f"{col}_nulls"]
        n_unique = stats[f"{col}_unique"]
        null_ratio = n_nulls / row_count

        # FATAL: If primary event time is > 30% null, it's likely not the primary event
        # (e.g. 'deleted_at' or 'shipped_date' are often null)
        if null_ratio > 0.3:
            scores[col] = -100
            continue

        # BONUS: High Cardinality (Granularity)
        # Real timestamps are usually unique or high variance.
        # Low cardinality implies a "Date" dimension (e.g. 'fiscal_month')
        if row_count > 100 and (n_unique / row_count) > 0.5:
            score += 2

        # --- B. NAMING HEURISTICS (Your Logic + Improvements) ---

        # 1. The "Gold Standard" -> Table Name + Date (e.g. 'order_date' inside 'orders')
        # We strip 's' to handle plural tables
        root_table_name = table_lower.rstrip("s")
        if (
            f"{root_table_name}_date" in col_lower
            or f"{root_table_name}_time" in col_lower
        ):
            score += 5

        # 2. The "Creation" Concept (Strongest generic signal)
        if any(
            k in col_lower
            for k in ["created_at", "record_date", "insertion_date", "event_time"]
        ):
            score += 4

        # 3. Explicit "Timestamp" or "Datetime" word
        if "timestamp" in col_lower or "datetime" in col_lower:
            score += 3

        # 4. Penalize "Lifecycle" Events (Secondary events)
        # 'updated_at' is valid, but usually 'created_at' is the immutable anchor
        if "updated" in col_lower or "modified" in col_lower:
            score -= 1  # Not terrible, but prefer created

        # 5. Penalize "Attributes" or "Windows" (Strong Penalty)
        # These are properties of the entity, not the event itself
        if any(
            k in col_lower
            for k in [
                "birth",
                "dob",
                "expiry",
                "expire",
                "due",
                "start",
                "end",
                "valid",
            ]
        ):
            score -= 10

        # 6. Penalize "Future" looking names
        if "next" in col_lower or "schedule" in col_lower or "forecast" in col_lower:
            score -= 5

        scores[col] = score

    # Filter out negative scores and sort
    valid_candidates = {k: v for k, v in scores.items() if v > 0}

    if not valid_candidates:
        return ""

    # Return the highest score
    return max(valid_candidates, key=valid_candidates.get)

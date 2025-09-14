from typing import TypedDict, Dict
from langgraph.graph import StateGraph, END
from decider.agent import DeciderAgent
from validator.agent import ValidatorAgent
from profiler.profiling import get_data_sample, suggest_primary_keys, suggest_foreign_keys, get_schema, detect_timestamp_column

class ClassificationState(TypedDict):
    """The state of the classification graph."""
    classifications: Dict
    validated_classifications: Dict  # Store classifications that have passed validation
    critique: str
    retries: int
    max_retries: int
    dataframes: Dict
    decider_agent: DeciderAgent
    validator_agent: ValidatorAgent
    rules: str
    schemas: Dict
    samples: Dict
    primary_keys: Dict
    foreign_keys: Dict
    timestamp_columns: Dict

async def discover_node(state: ClassificationState):
    """Node to discover schema, primary keys, and foreign keys for all tables."""
    print("\n--- Running Discovery Node ---")
    dataframes = state["dataframes"]
    
    schemas = {name: get_schema(df) for name, df in dataframes.items()}
    samples = {name: get_data_sample(df) for name, df in dataframes.items()}
    initial_candidates = {name: suggest_primary_keys(df) for name, df in dataframes.items()}
    

    # Step 2: Refine the candidates using cross-table validation.
    all_primary_keys = {}
    for table_name, candidates in initial_candidates.items():
        # Skip if no initial candidates were found.
        if not candidates:
            all_primary_keys[table_name] = []
            continue

        # If there are multiple candidates, we need to disqualify likely foreign keys.
        refined_pks = []
        for pk_candidate in candidates:
            is_likely_foreign_key = False
            current_unique_count = dataframes[table_name][pk_candidate].n_unique()

            # Check this candidate's unique count against all *other* tables.
            for other_name, other_df in dataframes.items():
                if table_name == other_name:
                    continue  # Don't compare a table to itself.

                if pk_candidate in other_df.columns:
                    # If another table has more unique values for this column,
                    # then it's the "home" table, and our candidate is a foreign key.
                    if other_df[pk_candidate].n_unique() > current_unique_count:
                        is_likely_foreign_key = True
                        all_primary_keys[table_name] = [c for c in candidates if c!=pk_candidate]
                        break  # Found disqualifying evidence, no need to check further.

            # If, after checking all other tables, no disqualifying evidence was found,
            # it's a valid primary key candidate.
            if not is_likely_foreign_key:
                refined_pks.append(pk_candidate)

        if refined_pks:
            all_primary_keys[table_name] = refined_pks
            
    foreign_keys = {name: suggest_foreign_keys(df, name, all_primary_keys, dataframes) for name, df in dataframes.items()}
    timestamp_columns = {name: detect_timestamp_column(df, name) for name, df in dataframes.items()}
    
    return {
        "schemas": schemas,
        "samples": samples,
        "primary_keys": all_primary_keys,
        "foreign_keys": foreign_keys,
        "timestamp_columns": timestamp_columns,
        "validated_classifications": {},  # Initialize as empty
    }

async def classify_node(state: ClassificationState):
    """Node to classify all tables."""
    print("\n--- Running Classification Node ---")
    critique = state.get("critique", "")
    dataframes = state["dataframes"]
    decider_agent = state["decider_agent"]
    schemas = state["schemas"]
    primary_keys = state["primary_keys"]
    foreign_keys = state["foreign_keys"]
    samples = state["samples"]
    timestamp_columns = state["timestamp_columns"]
    validated_classifications = state.get("validated_classifications", {})
    
    classifications = {}
    tables_to_classify = {name: df for name, df in dataframes.items() if name not in validated_classifications}

    for table_name, df in tables_to_classify.items():
        classification = await decider_agent.classify_table(
            table_name,
            schemas[table_name],
            primary_keys[table_name],
            foreign_keys[table_name],
            samples[table_name],
            timestamp_columns[table_name],
            critique
        )
        classifications[table_name] = classification
        
    # Merge new classifications with already validated ones
    classifications.update(validated_classifications)
        
    return {"classifications": classifications, "retries": state.get("retries", 0) + 1}

async def validate_node(state: ClassificationState):
    """Node to validate the classifications."""
    print("\n--- Running Validation Node ---")
    classifications = state["classifications"]
    rules = state["rules"]
    samples = state["samples"]
    primary_keys = state["primary_keys"]
    foreign_keys = state["foreign_keys"]
    timestamp_columns = state["timestamp_columns"]
    validator_agent = state["validator_agent"]
    
    result = await validator_agent.validate(
        classifications,
        rules,
        samples,
        primary_keys,
        foreign_keys,
        timestamp_columns
    )
    
    if not result.is_valid:
        critique_str = ""
        failed_tables = {c.table_name for c in result.critiques}
        validated_classifications = {
            name: cls for name, cls in classifications.items() if name not in failed_tables
        }
        for c in result.critiques:
            critique_str += f"- {c.issue} for table {c.table_name}. Reasoning: {c.reasoning}\n"
        return {
            "critique": critique_str,
            "validated_classifications": validated_classifications,
        }
    else:
        return {
            "critique": "",
            "validated_classifications": classifications,
        }

def should_continue(state: ClassificationState):
    """Conditional edge to decide whether to continue or finish."""
    if state.get("critique"):
        if state.get("retries", 0) >= state.get("max_retries", 3):
            print("Max retries reached. Finishing.")
            return END
        else:
            print("Validation failed. Retrying with critique...")
            return "classify"
    else:
        print("Validation successful. Finishing.")
        return END

def build_graph():
    """Builds the classification graph."""
    workflow = StateGraph(ClassificationState)
    workflow.add_node("discover", discover_node)
    workflow.add_node("classify", classify_node)
    workflow.add_node("validate", validate_node)
    workflow.add_conditional_edges(
        "validate",
        should_continue,
    )
    workflow.set_entry_point("discover")
    workflow.add_edge("discover", "classify")
    workflow.add_edge("classify", "validate")
    
    return workflow.compile()

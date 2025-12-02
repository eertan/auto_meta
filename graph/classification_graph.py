from typing import TypedDict, Dict
from langgraph.graph import StateGraph, END
from decider.agent import DeciderAgent
from validator.agent import ValidatorAgent
from profiler.profiling import get_data_sample, SmartSchemaDetector, analyze_foreign_keys, detect_primary_event_timestamp

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
    
    schemas = {}
    primary_keys = {}
    samples = {}
    
    for name, df in dataframes.items():
        # Smart Profiling
        detector = SmartSchemaDetector(df)
        schema_report = detector.get_smart_schema()
        
        # Format schema for the agent
        schemas[name] = {
            col: f"{info['semantic_type']} ({info['storage_type']})" 
            for col, info in schema_report.items()
        }
        
        # Identify PK
        primary_keys[name] = detector.identify_primary_key(schema_report)
        samples[name] = get_data_sample(df)

    # Detect Foreign Keys
    foreign_keys = {}
    for name, df in dataframes.items():
        suggestions = analyze_foreign_keys(df, name, dataframes, primary_keys)
        foreign_keys[name] = [
            f"{s['column']} -> {s['references']} (Conf: {s['confidence']})" 
            for s in suggestions
        ]

    # Detect Timestamps
    timestamp_columns = {
        name: detect_primary_event_timestamp(df, name) 
        for name, df in dataframes.items()
    }
    
    return {
        "schemas": schemas,
        "samples": samples,
        "primary_keys": primary_keys,
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

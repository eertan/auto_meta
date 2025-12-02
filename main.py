import polars as pl
import os
import asyncio
from decider.agent import DeciderAgent
from validator.agent import ValidatorAgent
from graph.classification_graph import build_graph


async def main():
    """Main function to classify data sources using the classification graph."""
    data_dir = "data"
    dataframes = {}

    # --- Step 1: Load all data ---
    print("--- Step 1: Loading data ---")
    for filename in os.listdir(data_dir):
        if filename.endswith(".csv"):
            file_path = os.path.join(data_dir, filename)
            table_name = filename.replace(".csv", "")
            df = pl.read_csv(file_path, truncate_ragged_lines=True)
            dataframes[table_name] = df

    # --- Step 2: Initialize Agents ---
    print("\n--- Step 2: Initializing Agents ---")
    decider_agent = DeciderAgent()
    validator_agent = ValidatorAgent()

    # --- Step 3: Define Classification Rules ---
    rules = """
    - Entity: Describes a business object. Has a primary key.
    - Event: Records something that happened. Must have a timestamp.
    - State: Records a state valid for a period. Must have a timestamp.
    - Relationship: Connects two or more entities.
    - Participation: Connects an event to an entity. A key indicator is a foreign key to a table classified as 'Event'.
    """

    # --- Step 4: Build and run the graph ---
    print("\n--- Step 4: Running classification graph ---")
    app = build_graph()
    initial_state = {
        "dataframes": dataframes,
        "decider_agent": decider_agent,
        "validator_agent": validator_agent,
        "rules": rules,
        "retries": 0,
        "max_retries": 3,
    }
    
    final_state = await app.ainvoke(initial_state)

    # --- Step 5: Print final classifications ---
    print("\n--- Final Classifications ---")
    if not final_state.get("critique"):
        for table_name, classification in sorted(final_state["classifications"].items()):
            print(f"- {table_name}: {classification}")
    else:
        print("Classification failed after max retries.")
        print(f"Final critique: {final_state['critique']}")


if __name__ == "__main__":
    asyncio.run(main())

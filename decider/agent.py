import asyncio
import os

import dspy

# Check for API key
api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("Either GEMINI_API_KEY or GOOGLE_API_KEY environment variable must be set.")


class Classify(dspy.Signature):
    """You are a data architect. Your task is to classify a data source based on the evidence provided.
Classify the table as one of the following types:
- Entity
- Event
- State
- Relationship
- Participation

{definitions}

{critique_section}

Evidence for "{table_name}":
{evidence}

Respond with only the classification for "{table_name}".

Classification:"""

    table_name = dspy.InputField()
    definitions = dspy.InputField()
    critique_section = dspy.InputField()
    evidence = dspy.InputField()
    classification = dspy.OutputField()


class DeciderAgent(dspy.Module):
    """An agent that classifies data sources using generative AI."""

    def __init__(self):
        super().__init__()
        self.classify_signature = dspy.Predict(Classify)
        gemini = dspy.LM("gemini/gemini-2.5-pro", api_key=api_key)
        dspy.settings.configure(lm=gemini)

    async def classify_table(
            self,
            table_name: str,
            schema: str,
            pks: list,
            fks: list,
            sample_data: str,
            timestamp_column: str,
            critique: str = ""
    ) -> str:
        """
        Classifies a single table using the LLM agent.
        """
        definitions = """
        Here are the definitions and heuristics:
        - Entity: Describes a business object (e.g., Customers, Products). Has a primary key. Can have foreign keys. If it has a date column, it's usually for tracking creation or updates, not the primary purpose of the record.
        - Event: Records something that happened at a specific point in time (e.g., Orders, Transactions). Must have a column with a Date or Datetime data type. Table names often imply actions.
        - State: Records a state that is valid for a period (e.g., Weather, Marital Status). Must have a column with a Date or Datetime data type.
        - Relationship: Connects two or more entities. Usually consists of only foreign keys.
        - Participation: Connects an event to an entity. To identify this, check the foreign keys. If one foreign key points to a table that looks like an Event (e.g., "orders", "transactions") and another points to an Entity (e.g., "products", "customers"), then it is a Participation table.
        """

        critique_section = ""
        if critique:
            critique_section = f"""
        A previous attempt at classification was found to be incorrect. You must correct it based on the following critique. Prioritize the critique over your own initial analysis.

        Critique:
        {critique}

        Based on this critique, re-evaluate the evidence and provide the correct classification.
        """

        evidence = f"""
        1.  **Schema (Column: DataType)**: {schema}
        2.  **Suggested Primary Key(s)**: {', '.join(pks) if pks else "None"}
        3.  **Suggested Foreign Key(s)**: {', '.join(fks) if fks else "None"}
        4.  **Detected Timestamp Column**: {timestamp_column or "None"}
        5.  **Data Sample**:
            {sample_data}
        """

        print(f"Asking decider agent to classify '{table_name}'...")

        response = await self.classify_signature.acall(
            table_name=table_name,
            definitions=definitions,
            critique_section=critique_section,
            evidence=evidence
        )

        # Parse the response to get the classification
        response_text = str(response.classification).strip()
        for classification in ["Entity", "Event", "State", "Relationship", "Participation"]:
            if classification.lower() in response_text.lower():
                return classification

        return "Unknown"
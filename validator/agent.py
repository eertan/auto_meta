import asyncio
import os
from typing import List

import dspy
from pydantic import BaseModel, Field

# Check for API key
api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("Either GEMINI_API_KEY or GOOGLE_API_KEY environment variable must be set.")


class CritiqueDetail(BaseModel):
    """
    A detailed, structured critique for a single table's classification error.
    """
    table_name: str = Field(
        ...,
        description="The name of the table with the classification error."
    )
    reasoning: str = Field(
        ...,
        description="Step-by-step reasoning explaining why the classification is wrong. Must explicitly reference the rule and the classifications of connected tables."
    )
    current_classification: str = Field(
        ...,
        description="The incorrect classification that was provided for the table."
    )
    issue: str = Field(
        ...,
        description="A sharp, one-sentence summary of the validation error."
    )
    suggested_classification: str = Field(
        ...,
        description="The correct classification for the table based on the rules."
    )
    violated_rule: str = Field(
        ...,
        description="The specific name of the validation rule that was violated (e.g., 'Relationship Check')."
    )


class ValidationResult(BaseModel):
    """
    The final validation result, containing an overall status and a list of specific critiques.
    """
    is_valid: bool = Field(
        ...,
        description="Overall validation status. Must be `true` if the critiques list is empty, and `false` otherwise."
    )
    critiques: List[CritiqueDetail] = Field(
        ...,
        description="A list of detailed critiques for each classification error found. If validation is successful, this MUST be an empty list `[]`."
    )


class ValidateClassifications(dspy.Signature):
    """You are a meticulous and demanding data architect AI. Your task is to rigorously validate a set of data source classifications based on a strict set of rules and evidence. Your analysis must be logical, evidence-based, and free of subjective opinion.

Your final output must be ONLY the JSON object and nothing else.
- For each critique, you MUST first fill out the "reasoning" field, explaining your step-by-step logic.
- For each critique, you MUST include the following fields: "table_name", "reasoning", "current_classification", "issue", "suggested_classification", and "violated_rule".
- If all classifications are correct, return: `{"is_valid": true, "critiques": []}`
- If incorrect, return `{"is_valid": false, "critiques": [...]}`.
"""
    rules = dspy.InputField(desc="The validation rules to apply.")
    classifications_with_evidence = dspy.InputField(desc="The classifications to validate, with evidence for each.")
    validation_result = dspy.OutputField(desc="The final validation result.", model=ValidationResult)


class ValidatorAgent(dspy.Module):
    """An agent that validates the classification of data sources."""

    def __init__(self):
        super().__init__()
        self.validate_signature = dspy.Predict(ValidateClassifications)
        gemini = dspy.LM("gemini/gemini-2.5-pro", api_key=api_key)
        dspy.settings.configure(lm=gemini)

    async def validate(
            self,
            classifications: dict,
            rules: str,
            samples: dict,
            primary_keys: dict,
            foreign_keys: dict,
            timestamp_columns: dict
    ) -> ValidationResult:
        """
        Validates the classification of data sources.
        """
        import json
        from pydantic import ValidationError

        classifications_with_evidence = ""
        for table, classification in classifications.items():
            classifications_with_evidence += f"""
- Table: {table}
  Classification: {classification}
  Primary Key(s): {primary_keys.get(table, "None")}
  Foreign Key(s): {foreign_keys.get(table, "None")}
  Timestamp Column: {timestamp_columns.get(table, "None")}

"""
        print("Asking validator agent to critique the classification...")

        response = await self.validate_signature.acall(
            rules=rules,
            classifications_with_evidence=classifications_with_evidence
        )

        json_str = response.validation_result
        # Clean up potential markdown code fences
        if not json_str:
            print("Warning: json_str is None or empty. Returning default invalid state.")
            return ValidationResult(is_valid=False, critiques=[])
        if json_str.startswith("```json"):
            json_str = json_str[7:-4].strip()

        try:
            data = json.loads(json_str)
            return ValidationResult.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as e:
            print(f"Error parsing validation result: {e}")
            # Return a default "invalid" state on parsing failure
            return ValidationResult(is_valid=False, critiques=[])
from weave.evaluation.eval_imperative import ScoreType
from inspect_ai.scorer import Value, CORRECT, INCORRECT
from typing import Sequence, Mapping
from logging import getLogger

utils_logger = getLogger(__name__)

def format_score_types(score_value: Value, scorer_name: str | None = None) -> ScoreType:
    # Scorers that return CORRECT/INCORRECT - convert to boolean for Weave aggregation
    CORRECT_INCORRECT_SCORERS = [
        # Built-in scorers
        "choice", "match", "model_graded_qa",
        # Custom scorers (extend this list as needed)
        "hle_grader", "bfcl_scorer", "grid_match", 
        "hallulens_qa_scorer", "hallulens_qa",
        "kobbq_scorer", 
        "refusal_scorer", "hallulens_refusal",
        "swebench_server_scorer",
    ]
    if scorer_name in CORRECT_INCORRECT_SCORERS:
        if score_value == CORRECT:
            return True
        elif score_value == INCORRECT:
            return False
        else:
            utils_logger.warning(f"{scorer_name} is expected only to return values {CORRECT}, {INCORRECT}. Logging raw value to Weave.")
    # Handle string scores that represent correct/incorrect
    if isinstance(score_value, str):
        # Convert common score strings to boolean for proper aggregation
        if score_value in ["C", CORRECT]:
            return True
        elif score_value in ["I", INCORRECT]:
            return False
        return {"score": score_value}
    elif isinstance(score_value, int):
        return float(score_value)
    elif isinstance(score_value, Sequence):
        if len(score_value) != 1:
            raise ValueError("Sequence score cannot be passed to Weave")
        return {"score": score_value[0]}
    elif isinstance(score_value, Mapping):
        return dict(score_value)
    else:
        return score_value
    
def format_sample_display_name(template: str, task_name: str, sample_id: int | str, epoch: int) -> str:
    """
    Format sample display name using template string with safe variable substitution.
    
    Args:
        template: Template string with placeholders like {task_name}, {sample_id}, etc.
        task_name: Name of the task being evaluated
        sample_id: ID of the sample (numeric or string)
        epoch: Epoch number
        
    Returns:
        Formatted display name string
    """
    try:
        if not template.strip():
            raise ValueError("Empty template")
        return template.format(task_name=task_name, sample_id=sample_id, epoch=epoch)
    except (KeyError, ValueError):
        return f"{task_name}-sample-{sample_id}-epoch-{epoch}"

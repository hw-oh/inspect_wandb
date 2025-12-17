

from inspect_wandb.config.settings.base import InspectWandBBaseSettings
from pydantic import Field
from pydantic_settings import SettingsConfigDict

class WeaveSettings(InspectWandBBaseSettings):
    """
    Settings model for the Weave integration.
    """

    model_config = SettingsConfigDict(
        env_prefix="INSPECT_WANDB_WEAVE_", 
        pyproject_toml_table_header=("tool", "inspect-wandb", "weave"),
    )

    sample_name_template: str = Field(default="{task_name}-sample-{sample_id}-epoch-{epoch}", description="Template for sample display names. Available variables: {task_name}, {sample_id}, {epoch}")
    exclude_version_changing_metadata: bool = Field(default=True, description="Exclude run_id, task_id, eval_id from eval metadata to prevent Weave evaluation version changes on each run")
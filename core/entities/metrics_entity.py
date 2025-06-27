from pydantic import BaseModel, Field, AliasChoices


class FeatureMetrics(BaseModel):
    ba_score: float = Field(
        ...,
        alias="body_alignment",
        serialization_alias="ba_score",
        validation_alias=AliasChoices("body_alignment", "ba_score"),
    )
    jc_score: float = Field(
        ...,
        alias="joint_consistency",
        serialization_alias="jc_score",
        validation_alias=AliasChoices("joint_consistency", "jc_score"),
    )
    lc_score: float = Field(
        ...,
        alias="load_control",
        serialization_alias="lc_score",
        validation_alias=AliasChoices("load_control", "lc_score"),
    )
    sc_score: float = Field(
        ...,
        alias="speed_control",
        serialization_alias="sc_score",
        validation_alias=AliasChoices("speed_control", "sc_score"),
    )
    os_score: float = Field(
        ...,
        alias="overall_stability",
        serialization_alias="os_score",
        validation_alias=AliasChoices("overall_stability", "os_score"),
    )

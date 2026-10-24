"""
Configuration module for AI Developer Tools Prediction pipeline.
Uses dataclasses for type safety and reproducibility.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


SEED: int = 42

# ── AI tool labels (union of HaveWorkedWith + WantToWorkWith) ──────────────
AI_TOOL_LABELS: list[str] = [
    "openAI GPT (chatbot models)",
    "Anthropic: Claude Sonnet",
    "openAI Reasoning models",
    "Gemini (Flash general purpose models)",
    "Gemini (Pro Reasoning models)",
    "openAI Image generating models",
    "DeepSeek (R- Reasoning models)",
    "Meta Llama (all models)",
    "DeepSeek (V- General purpose models)",
    "Mistral AI models",
    "X Grok models",
    "Microsoft Phi-4 models",
    "Perplexity Sonar models",
    "Alibaba Cloud Qwen models",
    "Amazon Titan models",
    "Cohere: Command A",
    "Reka (Flash 3 or other Reka models)",
]

# ── Raw feature columns to include ────────────────────────────────────────
CATEGORICAL_COLS: list[str] = [
    "MainBranch", "Age", "EdLevel", "Employment", "RemoteWork",
    "OrgSize", "ICorPM", "Country", "DevType", "Industry",
    "AISelect", "AISent", "AIAcc", "AIComplex", "AIThreat",
    "AIAgents", "AIAgentChange", "AIHuman", "NewRole",
    "SOVisitFreq", "SODuration", "SOPartFreq", "SOComm",
    "PurchaseInfluence", "LearnCodeChoose", "JobSat",
]

MULTI_HOT_COLS: list[str] = [
    "LanguageHaveWorkedWith",
    "DatabaseHaveWorkedWith",
    "PlatformHaveWorkedWith",
    "WebframeHaveWorkedWith",
    "DevEnvsHaveWorkedWith",
    "CommPlatformHaveWorkedWith",
    "OfficeStackAsyncHaveWorkedWith",
    "LearnCode",
    "LearnCodeAI",
    "AILearnHow",
]

NUMERIC_COLS: list[str] = [
    "WorkExp", "YearsCode", "ToolCountWork", "ToolCountPersonal",
    "ConvertedCompYearly",
    "TechEndorse_1", "TechEndorse_2", "TechEndorse_3", "TechEndorse_4",
    "TechEndorse_5", "TechEndorse_6", "TechEndorse_7", "TechEndorse_8",
    "TechEndorse_9", "TechEndorse_13",
    "JobSatPoints_1", "JobSatPoints_2", "JobSatPoints_3", "JobSatPoints_4",
    "JobSatPoints_5", "JobSatPoints_6", "JobSatPoints_7", "JobSatPoints_8",
    "JobSatPoints_9", "JobSatPoints_10", "JobSatPoints_11", "JobSatPoints_13",
    "JobSatPoints_14", "JobSatPoints_15", "JobSatPoints_16",
]

BINARY_COLS: list[str] = [
    "AIToolCurrently partially AI",
    "AIToolDon't plan to use AI for this task",
    "AIToolPlan to partially use AI",
    "AIToolPlan to mostly use AI",
    "AIToolCurrently mostly AI",
    "AIModelsChoice",
    "TechOppose_1", "TechOppose_2", "TechOppose_3", "TechOppose_5",
    "TechOppose_7", "TechOppose_9", "TechOppose_11", "TechOppose_13",
    "TechOppose_16", "TechOppose_15",
    "SO_Actions_1", "SO_Actions_16", "SO_Actions_3", "SO_Actions_4",
    "SO_Actions_5", "SO_Actions_6", "SO_Actions_9", "SO_Actions_7",
    "SO_Actions_10", "SO_Actions_15",
    "AIAgentImpactSomewhat agree", "AIAgentImpactNeutral",
    "AIAgentImpactSomewhat disagree", "AIAgentImpactStrongly agree",
    "AIAgentImpactStrongly disagree",
    "AIAgentChallengesNeutral", "AIAgentChallengesSomewhat disagree",
    "AIAgentChallengesStrongly agree", "AIAgentChallengesSomewhat agree",
    "AIAgentChallengesStrongly disagree",
]


@dataclass
class DataConfig:
    """Data loading and splitting configuration."""
    data_path: Path = Path("survey_demo.csv")
    test_size: float = 0.15
    val_size: float = 0.15          # fraction of train set used for validation
    min_label_freq: int = 50        # drop labels appearing fewer times
    multi_hot_top_k: int = 30       # top-K values to keep per multi-hot column
    max_cat_cardinality: int = 50   # OHE truncation threshold


@dataclass
class ModelConfig:
    """Gradient-boosted multi-label classifier configuration."""
    # sklearn MultiOutputClassifier wraps a base estimator
    n_estimators: int = 300
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    n_jobs: int = -1
    random_state: int = SEED


@dataclass
class TrainConfig:
    """Training loop configuration."""
    cv_folds: int = 5
    mlflow_experiment: str = "ai_tools_prediction"
    mlflow_run_name: Optional[str] = None
    output_dir: Path = Path("outputs")
    model_artifact_name: str = "ai_tools_xgb_model"

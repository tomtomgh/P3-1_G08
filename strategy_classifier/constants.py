"""Global constants: parameter names and strategy list."""

PARAMS = ["frequency", "amplitude", "offset", "phase shift"]

EXPLORATION_STRATEGIES = [
    "structured_exploration",
    "random_trial_error",
    "curiosity_broad_exploration",
    "systematic_parameter_sweep",
]

TUNING_STRATEGIES = [
    "goal_directed_tuning",
    "iterative_finetuning",
    "incremental_adjustment",
]

RULE_BASED_PARAM_STRATEGIES = [
    "votat_like",
    "hotat_like",
    "change_all_like",
    "mixed_strategy",
]

REPETITION_STRATEGIES = [
    "repetition_practice",
    "trial_repetition_improvement",
    "trial_repetition_no_improvement",
]

ERROR_BACKTRACKING_STRATEGIES = [
    "backtracking_recovery",
    "undo_correction",
    "loop_stuck_state",
]

COORDINATION_STRATEGIES = [
    "coordinated_with_pair",
    "coordinated_group",
    "uncoordinated",
]

BEHAVIOURAL_STRATEGIES = [
    "help_seeking_pause",
    "inactivity_wait",
    "playful_inefficient",
]

ALL_STRATEGIES = (
    EXPLORATION_STRATEGIES
    + TUNING_STRATEGIES
    + RULE_BASED_PARAM_STRATEGIES
    + REPETITION_STRATEGIES
    + ERROR_BACKTRACKING_STRATEGIES
    + COORDINATION_STRATEGIES
    + BEHAVIOURAL_STRATEGIES
)

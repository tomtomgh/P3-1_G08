from .base import StrategyRule
from .exploration import get_exploration_rules
from .tuning import get_tuning_rules
from .repetition import get_repetition_rules
from .error_backtracking import get_error_backtracking_rules
# from .coordination import get_coordination_rules
from .behavioural import get_behavioural_rules
# from .rule_based_params import get_param_strategy_rules

def get_all_strategy_rules():
    rules = {}
    rules.update(get_exploration_rules())
    rules.update(get_tuning_rules())
    # rules.update(get_param_strategy_rules())
    rules.update(get_repetition_rules())
    rules.update(get_error_backtracking_rules())
    # rules.update(get_coordination_rules())
    rules.update(get_behavioural_rules())
    return rules

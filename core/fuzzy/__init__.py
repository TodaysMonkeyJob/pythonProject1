"""
Fuzzy inference core (F15 / TOD-26): linguistic variables, membership functions and a Mamdani rule base per
location. Shared by the fuzzy trust layer (TOD-27) and the fuzzy risk corridor (TOD-28). numpy only, no network.
"""
from core.fuzzy.engine import FuzzyEngine, load_yaml
from core.fuzzy.rules import Rule, RuleError, parse_rule
from core.fuzzy.validate import FuzzyConfigError
from core.fuzzy.variables import LinguisticVariable

__all__ = ["FuzzyEngine", "load_yaml", "FuzzyConfigError", "LinguisticVariable", "Rule", "RuleError", "parse_rule"]

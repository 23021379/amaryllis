class PipelineError(Exception):
    """[REDACTED_BY_SCRIPT]"""
    pass

class TriageError(PipelineError):
    """[REDACTED_BY_SCRIPT]"""
    pass

class StrategyError(PipelineError):
    """[REDACTED_BY_SCRIPT]"""
    pass

class ValidationError(PipelineError):
    """[REDACTED_BY_SCRIPT]"""
    pass

class MaxRetriesExceededError(ValidationError):
    """[REDACTED_BY_SCRIPT]"""
    pass

class SynthesisError(PipelineError):
    """[REDACTED_BY_SCRIPT]"""
    pass

class PreflightCheckError(PipelineError):
    """[REDACTED_BY_SCRIPT]"""
    pass
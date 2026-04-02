import logging
from typing import final, Any
from urllib.parse import urlparse

from pydantic import ValidationError as PydanticValidationError

from ..data_contracts import DraftPlan, ExecutionPlan
from ..exceptions import SynthesisError

@final
class SynthesisAgent:
    """
    Takes a validated DraftPlan and synthesizes it into a final,
    production-ready ExecutionPlan.
    """
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    async def run(self, draft_plan: DraftPlan) -> ExecutionPlan:
        """
        Transforms a DraftPlan into a final ExecutionPlan object.
        Returns the validated ExecutionPlan object.
        """
        domain = self._get_domain(draft_plan)
        self.logger.info(f"[REDACTED_BY_SCRIPT]'{domain}'.")
        try:
            execution_plan_data = self._transform_to_final_schema(draft_plan, domain)
            validated_plan = ExecutionPlan.model_validate(execution_plan_data)
            
            # Local Mode: We do not push to dataset here. 
            # The orchestrator/main loop handles file saving.
            
            self.logger.info(f"[REDACTED_BY_SCRIPT]'{domain}'.")
            return validated_plan

        except PydanticValidationError as e:
            raise SynthesisError(f"[REDACTED_BY_SCRIPT]")
        except Exception as e:
            raise SynthesisError(f"[REDACTED_BY_SCRIPT]")

    def _get_domain(self, draft_plan: DraftPlan) -> str:
        """[REDACTED_BY_SCRIPT]"""
        domain = draft_plan.source_triage_result.domain
        if not domain:
            raise SynthesisError("Cannot determine domain: 'domain'[REDACTED_BY_SCRIPT]")
        return domain

    def _transform_to_final_schema(self, draft_plan: DraftPlan, domain: str) -> dict[str, Any]:
        """
        Pure data mapping function from the internal DraftPlan to a dictionary
        matching the ExecutionPlan schema.
        """
        # PROJECT GENESIS DIRECTIVE: This is the designated point for final
        # structural transformation. The raw dict from the DraftPlan is wrapped
        # to match the Pydantic model required by ExecutionPlan.
        detail_processing_for_synthesis = draft_plan.detail_page_processing
        
        # Note: '[REDACTED_BY_SCRIPT]' is currently populated in the Orchestrator,
        # not the DraftPlan, because Phase 5 is an Orchestrator-level loop.
        # This agent simply passes through what it has.
        # However, for consistency, if DraftPlan is updated later, we map it here.
        # For now, DraftPlan doesn't carry document_page_processing, so we return None.
        # The Orchestrator overrides this in the final construction anyway.

        return {
            "domain": domain,
            "instructions": draft_plan.instructions,
            "[REDACTED_BY_SCRIPT]": draft_plan.[REDACTED_BY_SCRIPT],
            "[REDACTED_BY_SCRIPT]": detail_processing_for_synthesis,
            "[REDACTED_BY_SCRIPT]": None
        }
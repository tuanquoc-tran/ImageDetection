"""
Behavior Controller - Central State Management
Moved from root to controller module.
Implements deterministic state machine for all system behaviors.
"""

from enum import Enum, auto
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class SystemState(Enum):
    """System behavior states."""
    IDLE = auto()
    IMAGE_MODE = auto()
    CAMERA_MODE = auto()
    TEACH_SAMPLE = auto()
    READY = auto()
    INSPECTION = auto()
    ERROR = auto()
    SHUTDOWN = auto()


@dataclass
class StateTransition:
    """Represents a state transition event."""
    from_state: SystemState
    to_state: SystemState
    timestamp: str
    trigger: str
    metadata: Dict[str, Any] = None


class BehaviorController:
    """
    Central behavior/state controller.
    Manages all system state transitions and validates workflows.
    """
    
    # Valid state transitions (from_state -> [valid_to_states])
    VALID_TRANSITIONS = {
        SystemState.IDLE: [
            SystemState.IMAGE_MODE,
            SystemState.CAMERA_MODE,
            SystemState.TEACH_SAMPLE,
            SystemState.READY,
            SystemState.SHUTDOWN
        ],
        SystemState.IMAGE_MODE: [
            SystemState.IMAGE_MODE,
            SystemState.TEACH_SAMPLE,
            SystemState.INSPECTION,
            SystemState.IDLE,
            SystemState.CAMERA_MODE,
            SystemState.READY,
            SystemState.ERROR
        ],
        SystemState.CAMERA_MODE: [
            SystemState.CAMERA_MODE,
            SystemState.TEACH_SAMPLE,
            SystemState.INSPECTION,
            SystemState.IDLE,
            SystemState.IMAGE_MODE,
            SystemState.READY,
            SystemState.ERROR
        ],
        SystemState.TEACH_SAMPLE: [
            SystemState.READY,
            SystemState.IMAGE_MODE,
            SystemState.CAMERA_MODE,
            SystemState.IDLE,
            SystemState.ERROR
        ],
        SystemState.READY: [
            SystemState.IMAGE_MODE,
            SystemState.CAMERA_MODE,
            SystemState.INSPECTION,
            SystemState.TEACH_SAMPLE,
            SystemState.IDLE,
            SystemState.ERROR
        ],
        SystemState.INSPECTION: [
            SystemState.READY,
            SystemState.IMAGE_MODE,
            SystemState.CAMERA_MODE,
            SystemState.ERROR
        ],
        SystemState.ERROR: [
            SystemState.IDLE,
            SystemState.READY,
            SystemState.SHUTDOWN
        ],
        SystemState.SHUTDOWN: []
    }
    
    def __init__(self):
        """Initialize behavior controller."""
        self._current_state = SystemState.IDLE
        self._previous_state = None
        self._transition_history = []
        self._state_callbacks = {}
        self._golden_sample_loaded = False
        self._source_configured = False
        self._error_message = None
        
        logger.info("BehaviorController initialized")
    
    @property
    def current_state(self) -> SystemState:
        """Get current system state."""
        return self._current_state
    
    @property
    def previous_state(self) -> Optional[SystemState]:
        """Get previous system state."""
        return self._previous_state
    
    @property
    def is_golden_sample_loaded(self) -> bool:
        """Check if golden sample is loaded."""
        return self._golden_sample_loaded
    
    @property
    def is_source_configured(self) -> bool:
        """Check if image source is configured."""
        return self._source_configured
    
    @property
    def error_message(self) -> Optional[str]:
        """Get current error message."""
        return self._error_message
    
    def can_transition_to(self, target_state: SystemState) -> bool:
        """Check if transition to target state is valid."""
        valid_targets = self.VALID_TRANSITIONS.get(self._current_state, [])
        return target_state in valid_targets
    
    def transition_to(self, target_state: SystemState, trigger: str = "manual",
                     metadata: Dict[str, Any] = None) -> bool:
        """
        Transition to target state with validation.
        
        Args:
            target_state: Desired state
            trigger: What triggered this transition
            metadata: Optional metadata about transition
            
        Returns:
            True if transition successful
        """
        # Allow self-transitions for IMAGE_MODE and CAMERA_MODE
        if self._current_state == target_state and target_state in [SystemState.IMAGE_MODE, SystemState.CAMERA_MODE]:
            logger.info(f"Self-transition allowed: {target_state.name} (trigger: {trigger})")
            return True
        
        # Validate transition
        if not self.can_transition_to(target_state):
            logger.warning(
                f"Invalid transition: {self._current_state.name} -> {target_state.name}"
            )
            return False
        
        # Validate prerequisites
        if not self._validate_state_prerequisites(target_state):
            return False
        
        # Execute transition
        self._previous_state = self._current_state
        self._current_state = target_state
        
        # Clear error message if leaving error state
        if self._previous_state == SystemState.ERROR:
            self._error_message = None
        
        # Record transition
        transition = StateTransition(
            from_state=self._previous_state,
            to_state=target_state,
            timestamp=datetime.now().isoformat(),
            trigger=trigger,
            metadata=metadata
        )
        self._transition_history.append(transition)
        
        logger.info(
            f"State transition: {self._previous_state.name} -> {target_state.name} "
            f"(trigger: {trigger})"
        )
        
        # Execute state callbacks
        self._execute_state_callbacks(target_state, metadata)
        
        return True
    
    def _validate_state_prerequisites(self, target_state: SystemState) -> bool:
        """Validate prerequisites for entering target state."""
        # READY state requires golden sample
        if target_state == SystemState.READY:
            if not self._golden_sample_loaded:
                logger.error("Cannot enter READY: No golden sample loaded")
                self._enter_error_state("No golden sample loaded")
                return False
        
        # INSPECTION state requires golden sample and source
        if target_state == SystemState.INSPECTION:
            if not self._golden_sample_loaded:
                logger.error("Cannot enter INSPECTION: No golden sample")
                self._enter_error_state("No golden sample for inspection")
                return False
            if not self._source_configured:
                logger.error("Cannot enter INSPECTION: No source configured")
                self._enter_error_state("No image source configured")
                return False
        
        return True
    
    def _enter_error_state(self, error_message: str):
        """Force entry into error state."""
        self._error_message = error_message
        self._previous_state = self._current_state
        self._current_state = SystemState.ERROR
        
        logger.error(f"Entered ERROR state: {error_message}")
        
        # Execute error callbacks
        self._execute_state_callbacks(SystemState.ERROR, {'error': error_message})
    
    def register_state_callback(self, state: SystemState, callback: Callable):
        """Register callback for state entry."""
        if state not in self._state_callbacks:
            self._state_callbacks[state] = []
        self._state_callbacks[state].append(callback)
        
        logger.debug(f"Registered callback for state: {state.name}")
    
    def _execute_state_callbacks(self, state: SystemState, metadata: Dict = None):
        """Execute callbacks for state entry."""
        callbacks = self._state_callbacks.get(state, [])
        for callback in callbacks:
            try:
                if metadata:
                    callback(state, metadata)
                else:
                    callback(state)
            except Exception as e:
                logger.error(f"Callback error for {state.name}: {e}")
    
    # State flag setters
    
    def set_golden_sample_loaded(self, loaded: bool):
        """Update golden sample loaded flag."""
        self._golden_sample_loaded = loaded
        logger.info(f"Golden sample loaded: {loaded}")
        
        if loaded and self._current_state == SystemState.IDLE:
            self.transition_to(SystemState.READY, trigger="golden_sample_loaded")
    
    def set_source_configured(self, configured: bool):
        """Update source configured flag."""
        self._source_configured = configured
        logger.info(f"Source configured: {configured}")
    
    # High-level workflow methods
    
    def start_image_mode(self) -> bool:
        """Start image mode workflow."""
        if self.transition_to(SystemState.IMAGE_MODE, trigger="user_action"):
            self.set_source_configured(True)
            return True
        return False
    
    def start_camera_mode(self) -> bool:
        """Start camera mode workflow."""
        if self.transition_to(SystemState.CAMERA_MODE, trigger="user_action"):
            self.set_source_configured(True)
            return True
        return False
    
    def start_teaching(self) -> bool:
        """Start teaching workflow."""
        return self.transition_to(SystemState.TEACH_SAMPLE, trigger="user_action")
    
    def complete_teaching(self, success: bool) -> bool:
        """Complete teaching workflow."""
        if success:
            self.set_golden_sample_loaded(True)
            return self.transition_to(SystemState.READY, trigger="teaching_complete")
        else:
            self._enter_error_state("Teaching failed")
            return False
    
    def start_inspection(self) -> bool:
        """Start inspection workflow."""
        return self.transition_to(SystemState.INSPECTION, trigger="user_action")
    
    def complete_inspection(self, success: bool) -> bool:
        """Complete inspection and return to appropriate state."""
        if success:
            # Return to mode-specific state
            if self._previous_state == SystemState.IMAGE_MODE:
                return self.transition_to(SystemState.IMAGE_MODE, trigger="inspection_complete")
            elif self._previous_state == SystemState.CAMERA_MODE:
                return self.transition_to(SystemState.CAMERA_MODE, trigger="inspection_complete")
            else:
                return self.transition_to(SystemState.READY, trigger="inspection_complete")
        else:
            self._enter_error_state("Inspection failed")
            return False
    
    def reset_to_idle(self, clear_golden_sample: bool = True) -> bool:
        """Reset system to IDLE state."""
        if clear_golden_sample:
            self.set_golden_sample_loaded(False)
        self.set_source_configured(False)
        return self.transition_to(SystemState.IDLE, trigger="reset")
    
    def reset_to_ready(self) -> bool:
        """Reset to READY state (keeps golden sample)."""
        if not self._golden_sample_loaded:
            logger.error("Cannot reset to READY: No golden sample loaded")
            return False
        
        self.set_source_configured(False)
        return self.transition_to(SystemState.READY, trigger="reset")
    
    def reset_from_error(self) -> bool:
        """Recover from error state."""
        if self._current_state != SystemState.ERROR:
            return False
        
        # Return to READY if golden sample still loaded, else IDLE
        if self._golden_sample_loaded:
            return self.transition_to(SystemState.READY, trigger="error_recovery")
        else:
            return self.transition_to(SystemState.IDLE, trigger="error_recovery")
    
    def shutdown(self) -> bool:
        """Initiate system shutdown."""
        return self.transition_to(SystemState.SHUTDOWN, trigger="shutdown_request")
    
    def get_state_summary(self) -> str:
        """Get human-readable state summary."""
        lines = [
            f"Current State: {self._current_state.name}",
            f"Golden Sample: {'Loaded' if self._golden_sample_loaded else 'Not Loaded'}",
            f"Source: {'Configured' if self._source_configured else 'Not Configured'}"
        ]
        
        if self._current_state == SystemState.ERROR:
            lines.append(f"Error: {self._error_message}")
        
        if self._previous_state:
            lines.append(f"Previous State: {self._previous_state.name}")
        
        return "\n".join(lines)
    
    def get_transition_history(self, limit: int = 10) -> list:
        """Get recent state transition history."""
        return self._transition_history[-limit:]

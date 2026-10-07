import time
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class VersionState:
    system_prompt_version: str = "v1"
    catalogue_version: str = "v1"
    policy_version: str = "v1"
    last_bumped: Dict[str, float] = field(default_factory=dict)


class InvalidationManager:
    """
    Central source of truth for 'what version are we currently on' for
    each thing that can go stale. Bumping a version here is what actually
    changes the CacheContext fields fed into build_exact_key /
    build_semantic_namespace -- so invalidation happens by CONSTRUCTION,
    not by hunting down and deleting old entries.
    """
    def __init__(self):
        self.state = VersionState()

    def _bump(self, field_name: str) -> str:
        current = getattr(self.state, field_name)
        prefix, _, num = current.partition("v")
        new_version = f"v{int(num) + 1}"
        setattr(self.state, field_name, new_version)
        self.state.last_bumped[field_name] = time.time()
        return new_version

    def on_prompt_change(self) -> str:
        """Call this the moment the system prompt text changes."""
        return self._bump("system_prompt_version")

    def on_catalogue_reindex(self) -> str:
        """Call this after a product catalogue reindex -- answers that
        reference product availability/shipping may now be stale."""
        return self._bump("catalogue_version")

    def on_policy_change(self) -> str:
        """Call this when a support policy changes (e.g. return window)."""
        return self._bump("policy_version")

    def current_prompt_version(self) -> str:
        # NOTE: for simplicity here, prompt_version alone drives the
        # CacheContext field. In a system with catalogue/policy text
        # baked INTO the system prompt (the realistic case), bumping
        # either of those should also bump this. See combined_version().
        return self.state.system_prompt_version

    def combined_version(self) -> str:
        """
        If catalogue/policy facts are embedded in the system prompt text
        (typical for a support bot), ANY of the three changing means the
        effective prompt content changed. Combine all three into one
        version string so CacheContext picks up ALL of them.
        """
        return f"{self.state.system_prompt_version}-{self.state.catalogue_version}-{self.state.policy_version}"
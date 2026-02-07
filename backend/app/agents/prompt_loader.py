"""Loads and caches prompt templates from /prompt/ directory."""

import re
from pathlib import Path

from app.exceptions import PromptTemplateError


class PromptLoader:
    """
    Loads and caches prompt templates from /prompt/ directory.

    Uses sentinel-based substitution for {variable_name} placeholders that
    safely handles literal braces in JSON examples (escaped as {{ and }}).
    """

    _PLACEHOLDER_RE = re.compile(r"\{(\w+)\}")

    _LBRACE_SENTINEL = "\x00LBRACE\x00"
    _RBRACE_SENTINEL = "\x00RBRACE\x00"

    def __init__(self, prompt_dir: str | Path = "prompt") -> None:
        self._prompt_dir = Path(prompt_dir)
        self._cache: dict[str, str] = {}
        self._load_all()

    def _load_all(self) -> None:
        """Load every .txt file in prompt_dir into memory at startup."""
        if not self._prompt_dir.exists():
            raise PromptTemplateError(f"Prompt directory not found: {self._prompt_dir}")
        for txt_file in self._prompt_dir.glob("*.txt"):
            key = txt_file.stem
            self._cache[key] = txt_file.read_text()

    def get(self, template_name: str, **variables: str) -> str:
        """
        Return a prompt with {variable_name} placeholders substituted.

        Substitution rules:
        - {variable_name} -> replaced with provided value
        - {{literal_braces}} -> preserved as {literal_braces}
        """
        raw = self._cache.get(template_name)
        if raw is None:
            raise PromptTemplateError(f"Prompt template not found: {template_name}")

        # Step 1: Protect escaped braces
        temp = raw.replace("{{", self._LBRACE_SENTINEL).replace("}}", self._RBRACE_SENTINEL)

        # Step 2: Replace {variable} placeholders
        missing = []

        def _replacer(match: re.Match) -> str:
            key = match.group(1)
            if key in variables:
                return variables[key]
            missing.append(key)
            return match.group(0)

        result = self._PLACEHOLDER_RE.sub(_replacer, temp)
        if missing:
            raise PromptTemplateError(
                f"Missing variable(s) in prompt '{template_name}': {missing}"
            )

        # Step 3: Restore escaped braces
        result = result.replace(self._LBRACE_SENTINEL, "{").replace(self._RBRACE_SENTINEL, "}")
        return result

    @property
    def loaded_templates(self) -> list[str]:
        return list(self._cache.keys())

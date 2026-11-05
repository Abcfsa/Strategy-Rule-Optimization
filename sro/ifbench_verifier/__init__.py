"""Vendored IFBench instruction-following verifiers (Apache-2.0, Allen Institute for AI).

Source: https://github.com/allenai/IFBench  (`ifbench/` package)

Files `instructions.py`, `classic_instructions.py`, `instructions_registry.py`,
`instructions_util.py` are copied verbatim from the upstream `ifbench/` package
with two mechanical edits so they import as a self-contained subpackage:

  * `from ifbench import X`            -> `from . import X`
  * `from absl import logging`         -> `import logging`  (stdlib; API-compatible
                                                              for the `logging.error`
                                                              calls used here)

Everything else — the `INSTRUCTION_DICT` registry, each `*Checker` class, and the
nltk resource bootstrap in `instructions_util` — is unchanged from upstream, so
constraint semantics match the official IFBench evaluation exactly.

NOTE (optional heavy deps): importing `instructions_registry` transitively imports
`instructions` / `instructions_util`, which require `nltk`, `emoji`, `syllapy`,
`immutabledict`, `langdetect` (see requirements-if.txt). To keep `import sro.grading`
dependency-free, this package's `__init__` does NOT eagerly import those submodules;
consumers must import `sro.ifbench_verifier.instructions_registry` lazily (inside the
judger function) and handle ImportError. The upstream `download_nltk_resources()`
runs on first import of `instructions_util` and will fetch punkt/stopwords if absent.
"""

__all__ = [
    "classic_instructions",
    "instructions",
    "instructions_registry",
    "instructions_util",
]

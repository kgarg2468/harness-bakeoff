import os

from bakeoff.shared import netguard

# I6: nothing in the test suite may leave 127.0.0.1.
netguard.install()
os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")
# Never let a stray shell variable point a client at a real endpoint or key.
for _name in list(os.environ):
    if _name.startswith(
        ("OPENAI_", "OPENROUTER_", "ANTHROPIC_", "ROCKETRIDE_")
    ) or _name.upper().endswith("_PROXY"):
        del os.environ[_name]

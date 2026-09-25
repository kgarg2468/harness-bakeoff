#!/usr/bin/env bash
# Does each loop's dependency set fit into the RocketRide engine's single Python environment?
#
# The engine compiles every requirements*.txt (core + all nodes) into one constraint set. The
# binding constraint today is the crewai node: crewai>=1.14.1,<2 requires openai<3.
# Expected: our_version fits; pydantic_version (2.31.1) fits; pydantic-ai >= 2.32 does not.
set -uo pipefail
cd "$(dirname "$0")/.."
ENGINE_REQS=(
  "crewai>=1.14.1,<2"   # nodes/src/nodes/agent_crewai/requirements.txt (rocketride-server develop)
  "pydantic==2.12.5"    # packages/ai/src/ai/requirements.txt
  "httpx==0.28.1"
)
# The project's own declared dependency sets, read from pyproject.toml so they never drift.
mapfile -t CORE_DEPS < <(python3 -c 'import tomllib; print("\n".join(tomllib.load(open("pyproject.toml","rb"))["project"]["dependencies"]))')
mapfile -t PAI_DEPS < <(python3 -c 'import tomllib; print("\n".join(tomllib.load(open("pyproject.toml","rb"))["project"]["optional-dependencies"]["pydantic"]))')
if [[ ${#CORE_DEPS[@]} -eq 0 || ${#PAI_DEPS[@]} -eq 0 ]]; then
  echo "could not read dependencies from pyproject.toml" >&2; exit 2
fi

check() {
  local label="$1" expect="$2"; shift 2
  local dir; dir="$(mktemp -d)"
  printf '%s\n' "${ENGINE_REQS[@]}" "$@" > "$dir/requirements.in"
  if uv pip compile --quiet --python-version 3.12 "$dir/requirements.in" -o "$dir/out.txt" 2>"$dir/err"; then
    got=fits
  else
    got=conflict
  fi
  printf '%-45s expected=%-8s got=%s\n' "$label" "$expect" "$got"
  if [[ "$got" == conflict ]]; then
    # Show why, so a conflict for any reason other than openai is visible in CI logs.
    grep -m3 -iE "openai|because" "$dir/err" | sed 's/^/    /'
  fi
  rm -rf "$dir"
  [[ "$got" == "$expect" ]]
}

# Pin the exact latest release; an unpinned spec would just backtrack to 2.31.1.
latest="$(curl -fsSL https://pypi.org/pypi/pydantic-ai-slim/json \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["info"]["version"])')"
if [[ ! "$latest" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "could not determine the latest pydantic-ai-slim release (got: '$latest')" >&2; exit 2
fi

status=0
check "our_version (declared deps)"                fits     "${CORE_DEPS[@]}"                                            || status=1
check "pydantic_version (declared deps, 2.31.1)"   fits     "${CORE_DEPS[@]}" "${PAI_DEPS[@]}"                           || status=1
check "pydantic-ai 2.32.0 (needs openai 3)"        conflict "${CORE_DEPS[@]}" "pydantic-ai-slim[openai,openrouter]==2.32.0" || status=1
check "pydantic-ai latest ($latest)"               conflict "${CORE_DEPS[@]}" "pydantic-ai-slim[openai,openrouter]==$latest" || status=1
exit $status

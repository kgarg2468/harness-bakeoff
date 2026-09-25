# Third-party notices

This project is MIT-licensed (see `LICENSE`). It includes material derived from the projects
below. Each ported file names its source at the top.

## RocketRide (MIT)

`data/rocketride/` is generated from node definitions and example pipelines in
[rocketride-org/rocketride-server](https://github.com/rocketride-org/rocketride-server).
Tool names, descriptions and argument schemas in `src/bakeoff/shared/tools/engine_tools.py`
follow the engine's MCP tool registry.

```
MIT License

Copyright (c) 2026 Aparavi Software AG
```
<!-- Ported sources (Pi, OpenCode, ...) are appended below with their license text. -->

## Pi (MIT)

`src/bakeoff/our_version/compat.py` and `src/bakeoff/our_version/retry.py` are ported (TypeScript
to Python) from [earendil-works/pi](https://github.com/earendil-works/pi) at commit
`5fd446ca1843682e8da3fec4ceb71c42f56fbace`:

- `packages/ai/src/api/openai-completions.ts`: endpoint compat flags, request parameters,
  `reasoning_details` merging, usage parsing
- `packages/ai/src/utils/retry.ts`, `packages/ai/src/utils/provider-retry.ts`: retryable-error
  classification, `Retry-After` parsing, backoff
- `packages/ai/src/utils/overflow.ts`: context-overflow detection

```
MIT License

Copyright (c) 2025 Mario Zechner

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

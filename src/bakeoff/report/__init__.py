"""The comparison report: one self-contained HTML page built from the run outputs.

- `data`: reads `out/runs/<run>`, `out/live/*` and `out/metrics.json` into JSON-ready dicts
  (scenario results, replay timelines, wire recordings), tolerating anything that is missing.
- `render`: turns that data into the HTML sections (scorecard, matrix, live runs, ...).
- `build`: puts the page together from `template.html`, `style.css` and `report.js`, and is
  the `bakeoff report` command (`build.main(argv)`).

Submodules are not imported here, so `python -m bakeoff.report.build` runs the build once.
"""

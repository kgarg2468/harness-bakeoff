"use strict";
// harness-bakeoff report: tooltips, the side-by-side replay and the wire diff.
// All data comes from the #report-data JSON; recorded text is only ever set via textContent.
(() => {
  const D = JSON.parse(document.getElementById("report-data").textContent);
  const SVG = "http://www.w3.org/2000/svg";
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

  function h(tag, attrs = {}, ...kids) {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (v == null || v === false) continue;
      if (k === "text") el.textContent = v;
      else if (k === "class") el.className = v;
      else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
      else el.setAttribute(k, v === true ? "" : v);
    }
    for (const kid of kids.flat()) if (kid != null && kid !== false) el.append(kid);
    return el;
  }
  function s(tag, attrs = {}) {
    const el = document.createElementNS(SVG, tag);
    for (const [k, v] of Object.entries(attrs)) if (v != null) el.setAttribute(k, v);
    return el;
  }
  const fmtMs = (ms) =>
    ms == null ? "n/a" : ms >= 1000 ? `${(ms / 1000).toFixed(2)} s` : ms < 10 ? `${ms.toFixed(2)} ms` : `${ms.toFixed(1)} ms`;
  const fmtInt = (n) => (n == null ? "n/a" : Math.round(n).toLocaleString("en-US"));
  // A glossary term drawn by this script: the same dotted underline and tooltip as the page's.
  function term(key, text) {
    const [label, definition] = (D.terms || {})[key] || [key, ""];
    return h("span", { class: "t", tabindex: 0, "data-term": label, "data-tip": definition, text: text ?? label });
  }

  // ---- tooltips: any element with data-tip (and an optional data-term title) -------------
  const tip = $("#tip");
  let tipFor = null;
  function place(x, y) {
    const r = tip.getBoundingClientRect();
    let left = x + 14;
    let top = y + 16;
    if (left + r.width > innerWidth - 8) left = Math.max(8, x - r.width - 14);
    if (top + r.height > innerHeight - 8) top = Math.max(8, y - r.height - 12);
    tip.style.left = `${left}px`;
    tip.style.top = `${top}px`;
  }
  function showTip(el, x, y) {
    const text = el.getAttribute("data-tip");
    if (!text) return;
    tip.replaceChildren();
    const term = el.getAttribute("data-term");
    if (term) tip.append(h("b", { text: term }));
    tip.append(document.createTextNode(text));
    tip.classList.add("show");
    tipFor = el;
    place(x, y);
  }
  function hideTip() {
    tip.classList.remove("show");
    tipFor = null;
  }
  document.addEventListener("pointerover", (e) => {
    const el = e.target.closest && e.target.closest("[data-tip]");
    if (el) showTip(el, e.clientX, e.clientY);
    else hideTip();
  });
  document.addEventListener("pointermove", (e) => tipFor && place(e.clientX, e.clientY));
  document.addEventListener("focusin", (e) => {
    const el = e.target.closest && e.target.closest("[data-tip]");
    if (!el) return;
    const r = el.getBoundingClientRect();
    showTip(el, r.left, r.bottom);
  });
  document.addEventListener("focusout", hideTip);
  document.addEventListener("keydown", (e) => e.key === "Escape" && hideTip());

  // ---- expand controls for long texts (server- and client-rendered) ------------------------
  function expandable(text, preview = 600) {
    if (!text) return h("div", { class: "txt muted", text: "(empty)" });
    if (text.length <= preview) return h("div", { class: "txt", text });
    const label = `show all (${text.length.toLocaleString("en-US")} characters)`;
    return h(
      "div",
      { class: "txt exp" },
      h("span", { class: "pv", text: `${text.slice(0, preview)}…` }),
      h("span", { class: "full", hidden: true, text }),
      h("button", { type: "button", class: "more", "data-label": label, text: label }),
    );
  }
  document.addEventListener("click", (e) => {
    const btn = e.target.closest("button.more");
    if (!btn) return;
    const box = btn.parentElement;
    const full = $(".full", box);
    const open = full.hidden;
    full.hidden = !open;
    $(".pv", box).hidden = open;
    btn.textContent = open ? "show less" : btn.getAttribute("data-label");
  });

  const scenarios = D.scenarios || [];
  if (!scenarios.length) return;
  const byId = Object.fromEntries(scenarios.map((x) => [x.id, x]));
  const st = { scn: scenarios[0].id, left: "our", right: "pydantic", cursor: Infinity, span: 1, req: 1, mode: "aligned", all: false };
  const STATUS = { pass: ["✓", "PASS"], fail: ["✕", "FAIL"], xfail: ["≈", "XFAIL"], xpass: ["!", "XPASS"], none: ["–", "no run"] };

  function loopTag(impl) {
    const info = D.loops[impl] || { letter: "?", label: impl, color: "grey" };
    return h("span", { class: `lp ${info.color}` }, h("i"), h("b", { text: info.letter }), ` ${info.label}`);
  }
  function badge(status) {
    const [icon, label] = STATUS[status] || STATUS.none;
    return h("span", { class: `st ${status}`, text: `${icon} ${label}` });
  }

  // ---- replay -----------------------------------------------------------------------------
  const W = 600;
  const LEFT = 58;
  const PW = W - LEFT - 8;
  const END = {
    ok: "a complete response", cut: "cut short by a cancel", retry: "failed; retried", error: "an error",
    cancel: "cancelled", killed: "the process was killed", open: "no response saved",
  };
  const LANE_TIP = {
    model: "One bar per model request (a step): pale while waiting for the first token, solid while the answer streams in.",
    tools: "One bar per tool run, from start to end. Tools that run at the same time get their own rows.",
    approval: "◆ the loop asked for an approval; ● the user's answer (green: allowed, red: denied).",
    git: "● a git commit after a finished turn; ■ a compaction (the history is replaced by a summary).",
  };
  const GLYPH = { retry: ["↻", "var(--warn)"], error: ["✕", "var(--bad)"], cancel: ["■", "var(--ink)"], killed: ["✕", "var(--bad)"], cut: ["✂", "var(--ink)"] };

  function prepare(replay, gap) {
    const off = [];
    let t = 0;
    for (const turn of replay.turns) {
      off.push(t);
      t += turn.ms + gap;
    }
    return { replay, off, gap, span: Math.max(0, t - gap), abs: (ti, ms) => (off[ti] || 0) + ms };
  }

  function lanes(lay, span, color) {
    const x = (t) => LEFT + (t / span) * PW;
    const turns = lay.replay.turns;
    const toolRows = Math.max(1, ...turns.map((t) => t.rows || 0));
    const yModel = 16;
    const yTools = yModel + 20;
    const hTools = toolRows * 9 + 3;
    const yPerm = yTools + hTools + 6;
    const yGit = yPerm + 20;
    const yAxis = yGit + 18;
    const svg = s("svg", { class: "lanes", viewBox: `0 0 ${W} ${yAxis + 16}`, role: "img", "aria-label": "timeline" });
    const fill = `var(--${color})`;
    for (const [label, y, hh] of [["model", yModel, 14], ["tools", yTools, hTools], ["approval", yPerm, 14], ["git", yGit, 14]]) {
      svg.append(s("rect", { class: "lanebg", x: LEFT, y, width: PW, height: hh }));
      const text = s("text", { x: LEFT - 6, y: y + 10, "text-anchor": "end", "data-tip": LANE_TIP[label] });
      text.textContent = label;
      svg.append(text);
    }
    turns.forEach((turn, i) => {
      if (i > 0) {
        const cut = s("rect", { class: "cut", x: x(lay.off[i] - lay.gap), y: yModel, width: Math.max(1, x(lay.off[i]) - x(lay.off[i] - lay.gap)), height: yAxis - yModel });
        cut.setAttribute("data-tip", "Idle time between turns, cut from the axis (the same for both loops).");
        svg.append(cut, s("line", { class: "turnsep", x1: x(lay.off[i]), x2: x(lay.off[i]), y1: 4, y2: yAxis }));
      }
      // The full label only where it fits before the next turn; the number otherwise.
      const full = `${i + 1} · ${turn.kind}${turn.stop ? ` → ${turn.stop}` : ""}`;
      const room = (i + 1 < turns.length ? x(lay.off[i + 1]) : LEFT + PW) - x(lay.off[i]);
      const label = s("text", { x: x(lay.off[i]) + 2, y: 11, "data-tip": `turn ${full}` });
      label.textContent = full.length * 6 < room ? full : String(i + 1);
      svg.append(label);
    });
    for (const seg of lay.replay.lanes.model) {
      const t0 = lay.abs(seg.turn, seg.t0);
      const t1 = lay.abs(seg.turn, seg.t1);
      const first = seg.first == null ? t1 : lay.abs(seg.turn, seg.first);
      const u = seg.usage;
      const cost = u && u.cost_usd != null && u.cost_source !== "none" ? ` · $${u.cost_usd.toFixed(4)} (${u.cost_source})` : "";
      const usage = u ? `\n${fmtInt(u.input_tokens)} in / ${fmtInt(u.output_tokens)} out tokens${cost}` : "";
      const g = s("g", {
        "data-tip": `step ${seg.step ?? "?"}${seg.attempt > 1 ? `, attempt ${seg.attempt}` : ""}\nwaited ${fmtMs(first - t0)} for the first token, streamed ${fmtMs(t1 - first)}\nended with ${END[seg.end] || seg.end}${usage}`,
      });
      g.append(s("rect", { x: x(t0), y: yModel + 1, width: Math.max(1.2, x(first) - x(t0)), height: 12, style: `fill:${fill};fill-opacity:.25` }));
      if (seg.first != null) g.append(s("rect", { x: x(first), y: yModel + 1, width: Math.max(1.2, x(t1) - x(first)), height: 12, style: `fill:${fill}` }));
      if (GLYPH[seg.end]) {
        const mark = s("text", { x: x(t1) + 1.5, y: yModel + 11, style: `fill:${GLYPH[seg.end][1]};font-weight:800` });
        mark.textContent = GLYPH[seg.end][0];
        g.append(mark);
      }
      svg.append(g);
    }
    for (const seg of lay.replay.lanes.tools) {
      const t0 = lay.abs(seg.turn, seg.t0);
      const t1 = lay.abs(seg.turn, seg.t1);
      const state = seg.ok ? "ok" : seg.ok === false ? "failed" : "never finished: its process died";
      const style = seg.ok ? `fill:${fill};fill-opacity:.55` : seg.ok === false ? "fill:var(--bad);fill-opacity:.75" : "fill:var(--grey)";
      svg.append(s("rect", {
        x: x(t0), y: yTools + 2 + (seg.row || 0) * 9, width: Math.max(1.5, x(t1) - x(t0)), height: 7, rx: 1.5, style,
        "data-tip": `${seg.name} (${seg.call})\nran ${fmtMs(t1 - t0)} · ${state}${seg.eager ? "\nread-only, started while the model was still streaming (eager)" : ""}${seg.late ? "\nstarted after the loop's turn.end (late)" : ""}`,
      }));
    }
    for (const p of lay.replay.lanes.perm) {
      const cx = x(lay.abs(p.turn, p.t));
      const cy = yPerm + 7;
      const what = p.kind === "asked" ? "approval asked for" : p.kind === "allow" ? "user allowed" : "user denied";
      const tipText = `${what} ${p.name || "a tool"} (${p.call})`;
      svg.append(p.kind === "asked"
        ? s("path", { d: `M${cx},${cy - 5}L${cx + 5},${cy}L${cx},${cy + 5}L${cx - 5},${cy}Z`, style: "fill:var(--warn)", "data-tip": tipText })
        : s("circle", { cx, cy, r: 4, style: `fill:var(--${p.kind === "allow" ? "ok" : "bad"})`, "data-tip": tipText }));
    }
    for (const g of lay.replay.lanes.git) {
      const cx = x(lay.abs(g.turn, g.t));
      const cy = yGit + 7;
      if (g.kind === "compact") {
        svg.append(s("rect", { x: cx - 3.5, y: cy - 3.5, width: 7, height: 7, style: "fill:var(--muted)", "data-tip": "compaction: the history before this point is replaced by a summary (no commit)" }));
      } else {
        const revert = turns[g.turn] && turns[g.turn].kind === "revert" ? "revert " : "";
        svg.append(s("circle", { cx, cy, r: 4, style: "fill:var(--ink)", "data-tip": `${revert}commit ${String(g.sha || "").slice(0, 7)}\nfiles: ${(g.files || []).join(", ") || "none"}` }));
      }
    }
    svg.append(s("line", { class: "axis", x1: LEFT, x2: LEFT + PW, y1: yAxis, y2: yAxis }));
    const raw = span / 4;
    const mag = 10 ** Math.floor(Math.log10(raw || 1));
    const step = [1, 2, 5, 10].map((m) => m * mag).find((v) => v >= raw) || raw;
    for (let v = 0; v <= span + 1e-9; v += step) {
      svg.append(s("line", { class: "axis", x1: x(v), x2: x(v), y1: yAxis, y2: yAxis + 3 }));
      const t = s("text", { x: x(v), y: yAxis + 13, "text-anchor": v === 0 ? "start" : "middle" });
      t.textContent = v === 0 ? "0" : v >= 1000 ? `${+(v / 1000).toPrecision(3)} s` : `${+v.toPrecision(3)} ms`;
      svg.append(t);
    }
    const cursor = s("line", { class: "cursor", x1: x(span), x2: x(span), y1: 4, y2: yAxis });
    svg.append(cursor);
    return { svg, cursor, x };
  }

  function pre(text, preview = 600) {
    const box = expandable(text, preview);
    box.classList.add("mono");
    return h("pre", {}, box);
  }
  const TITLES = {
    user: "User", note: "Harness note", summary: "Compaction summary", assistant: "Model", result: "Tool result",
    ask: "Approval asked", resume: "Resumed", retry: "Retry", error: "Error", end: "Turn ended", commit: "Commit",
    crash: "Worker killed",
  };
  const STOP_TAG = { end_turn: "ok", error: "bad" };
  // A tool result's origin: [tag class, label, what it means].
  const RESULT = {
    ok: ["ok", "ok", "The tool ran and succeeded."],
    failed: ["bad", "failed", "The tool ran and reported an error; the model sees the message."],
    denied: ["warn", "denied", "Denied by the user or by a permission rule: the tool changed nothing."],
    unfinished: ["bad", "never finished", "The tool started, but its process died before it ended."],
    "not run": ["nt", "not run", "The loop wrote this result without running the tool, e.g. because the turn reached its step limit or was cancelled."],
  };
  // Card titles that are glossary terms.
  const TITLE_TERM = { result: "tool-result", ask: "approval", retry: "retry", commit: "commit", end: "turn" };

  function cardBody(c) {
    switch (c.k) {
      case "assistant": {
        const parts = [];
        if (c.incomplete) parts.push(h("span", { class: "tag warn", text: "cut short" }));
        if (c.reasoning) parts.push(h("details", {}, h("summary", { text: "reasoning" }), pre(c.reasoning)));
        if (c.text) parts.push(expandable(c.text));
        for (const call of c.calls || []) {
          parts.push(h("div", {}, "→ ", h("b", { text: call.name || "?" }), h("span", { class: "muted", text: ` ${call.id}` })), pre(call.args, 300));
        }
        return parts;
      }
      case "result": {
        const [cls, label, meaning] = RESULT[c.state] || ["nt", c.state || "?", ""];
        return [h("span", { class: `tag ${cls}`, tabindex: 0, "data-tip": meaning, text: label }), pre(c.text, 400)];
      }
      case "ask":
        return [pre(c.args, 300)];
      case "resume":
        if (c.kind === "crash") return [h("div", {}, term("crash-resume", "Continued"), " from the saved history after the worker was killed.")];
        return [
          h("div", {}, ...Object.entries(c.decisions || {}).map(([id, d]) => h("span", { class: `tag ${d === "allow" ? "ok" : "bad"}`, text: `${id}: ${d}` }))),
          c.reason ? h("div", { text: `reason: ${c.reason}` }) : null,
        ];
      case "retry":
        return [h("div", { text: `attempt ${c.attempt ?? "?"} · ${c.status ? `HTTP ${c.status} · ` : ""}waited ${fmtMs(c.wait_ms)}` }), c.reason ? h("div", { class: "muted", text: c.reason }) : null];
      case "error":
        return [h("div", { class: "muted", text: c.kind || "" }), expandable(c.text || "")];
      case "end": {
        const u = c.usage || {};
        const cost = u.cost_usd ? ` · $${u.cost_usd.toFixed(4)}` : "";
        const [stopLabel, stopTip] = (D.terms || {}).stop || ["stop reason", ""];
        return [
          h("span", { class: `tag ${STOP_TAG[c.stop] || "warn"}`, tabindex: 0, "data-term": stopLabel, "data-tip": stopTip, text: c.stop || "?" }),
          h("span", { class: "muted" }, " ", term("step", `${c.steps ?? "?"} step${c.steps === 1 ? "" : "s"}`), " · ", term("tokens", `${fmtInt(u.input_tokens)} in / ${fmtInt(u.output_tokens)} out tokens`), cost),
          c.pending && c.pending.length ? h("div", { text: `waiting for: ${c.pending.join(", ")}` }) : null,
          c.error ? h("div", { class: "bad-t", text: c.error }) : null,
        ];
      }
      case "commit":
        return [h("div", {}, h("code", { text: String(c.sha || "").slice(0, 7) }), ` ${(c.files || []).join(", ") || "no file changes"}`)];
      case "crash":
        return [h("div", { text: "The worker process was killed here (crash test). The turn never ended; the next turn resumes it." })];
      default:
        return [expandable(c.text || "")];
    }
  }

  function cardEl(c, t) {
    const name = TITLES[c.k] || c.k;
    const title = h("b", {}, TITLE_TERM[c.k] ? term(TITLE_TERM[c.k], name) : name, c.k === "result" || c.k === "ask" ? ` · ${c.name || "?"}` : null);
    return h(
      "div",
      { class: `card ${c.k}` },
      h("div", { class: "ch" }, title, h("span", { text: `+${fmtMs(t)}` }), h("span", { text: `turn ${c.turn + 1}` })),
      h("div", { class: "body" }, cardBody(c)),
    );
  }

  function column(impl, run, lay, span) {
    const info = D.loops[impl] || { color: "grey", label: impl };
    const col = h("div", { class: "col", "data-impl": impl });
    const head = h("div", { class: "hd" }, loopTag(impl));
    col.append(head);
    if (!run) {
      col.append(h("div", { class: "nodata", text: `No run of ${info.label} for this scenario here.` }));
      return col;
    }
    head.append(badge(run.status));
    for (const name of ["I1", "I2", "I3", "I5", "I7"]) {
      const v = (run.invariants || {})[name];
      const cls = v == null ? "na" : v.ok ? "ok" : "bad";
      const [label, definition] = (D.terms || {})[name] || [name, ""];
      head.append(h("span", {
        class: `inv ${cls}`, tabindex: 0, "data-term": label,
        "data-tip": `${definition}\n\n${v ? v.detail : "not checked in this run"}`,
        text: `${v == null ? "–" : v.ok ? "✓" : "✕"} ${name}`,
      }));
    }
    const checks = Object.values(run.expect || {});
    const okCount = checks.filter((c) => c.ok).length;
    const stops = (run.stops || []).join(", ");
    col.append(h("div", { class: "why", text: `${run.reason}${stops ? ` · stops: ${stops}` : ""} · checks ${okCount}/${checks.length}` }));
    if (!lay) {
      col.append(h("div", { class: "nodata", text: "No session log or event file: nothing to replay." }));
      return col;
    }
    const view = lanes(lay, span, info.color);
    const list = h("div", { class: "cards" });
    list.style.position = "relative";
    const cards = lay.replay.cards.map((c) => {
      const t = lay.abs(c.turn, c.t);
      const el = cardEl(c, t);
      el.addEventListener("click", (e) => {
        if (e.target.closest("button, a, summary, pre")) return;
        stopPlay();
        setCursor(t, true);
      });
      list.append(el);
      return { el, t };
    });
    col.append(view.svg, list);
    col._view = { ...view, list, cards };
    return col;
  }

  function renderReplay() {
    const box = $("#rp-cols");
    if (!box) return;
    const scn = byId[st.scn];
    const sides = [st.left, st.right].map((impl) => ({ impl, run: scn.runs[impl] }));
    const raw = sides.map((x) => (x.run && x.run.replay ? x.run.replay.turns.reduce((a, t) => a + t.ms, 0) : 0));
    const gap = Math.max(2, 0.04 * Math.max(...raw, 1));
    for (const x of sides) x.lay = x.run && x.run.replay ? prepare(x.run.replay, gap) : null;
    st.span = Math.max(1, ...sides.map((x) => (x.lay ? x.lay.span : 0)));
    const times = new Set();
    for (const x of sides) if (x.lay) for (const c of x.lay.replay.cards) times.add(x.lay.abs(c.turn, c.t));
    st.times = [...times].sort((a, b) => a - b);
    box.replaceChildren(...sides.map((x) => column(x.impl, x.run, x.lay, st.span)));
    setCursor(Math.min(st.cursor, st.span), false);
  }

  function setCursor(t, scroll) {
    st.cursor = Math.max(0, Math.min(t, st.span));
    const range = $("#rp-range");
    if (range) range.value = String(Math.round((st.cursor / st.span) * 1000));
    const readout = $("#rp-time");
    if (readout) readout.textContent = `${fmtMs(st.cursor)} of ${fmtMs(st.span)}`;
    for (const col of $$("#rp-cols .col")) {
      const v = col._view;
      if (!v) continue;
      const x = v.x(st.cursor);
      v.cursor.setAttribute("x1", x);
      v.cursor.setAttribute("x2", x);
      let now = null;
      for (const card of v.cards) {
        const future = card.t > st.cursor + 1e-6;
        card.el.classList.toggle("future", future);
        card.el.classList.remove("now");
        if (!future) now = card;
      }
      if (now) {
        now.el.classList.add("now");
        const top = now.el.offsetTop;
        if (scroll && (top < v.list.scrollTop || top > v.list.scrollTop + v.list.clientHeight - now.el.offsetHeight)) {
          v.list.scrollTop = Math.max(0, top - 40);
        }
      }
    }
  }

  let raf = 0;
  let last = 0;
  function tick(now) {
    const dt = now - last;
    last = now;
    setCursor(st.cursor + (dt * st.span) / 8000, true); // the whole replay plays in 8 s
    if (st.cursor >= st.span) stopPlay();
    else raf = requestAnimationFrame(tick);
  }
  function startPlay() {
    if (st.cursor >= st.span - 1e-6) setCursor(0, true);
    st.playing = true;
    $("#rp-play").textContent = "❚❚ Pause";
    last = performance.now();
    raf = requestAnimationFrame(tick);
  }
  function stopPlay() {
    st.playing = false;
    cancelAnimationFrame(raf);
    const btn = $("#rp-play");
    if (btn) btn.textContent = "▶ Play";
  }

  // ---- wire diff ---------------------------------------------------------------------------
  function decode(req) {
    if (!req) return undefined;
    if ("raw" in req) return req.raw;
    const body = req.body;
    if (!body || typeof body !== "object" || Array.isArray(body)) return body;
    const out = {};
    for (const [k, v] of Object.entries(body)) {
      if (k === "messages" && v && Array.isArray(v.$refs)) out[k] = v.$refs.map((i) => D.pool[i]);
      else if (k === "tools" && v && typeof v.$ref === "number") out[k] = D.pool[v.$ref];
      else out[k] = v;
    }
    return out;
  }
  const sortKeys = (v) =>
    Array.isArray(v) ? v.map(sortKeys)
      : v && typeof v === "object" ? Object.fromEntries(Object.keys(v).sort().map((k) => [k, sortKeys(v[k])]))
        : v;
  // "aligned": both bodies list their top-level fields in the same order (left first, then
  // the right's extra ones), so a different field order does not hide the real differences.
  function arrange(v, other) {
    if (st.mode === "sorted") return sortKeys(v);
    if (st.mode === "sent" || !v || typeof v !== "object" || Array.isArray(v) || !other || typeof other !== "object") return v;
    const order = [...Object.keys(st.leftBody || {}), ...Object.keys(v)];
    return Object.fromEntries([...new Set(order)].filter((k) => k in v).map((k) => [k, v[k]]));
  }
  function pretty(v, other) {
    if (v === undefined) return [];
    if (typeof v === "string") return v.split("\n");
    return JSON.stringify(arrange(v, other), null, 2).split("\n");
  }

  function pairUp(ops) {
    // A run of deletions next to a run of insertions becomes rows of changed lines.
    const out = [];
    let k = 0;
    while (k < ops.length) {
      if (ops[k][0] === "=") {
        out.push(ops[k++]);
        continue;
      }
      const dels = [];
      const adds = [];
      while (k < ops.length && ops[k][0] !== "=") (ops[k][0] === "-" ? dels : adds).push(ops[k++]);
      for (let i = 0; i < Math.max(dels.length, adds.length); i++) {
        if (i < dels.length && i < adds.length) out.push(["~", dels[i][1], adds[i][2]]);
        else out.push(i < dels.length ? dels[i] : adds[i]);
      }
    }
    return out;
  }
  function diffLines(a, b) {
    let p = 0;
    while (p < a.length && p < b.length && a[p] === b[p]) p++;
    let q = 0;
    while (q < a.length - p && q < b.length - p && a[a.length - 1 - q] === b[b.length - 1 - q]) q++;
    const A = a.slice(p, a.length - q);
    const B = b.slice(p, b.length - q);
    const ops = [];
    for (let i = 0; i < p; i++) ops.push(["=", i, i]);
    const mid = [];
    if (A.length * B.length > 4e6) {
      // Too large for an exact diff: compare line by line.
      for (let i = 0; i < Math.max(A.length, B.length); i++) {
        if (i < A.length && i < B.length) mid.push(A[i] === B[i] ? ["=", p + i, p + i] : ["~", p + i, p + i]);
        else mid.push(i < A.length ? ["-", p + i, null] : ["+", null, p + i]);
      }
      ops.push(...mid);
    } else {
      const n = A.length;
      const m = B.length;
      const w = m + 1;
      const L = new Uint32Array((n + 1) * w);
      for (let i = n - 1; i >= 0; i--) {
        for (let j = m - 1; j >= 0; j--) L[i * w + j] = A[i] === B[j] ? L[(i + 1) * w + j + 1] + 1 : Math.max(L[(i + 1) * w + j], L[i * w + j + 1]);
      }
      let i = 0;
      let j = 0;
      while (i < n && j < m) {
        if (A[i] === B[j]) mid.push(["=", p + i++, p + j++]);
        else if (L[(i + 1) * w + j] >= L[i * w + j + 1]) mid.push(["-", p + i++, null]);
        else mid.push(["+", null, p + j++]);
      }
      while (i < n) mid.push(["-", p + i++, null]);
      while (j < m) mid.push(["+", null, p + j++]);
      ops.push(...pairUp(mid));
    }
    for (let i = 0; i < q; i++) ops.push(["=", a.length - q + i, b.length - q + i]);
    return ops;
  }

  function lineCell(text, n, cls) {
    const d = h("div", { class: cls || null, "data-n": n == null ? "" : n + 1 });
    if (text != null) d.textContent = text;
    return d;
  }
  function markedCell(x, y, n, cls) {
    let p = 0;
    while (p < x.length && p < y.length && x[p] === y[p]) p++;
    let q = 0;
    while (q < x.length - p && q < y.length - p && x[x.length - 1 - q] === y[y.length - 1 - q]) q++;
    const d = h("div", { class: cls, "data-n": n + 1 });
    d.append(x.slice(0, p), h("mark", { text: x.slice(p, x.length - q) }), x.slice(x.length - q));
    return d;
  }
  function diffRow(op, a, b) {
    const [kind, i, j] = op;
    if (kind === "=") return h("div", { class: "row" }, lineCell(a[i], i), lineCell(b[j], j));
    if (kind === "-") return h("div", { class: "row" }, lineCell(a[i], i, "del"), lineCell(null, null, "gone"));
    if (kind === "+") return h("div", { class: "row" }, lineCell(null, null, "gone"), lineCell(b[j], j, "add"));
    return h("div", { class: "row chg" }, markedCell(a[i], b[j], i, "del"), markedCell(b[j], a[i], j, "add"));
  }

  const CONTEXT = 3;
  function diffRows(ops, a, b) {
    const rows = [];
    let k = 0;
    while (k < ops.length) {
      let e = k;
      while (e < ops.length && ops[e][0] === "=") e++;
      const run = e - k;
      if (run > 2 * CONTEXT + 3 && !st.all) {
        const head = k === 0 ? 0 : CONTEXT;
        const tail = e === ops.length ? 0 : CONTEXT;
        for (let i = k; i < k + head; i++) rows.push(diffRow(ops[i], a, b));
        const hidden = ops.slice(k + head, e - tail);
        const fold = h("button", { type: "button", class: "fold", text: `⋯ ${hidden.length} identical lines (show)` });
        fold.addEventListener("click", () => fold.replaceWith(...hidden.map((op) => diffRow(op, a, b))));
        rows.push(fold);
        for (let i = e - tail; i < e; i++) rows.push(diffRow(ops[i], a, b));
      } else {
        for (let i = k; i < e; i++) rows.push(diffRow(ops[i], a, b));
      }
      if (e < ops.length) rows.push(diffRow(ops[e], a, b));
      k = e + 1;
    }
    return rows;
  }

  function wireMeta(impl, run, req) {
    if (!run) return [loopTag(impl), h("div", { class: "muted", text: "no run for this scenario" })];
    if (!req) return [loopTag(impl), h("div", { class: "muted", text: `sent no request #${st.req}` })];
    const m = req.meta || {};
    const bits = [`#${req.n}`, `${fmtInt(req.bytes)} bytes`];
    if (m.status != null) bits.push(`HTTP ${m.status}`);
    if (m.conn_id != null) bits.push(`connection ${m.conn_id}`);
    if (m.t_us != null) bits.push(`server clock ${fmtMs(m.t_us / 1000)}`);
    const cut = req.clipped;
    return [
      loopTag(impl),
      h("div", { class: "mono small", text: bits.join(" · ") }),
      m.error ? h("div", { class: "bad-t small", text: `rejected by the fake server: ${m.error}` }) : null,
      run.byte_prefix == null ? null : h("div", { class: "small muted" }, term("byte-prefix"), ` over all requests: ${run.byte_prefix ? "yes" : "no"}`),
      cut ? h("div", { class: "small muted", text: `${cut.strings} long string${cut.strings === 1 ? "" : "s"} cut here (${fmtInt(cut.chars)} characters not shown); the full body is in ${req.file}` }) : null,
    ];
  }

  function wireSummary(lv, rv) {
    const pills = [];
    const name = (impl) => (D.loops[impl] || { letter: impl }).letter;
    if (lv === undefined || rv === undefined) {
      if (lv !== undefined || rv !== undefined) pills.push(["", `only ${name(lv === undefined ? st.right : st.left)} sent request #${st.req}`]);
      return pills;
    }
    if (JSON.stringify(lv) === JSON.stringify(rv)) return [["ok", "identical JSON, same key order"]];
    if (JSON.stringify(sortKeys(lv)) === JSON.stringify(sortKeys(rv))) pills.push(["", "same content, different key order"]);
    if (lv && rv && typeof lv === "object" && typeof rv === "object") {
      const common = (x, y) => Object.keys(x).filter((k) => k in y).join();
      if (common(lv, rv) !== common(rv, lv)) pills.push(["", "top-level fields in a different order"]);
    }
    if (lv && rv && typeof lv === "object" && typeof rv === "object") {
      const lk = Object.keys(lv);
      const rk = Object.keys(rv);
      const onlyL = lk.filter((k) => !(k in rv));
      const onlyR = rk.filter((k) => !(k in lv));
      const diff = lk.filter((k) => k in rv && JSON.stringify(sortKeys(lv[k])) !== JSON.stringify(sortKeys(rv[k])));
      if (onlyL.length) pills.push(["bad", `only ${name(st.left)} sends: ${onlyL.join(", ")}`]);
      if (onlyR.length) pills.push(["bad", `only ${name(st.right)} sends: ${onlyR.join(", ")}`]);
      if (diff.length) pills.push(["bad", `different values: ${diff.join(", ")}`]);
      const lm = Array.isArray(lv.messages) ? lv.messages : null;
      const rm = Array.isArray(rv.messages) ? rv.messages : null;
      if (lm && rm) {
        const k = lm.findIndex((msg, i) => i >= rm.length || JSON.stringify(sortKeys(msg)) !== JSON.stringify(sortKeys(rm[i])));
        const first = k === -1 && lm.length !== rm.length ? Math.min(lm.length, rm.length) : k;
        pills.push([first === -1 ? "ok" : "", first === -1 ? `all ${lm.length} messages have the same content` : `${lm.length} vs ${rm.length} messages; first difference at message ${first + 1} (${(lm[first] || rm[first] || {}).role || "?"})`]);
      }
    }
    return pills;
  }

  function renderWire() {
    const view = $("#wd-view");
    if (!view) return;
    const scn = byId[st.scn];
    const L = scn.runs[st.left];
    const R = scn.runs[st.right];
    const lw = (L && L.wire) || [];
    const rw = (R && R.wire) || [];
    const n = Math.max(lw.length, rw.length);
    if (st.req > n) st.req = 1;
    const sel = $("#wd-req");
    sel.replaceChildren(...Array.from({ length: n }, (_, i) => {
      const same = lw[i] && rw[i] && JSON.stringify(decode(lw[i])) === JSON.stringify(decode(rw[i]));
      return h("option", { value: i + 1, selected: i + 1 === st.req, text: `#${i + 1}${same ? " · identical" : ""}` });
    }));
    const sum = $("#wd-sum");
    if (!n) {
      sum.replaceChildren();
      view.replaceChildren(h("div", { class: "nodata", text: "No recorded requests for this scenario and these loops." }));
      return;
    }
    const lr = lw[st.req - 1];
    const rr = rw[st.req - 1];
    const lv = decode(lr);
    const rv = decode(rr);
    sum.replaceChildren(...wireSummary(lv, rv).map(([cls, text]) => h("span", { class: `pill ${cls}`, text })));
    st.leftBody = lv && typeof lv === "object" ? lv : null;
    const a = pretty(lv, rv);
    const b = pretty(rv, lv);
    view.replaceChildren(
      h("div", { class: "dh" }, h("div", {}, wireMeta(st.left, L, lr)), h("div", {}, wireMeta(st.right, R, rr))),
      h("div", { class: "rows" }, diffRows(diffLines(a, b), a, b)),
    );
  }

  // ---- wiring ----------------------------------------------------------------------------
  function renderAll() {
    stopPlay();
    renderReplay();
    renderWire();
    for (const cell of $$("td.cell")) cell.classList.toggle("sel", cell.dataset.scn === st.scn && (cell.dataset.impl === st.left || cell.dataset.impl === st.right));
  }
  function select(scn, impl) {
    st.scn = scn;
    st.cursor = Infinity;
    st.req = 1;
    if (impl && impl !== st.left && impl !== st.right) {
      if (impl === "our") st.left = impl;
      else st.right = impl;
    }
    $("#rp-scn").value = scn;
    $("#rp-left").value = st.left;
    $("#rp-right").value = st.right;
    renderAll();
  }
  $("#rp-scn").addEventListener("change", (e) => select(e.target.value));
  $("#rp-left").addEventListener("change", (e) => { st.left = e.target.value; renderAll(); });
  $("#rp-right").addEventListener("change", (e) => { st.right = e.target.value; renderAll(); });
  $("#rp-play").addEventListener("click", () => (st.playing ? stopPlay() : startPlay()));
  $("#rp-range").addEventListener("input", (e) => { stopPlay(); setCursor((Number(e.target.value) / 1000) * st.span, true); });
  $("#rp-next").addEventListener("click", () => { stopPlay(); const t = st.times.find((v) => v > st.cursor + 1e-6); setCursor(t ?? st.span, true); });
  $("#rp-prev").addEventListener("click", () => { stopPlay(); const t = [...st.times].reverse().find((v) => v < st.cursor - 1e-6); setCursor(t ?? 0, true); });
  for (const cell of $$("td.cell")) {
    const open = () => {
      select(cell.dataset.scn, cell.dataset.impl);
      const section = $("#replay");
      if (section.scrollIntoView) section.scrollIntoView({ behavior: "smooth" });
    };
    cell.addEventListener("click", (e) => !e.target.closest("button") && open());
    cell.addEventListener("keydown", (e) => (e.key === "Enter" || e.key === " ") && (e.preventDefault(), open()));
  }
  $("#wd-req").addEventListener("change", (e) => { st.req = Number(e.target.value); renderWire(); });
  for (const btn of $$("#wire .seg button")) {
    btn.addEventListener("click", () => {
      st.mode = btn.dataset.mode;
      for (const b of $$("#wire .seg button")) b.setAttribute("aria-pressed", String(b === btn));
      renderWire();
    });
  }
  $("#wd-all").addEventListener("change", (e) => { st.all = e.target.checked; renderWire(); });
  renderAll();
})();

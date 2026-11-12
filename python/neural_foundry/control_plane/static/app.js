// The service owns workflow semantics. This module only presents DTOs and explicit commands.
import { buildExperiment, nextAction, parseSetting, supportsGuided } from "./workflow_builder.mjs";

const $ = (id) => document.getElementById(id);
let token = "";
let templates = [];
let plugins = [];
let runCache = [];
let ready = false;
let draftValid = false;
let candidateNumber = 0;
const selectedWorkloads = new Set();
const settingValues = new Map();
let current = null;
let sequence = 0;
let generation = 0;
let busy = false;
let polling = false;
let creation = null;
const starts = new Map();
const eventsSeen = new Set();
const pretty = (value) => JSON.stringify(value, null, 2);
const part = (value) => encodeURIComponent(value);
const terminal = new Set(["completed", "succeeded", "failed", "cancelled", "canceled"]);

function notice(message, error = false) {
  $("notice").textContent = message;
  $("notice").classList.toggle("error", error);
  $("notice").hidden = false;
  $("notice").setAttribute("role", error ? "alert" : "status");
}

async function request(path, body, method = body === undefined ? "GET" : "POST") {
  const session = token;
  const headers = { Accept: "application/json", Authorization: `Bearer ${session}` };
  if (body !== undefined) headers["Content-Type"] = "application/json";
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 20000);
  try {
    let response;
    try {
      response = await fetch(path, {
        method, headers, body: body === undefined ? undefined : JSON.stringify(body),
        cache: "no-store", credentials: "omit", redirect: "error", signal: controller.signal,
      });
    } catch (error) {
      if (error.name === "AbortError") throw new Error("Request timed out. Refresh state before retrying; draft and start retries keep their original request key.");
      throw new Error("Connection failed. Refresh state before retrying; already-submitted work may still be running.");
    }
    if (session !== token) throw new Error("Session changed. Unlock again to inspect the recorded result.");
    let document;
    try { document = await response.json(); } catch { throw new Error("Server returned an invalid response."); }
    if (!response.ok) {
      if (response.status === 401) lock();
      throw new Error(`${document.error?.code || "request_failed"}: ${document.error?.message || "Request rejected"}`);
    }
    return document;
  } finally {
    clearTimeout(timeout);
  }
}

function jsonObject(id) {
  let value;
  try { value = JSON.parse($(id).value); } catch { throw new Error(`${id}: enter valid JSON.`); }
  if (!value || Array.isArray(value) || typeof value !== "object") throw new Error(`${id}: a JSON object is required.`);
  return value;
}

function lock() {
  token = "";
  generation++;
  current = null;
  templates = [];
  plugins = [];
  runCache = [];
  selectedWorkloads.clear();
  settingValues.clear();
  candidateNumber = 0;
  draftValid = false;
  starts.clear();
  creation = null;
  $("token").value = "";
  $("login").hidden = false;
  $("console").hidden = true;
  $("logout").hidden = true;
  for (const id of ["runs", "events", "stages", "templates", "workload-settings", "candidate-list",
    "builder-limits", "metrics", "plan-summary", "outcome-summary", "lifecycle"]) $(id).replaceChildren();
  for (const id of ["spec", "artifact-document", "artifact-id", "experiment-name", "run-search", "run-filter", "limits"]) $(id).value = "";
  for (const id of ["run-json", "plan-json", "outcome", "artifact-output", "catalog-json", "spec-preview", "run-meta"]) $(id).textContent = "";
  $("plan-review").hidden = true;
  $("run-title").textContent = "Choose an experiment";
  $("run-state").textContent = "No selection";
  delete $("run-state").dataset.state;
  $("approval-status").textContent = "No approval.";
  $("draft-summary").textContent = "Choose a workload to begin.";
  $("spec-validation").textContent = "Choose workloads and name your experiment.";
  $("next-action").textContent = nextAction(null);
  $("mode-guided").checked = true;
  $("advanced-editor").hidden = true;
  $("guided-builder").hidden = false;
  controls();
}

function node(tag, text, className) {
  const element = document.createElement(tag);
  if (text !== undefined) element.textContent = text;
  if (className) element.className = className;
  return element;
}

function controls() {
  const state = String(current?.state || "").toLowerCase();
  const unavailable = busy || !ready || !token;
  $("plan").disabled = unavailable || !current || !["draft", "created", "planned"].includes(state);
  $("approve").disabled = unavailable || !current?.plan || !["planned", "awaiting_approval"].includes(state);
  $("start").disabled = unavailable || !current?.approval?.id || state !== "approved";
  $("cancel").disabled = unavailable || !current || terminal.has(state);
  $("copy-setup").disabled = busy || !token || !current;
  $("guided-builder").disabled = busy || !$("mode-guided").checked || !token;
  $("advanced-editor").disabled = busy || !$("mode-advanced").checked || !token;
  $("create-submit").disabled = unavailable || !draftValid;
  for (const id of ["mode-guided", "mode-advanced", "refresh"]) $(id).disabled = busy || !token;
  for (const form of ["login-form", "artifact-get", "artifact-import"]) {
    $(form).querySelector("button").disabled = busy || !ready;
  }
}

async function command(action) {
  if (busy) return;
  busy = true;
  controls();
  try { await action(); } catch (error) { notice(error.message || "Request failed.", true); }
  finally { busy = false; controls(); }
}

async function refreshRuns() {
  const stamp = generation;
  const document = await request("/v1/runs");
  if (stamp !== generation || !token) return;
  const runs = Array.isArray(document) ? document : document.runs || [];
  runCache = runs;
  renderRunList();
}

function renderRunList() {
  const query = $("run-search").value.toLowerCase();
  const filter = $("run-filter").value;
  const runs = runCache.filter((run) => (!filter || run.state === filter)
    && `${run.spec?.name || ""} ${run.id}`.toLowerCase().includes(query));
  $("runs").replaceChildren();
  if (!runs.length) $("runs").append(node("p", runCache.length
    ? "No experiments match these filters." : "No experiments yet. Choose Create experiment to begin.", "empty"));
  for (const run of runs) {
    const button = node("button", undefined, `run-item${current?.id === run.id ? " selected" : ""}`);
    button.append(node("span", run.spec?.name || run.id), node("small", `${run.state} · ${run.id}`));
    button.type = "button";
    button.addEventListener("click", () => command(() => selectRun(run.id)));
    $("runs").append(button);
  }
}

function renderRun(run) {
  if (!token) return;
  const previous = current;
  current = run;
  $("run-title").textContent = run.spec?.name || run.id;
  $("run-state").textContent = run.state;
  $("run-meta").textContent = `${run.id} · updated ${run.updated_at || "—"} · spec ${run.spec_digest || "—"}`;
  $("run-json").textContent = pretty(run);
  $("plan-json").textContent = run.plan ? pretty(run.plan) : "No plan yet. Generate and review a plan before approval.";
  $("outcome").textContent = run.outcome ? pretty(run.outcome) : "No outcome recorded. No quality or release claim.";
  $("next-action").textContent = nextAction(run);
  $("run-state").dataset.state = run.state;
  $("lifecycle").replaceChildren();
  const phase = { created: 0, draft: 0, planned: 1, approved: 2, queued: 3, running: 3, completed: 3 }[run.state];
  for (const [index, title] of ["Draft", "Review plan", "Approve", "Execute"].entries()) {
    const item = node("li", title, phase === undefined ? "" : index < phase ? "done" : index === phase ? "active" : "");
    if (index === phase) item.setAttribute("aria-current", "step");
    $("lifecycle").append(item);
  }
  $("plan-review").hidden = !run.plan;
  summaryCards("plan-summary", run.plan ? [
    ["Candidates", run.spec.candidates.length], ["Stages", run.plan.stages.length],
    ...Object.entries(run.plan.limits).map(([name, value]) => [friendly(name), value.toLocaleString()]),
    ["Model calls", run.plan.model_calls === false ? "None (software fixture)" : "Possible; review plan"],
  ] : []);
  summaryCards("outcome-summary", [
    ["Execution", run.outcome?.execution || run.state],
    ["Measured quality", run.outcome?.quality || "Not evaluated"],
    ["Release permission", run.outcome?.release_authorized === false ? "Not authorized" : "Not established here"],
  ]);
  renderMetrics(run);
  $("approval-status").textContent = run.approval?.id
    ? "This exact plan and its limits are approved. Start remains a separate action; audit identifiers are in the full run document."
    : "No approval. Approval and execution are separate actions.";
  if (previous?.id !== run.id || previous?.plan?.digest !== run.plan?.digest) {
    $("limits").value = pretty(run.approval?.limits || run.plan?.limits || run.spec?.limits || {});
  }
  $("stages").replaceChildren();
  if (!run.stages?.length) $("stages").append(node("p", "No stages recorded.", "empty"));
  for (const stage of run.stages || []) {
    const item = node("div", undefined, "stage");
    const header = node("div", undefined, "stage-header");
    header.append(node("strong", `${friendly(stage.operation.replace(/^software\./, ""))} · ${stage.id}`), node("span", stage.state));
    const attempts = Array.isArray(stage.attempts) ? stage.attempts.length : stage.attempts ?? 0;
    item.append(header, node("small", `${stage.operation} · attempts ${attempts}`));
    if (stage.candidate_id) item.append(node("small", `Candidate: ${stage.candidate_id}`));
    const errors = Array.isArray(stage.attempts) ? stage.attempts.filter((attempt) => attempt.error) : [];
    for (const attempt of errors) item.append(node("p", `${attempt.error.code}: ${attempt.error.message}`, "stage-error"));
    if (stage.artifact_id) {
      const inspect = node("button", "Inspect stage artifact", "quiet");
      inspect.type = "button";
      inspect.addEventListener("click", () => command(async () => {
        $("artifact-id").value = stage.artifact_id;
        $("artifact-output").textContent = pretty(await request(`/v1/artifacts/${part(stage.artifact_id)}`));
        $("artifacts").scrollIntoView({ behavior: "smooth" });
      }));
      item.append(inspect);
    }
    if (stage.progress !== null && stage.progress !== undefined) {
      // Progress schemas belong to plugins: never fabricate a percentage from an unknown object.
      if (typeof stage.progress === "number" && stage.progress >= 0 && stage.progress <= 1) {
        const progress = document.createElement("progress");
        progress.max = 1;
        progress.value = stage.progress;
        progress.setAttribute("aria-label", `${stage.id} progress`);
        item.append(progress);
      } else if (Number.isFinite(stage.progress.completed) && Number.isFinite(stage.progress.total)
        && stage.progress.total > 0 && stage.progress.completed >= 0) {
        const progress = document.createElement("progress");
        progress.max = stage.progress.total;
        progress.value = Math.min(stage.progress.completed, stage.progress.total);
        progress.setAttribute("aria-label", `${stage.id} ${stage.progress.kind || "stage"} progress`);
        item.append(progress);
      }
      item.append(node("pre", pretty(stage.progress)));
    }
    if (stage.result) {
      const evidence = document.createElement("details");
      evidence.append(node("summary", "Evidence & metric policy"), node("pre", pretty(stage.result)));
      item.append(evidence);
    }
    $("stages").append(item);
  }
  runCache = runCache.map((item) => item.id === run.id ? run : item);
  renderRunList();
  controls();
}

function friendly(value) {
  return value.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function summaryCards(id, values) {
  $(id).replaceChildren();
  for (const [label, value] of values) {
    const item = node("div", undefined, "summary-item");
    item.append(node("span", label), node("strong", String(value)));
    $(id).append(item);
  }
}

function renderMetrics(run) {
  $("metrics").replaceChildren();
  const rows = (run.stages || []).flatMap((stage) => (Array.isArray(stage.result?.metrics) ? stage.result.metrics : []).map((metric) => [
    stage.candidate_id || "Shared", stage.id, metric.name, metric.version, metric.value, metric.unit,
  ]));
  if (!rows.length) {
    $("metrics").append(node("p", "No versioned metrics recorded yet. Stage evidence may contain diagnostics instead.", "empty"));
    return;
  }
  const table = node("table");
  table.append(node("caption", "Recorded metrics · no cross-gym averaging"));
  const head = node("thead");
  const headings = node("tr");
  for (const text of ["Candidate", "Stage", "Metric", "Version", "Value", "Unit"]) {
    const cell = node("th", text);
    cell.scope = "col";
    headings.append(cell);
  }
  head.append(headings);
  table.append(head);
  const body = node("tbody");
  for (const values of rows) {
    const row = node("tr");
    for (const value of values) row.append(node("td", String(value ?? "Not recorded")));
    body.append(row);
  }
  table.append(body);
  $("metrics").append(table);
}

function addCandidate(value) {
  candidateNumber++;
  const row = node("div", undefined, "candidate-row");
  const label = node("label", "Candidate label");
  const input = node("input");
  input.id = `candidate-${candidateNumber}`;
  input.value = value || `candidate-${candidateNumber}`;
  input.required = true;
  input.maxLength = 128;
  input.dataset.candidate = "";
  label.htmlFor = input.id;
  const remove = node("button", "Remove", "quiet");
  remove.type = "button";
  remove.addEventListener("click", () => { row.remove(); creation = null; updateDraft(); });
  row.append(label, input, remove);
  $("candidate-list").append(row);
}

function renderBuilder() {
  $("templates").replaceChildren();
  templates.forEach((template, index) => {
    const label = node("label", undefined, "workload-card");
    const input = node("input");
    input.type = "checkbox";
    input.value = String(index);
    input.disabled = !supportsGuided(template, plugins);
    input.addEventListener("change", () => {
      if (input.checked) selectedWorkloads.add(index); else selectedWorkloads.delete(index);
      renderSettings();
      creation = null;
      updateDraft();
    });
    const text = node("span");
    text.append(node("strong", template.name || template.id),
      node("small", input.disabled ? "Not available in guided setup. Review the plugin contract in Advanced JSON."
        : `${template.spec.stages.length} stages · CPU software fixture`));
    label.append(input, text);
    $("templates").append(label);
  });
  if (!templates.length) $("templates").append(node("p", "No workload templates are registered on this server.", "empty"));
  addCandidate("candidate-a");
  const labels = { steps: "Steps per stage", episodes: "Episodes per stage",
    wall_seconds: "Wall time per stage (seconds)", recorded_bytes: "Recorded bytes per stage" };
  for (const [name, text] of Object.entries(labels)) {
    const wrapper = node("div");
    const label = node("label", text);
    const input = node("input");
    input.id = `limit-${name}`;
    input.type = "number";
    input.min = "1";
    input.step = "1";
    input.placeholder = "Use server default";
    input.dataset.limit = name;
    label.htmlFor = input.id;
    wrapper.append(label, input);
    $("builder-limits").append(wrapper);
  }
  renderSettings();
  updateDraft();
}

function renderSettings() {
  $("workload-settings").replaceChildren();
  for (const index of selectedWorkloads) {
    const template = templates[index];
    const group = node("fieldset", undefined, "settings-group");
    group.append(node("legend", template.name));
    let fields = 0;
    template.spec.stages.forEach((stage, stageIndex) => {
      const schema = plugins.find((plugin) => plugin.id === stage.operation).config_schema;
      for (const [name, setting] of Object.entries(schema?.properties || {})) {
        fields++;
        const key = `${index}:${stage.id}:${name}`;
        const input = node(setting.type === "array" && setting.items.type === "string" ? "textarea" : "input");
        input.id = `setting-${index}-${stageIndex}-${name}`;
        input.dataset.workload = String(index);
        input.dataset.stage = stage.id;
        input.dataset.setting = name;
        input.required = schema.required?.includes(name) || false;
        if (input.tagName === "INPUT") {
          input.type = setting.type === "array" ? "text" : "number";
          if (input.type === "number") {
            input.step = setting.type === "integer" ? "1" : "any";
            if (setting.minimum !== undefined) input.min = setting.minimum;
            if (setting.maximum !== undefined) input.max = setting.maximum;
          }
        } else input.rows = 3;
        const initial = stage.config[name];
        input.value = settingValues.get(key) ?? (Array.isArray(initial)
          ? initial.join(setting.items.type === "string" ? "\n" : ", ") : initial ?? "");
        input.placeholder = setting.type === "array"
          ? setting.items.type === "string" ? "One text example per line" : "Comma-separated numbers, for example 0, 1"
          : "Use plugin default";
        const label = node("label", `${friendly(stage.operation.replace(/^software\./, ""))}: ${friendly(name)}`);
        label.htmlFor = input.id;
        const help = node("small", setting.type === "array"
          ? `${setting.minItems ?? 0}-${setting.maxItems ?? "bounded"} entries. Blank keeps the default.`
          : `${setting.exclusiveMinimum !== undefined ? `Greater than ${setting.exclusiveMinimum}` : `Minimum ${setting.minimum ?? "not specified"}`}; maximum ${setting.maximum ?? "not specified"}.`, "muted");
        help.id = `${input.id}-help`;
        input.setAttribute("aria-describedby", help.id);
        input.addEventListener("input", () => settingValues.set(key, input.value));
        group.append(label, input, help);
      }
    });
    if (!fields) group.append(node("p", "This workload has no editable plugin settings.", "empty"));
    $("workload-settings").append(group);
  }
  if (!selectedWorkloads.size) $("workload-settings").append(node("p", "Select a workload to see its settings.", "empty"));
}

function guidedSpec() {
  const workloads = [...selectedWorkloads].map((index) => ({ template: templates[index], configs: {}, index }));
  for (const input of $("workload-settings").querySelectorAll("[data-setting]")) {
    const workload = workloads.find((item) => item.index === Number(input.dataset.workload));
    const stage = workload.template.spec.stages.find((item) => item.id === input.dataset.stage);
    const schema = plugins.find((item) => item.id === stage.operation).config_schema.properties[input.dataset.setting];
    const value = parseSetting(input.value, schema, friendly(input.dataset.setting));
    if (value !== undefined) {
      workload.configs[stage.id] ||= {};
      workload.configs[stage.id][input.dataset.setting] = value;
    }
  }
  const limits = {};
  for (const input of $("builder-limits").querySelectorAll("input")) {
    const maxima = workloads.flatMap(({ template }) => template.spec.stages.map((stage) =>
      plugins.find((item) => item.id === stage.operation).budgets?.[input.dataset.limit])).filter(Number.isFinite);
    const maximum = maxima.length ? Math.min(...maxima) : undefined;
    const value = parseSetting(input.value, { type: "integer", minimum: 1, maximum }, friendly(input.dataset.limit));
    if (value !== undefined) limits[input.dataset.limit] = value;
  }
  return buildExperiment({ name: $("experiment-name").value,
    candidateIds: [...$("candidate-list").querySelectorAll("input")].map((input) => input.value), workloads, limits });
}

function updateDraft() {
  try {
    if ($("mode-guided").checked && [...$("guided-builder").querySelectorAll("input")].some((input) => input.validity.badInput)) {
      throw new Error("Complete the invalid numeric setting before continuing.");
    }
    const spec = $("mode-guided").checked ? guidedSpec() : jsonObject("spec");
    $("spec-preview").textContent = pretty(spec);
    $("draft-summary").textContent = `${spec.candidates?.length ?? 0} candidates · ${selectedWorkloads.size} selected workloads · ${spec.stages?.length ?? 0} stages. Create saves a draft only; review, approval and start are separate.`;
    $("spec-validation").textContent = "Draft ready for server validation. No work will run on create.";
    draftValid = true;
  } catch (error) {
    draftValid = false;
    $("spec-validation").textContent = error.message;
    $("draft-summary").textContent = error.message;
    $("spec-preview").textContent = "";
  }
  $("spec-validation").classList.toggle("validation-error", !draftValid);
  controls();
}

async function selectRun(id) {
  const stamp = ++generation;
  const run = await request(`/v1/runs/${part(id)}`);
  if (stamp !== generation || !token) return;
  sequence = 0;
  eventsSeen.clear();
  $("events").replaceChildren();
  renderRun(run);
  await Promise.all([refreshRuns(), refreshEvents(id, stamp)]);
}

async function refreshEvents(id, stamp) {
  const document = await request(`/v1/runs/${part(id)}/events?after=${sequence}`);
  if (stamp !== generation || current?.id !== id) return;
  const events = (Array.isArray(document) ? document : document.events || [])
    .map((event) => ({ ...event, seq: event.seq ?? event.sequence }))
    .sort((a, b) => Number(a.seq) - Number(b.seq));
  for (const event of events) {
    if (!Number.isSafeInteger(event.seq) || event.seq <= sequence || eventsSeen.has(event.seq)) continue;
    eventsSeen.add(event.seq);
    sequence = event.seq;
    const item = node("li");
    item.append(node("strong", `#${event.seq} · ${event.type || event.kind || "event"}`),
      node("small", event.created_at || event.timestamp || ""));
    if (event.payload?.message) item.append(node("p", event.payload.message, "stage-error"));
    const detail = node("details");
    detail.append(node("summary", "Inspect event details"), node("pre", pretty(event)));
    item.append(detail);
    $("events").append(item);
  }
  while ($("events").children.length > 500) $("events").firstElementChild.remove();
}

$("login-form").addEventListener("submit", (event) => {
  event.preventDefault();
  command(async () => {
    token = $("token").value.trim();
    $("token").value = "";
    try {
      const catalog = await request("/v1/catalog");
      $("catalog-json").textContent = pretty(catalog.plugins || []);
      templates = catalog.templates || [];
      plugins = catalog.plugins || [];
      renderBuilder();
      $("login").hidden = true;
      $("console").hidden = false;
      $("logout").hidden = false;
      notice("Session unlocked. Browsing and polling are read-only.");
      await refreshRuns();
    } catch (error) { lock(); throw error; }
  });
});
$("logout").addEventListener("click", () => { lock(); notice("Console locked. Server-side runs continue independently."); });
$("refresh").addEventListener("click", () => command(async () => {
  if (current) {
    const id = current.id;
    renderRun(await request(`/v1/runs/${part(id)}`));
    await refreshEvents(id, generation);
  }
  await refreshRuns();
}));
$("run-search").addEventListener("input", renderRunList);
$("run-filter").addEventListener("change", renderRunList);
$("copy-setup").addEventListener("click", () => {
  if (!current) return;
  if (($("experiment-name").value || $("spec").value.trim())
      && !window.confirm("Replace the current unsaved setup with a copy of this experiment?")) return;
  $("spec").value = pretty(current.spec);
  $("mode-advanced").checked = true;
  $("guided-builder").hidden = true;
  $("advanced-editor").hidden = false;
  creation = null;
  updateDraft();
  $("new-run").scrollIntoView({ behavior: "smooth" });
  notice("Setup copied into Advanced JSON. The original run is unchanged; edit the name and create a fresh draft when ready.");
});
$("add-candidate").addEventListener("click", () => {
  if ($("candidate-list").children.length >= 16) {
    notice("An experiment supports at most 16 candidates.", true);
    return;
  }
  addCandidate();
  creation = null;
  updateDraft();
});
for (const mode of ["guided", "advanced"]) $(`mode-${mode}`).addEventListener("change", () => {
  if (mode === "guided" && $("spec").value.trim()
      && !window.confirm("Switch to guided setup? Your advanced changes will not be imported into the guided fields.")) {
    $("mode-advanced").checked = true;
    return;
  }
  if (mode === "advanced" && draftValid) $("spec").value = pretty(guidedSpec());
  $("guided-builder").hidden = mode !== "guided";
  $("advanced-editor").hidden = mode !== "advanced";
  creation = null;
  updateDraft();
});
$("create-form").addEventListener("input", () => {
  creation = null;
  updateDraft();
});
$("create-form").addEventListener("submit", (event) => {
  event.preventDefault();
  command(async () => {
    const spec = $("mode-guided").checked ? guidedSpec() : jsonObject("spec");
    // Retain the key after uncertain network failures; a retry must not create another run.
    creation ||= crypto.randomUUID();
    const run = await request("/v1/runs", { spec, idempotency_key: creation });
    creation = null;
    renderRun(run);
    await selectRun(run.id);
    $("workspace").scrollIntoView({ behavior: "smooth" });
    notice("Draft created. Review a generated plan before granting execution authority.");
  });
});
$("plan").addEventListener("click", () => command(async () => {
  const run = await request(`/v1/runs/${part(current.id)}/plan`, {}, "POST");
  renderRun(run);
  $("plan-details").open = true;
  notice("Plan generated. Inspect preflight, stages and approved limits.");
  await refreshRuns();
}));
$("approve").addEventListener("click", () => command(async () => {
  const limits = jsonObject("limits");
  if (!window.confirm("Approve this exact plan and these limits? This does not start the run.")) return;
  const id = current.id;
  const approval = await request(`/v1/runs/${part(id)}/approve`, { plan_digest: current.plan.digest, limits });
  const run = await request(`/v1/runs/${part(id)}`);
  if (!run.approval) run.approval = approval;
  renderRun(run);
  notice("Plan approved. Execution still requires a separate explicit Start.");
}));
$("start").addEventListener("click", () => command(async () => {
  if (!window.confirm("Start this approved workflow now? Its worker may execute the bounded stages.")) return;
  const id = current.id;
  if (!starts.has(id)) starts.set(id, crypto.randomUUID());
  renderRun(await request(`/v1/runs/${part(id)}/start`, {
    approval_id: current.approval.id, idempotency_key: starts.get(id),
  }));
  notice("Start accepted. Follow durable events; completion is not a quality or release decision.");
  await refreshRuns();
}));
$("cancel").addEventListener("click", () => command(async () => {
  if (!window.confirm("Request cancellation? Already-issued work may need to settle safely.")) return;
  renderRun(await request(`/v1/runs/${part(current.id)}/cancel`, {}, "POST"));
  notice("Cancellation recorded. Inspect stage state and events for acknowledgement.");
  await refreshRuns();
}));
$("artifact-get").addEventListener("submit", (event) => {
  event.preventDefault();
  command(async () => { $("artifact-output").textContent = pretty(await request(`/v1/artifacts/${part($("artifact-id").value.trim())}`)); });
});
$("artifact-import").addEventListener("submit", (event) => {
  event.preventDefault();
  command(async () => {
    $("artifact-output").textContent = pretty(await request("/v1/artifacts", jsonObject("artifact-document")));
    notice("Existing artifact document imported.");
  });
});

async function checkReady() {
  try {
    const response = await fetch("/ready", { cache: "no-store", credentials: "omit", signal: AbortSignal.timeout(10000) });
    const document = await response.json();
    ready = response.ok && document.ready === true;
    $("health").textContent = ready ? "Ready · private CPU runtime" : "Service not ready · actions paused";
  } catch {
    ready = false;
    $("health").textContent = "Connection unavailable · actions paused";
  }
  controls();
}
setInterval(async () => {
  if (!token || !current || busy || polling || document.hidden) return;
  polling = true;
  const stamp = generation;
  const id = current.id;
  try {
    const run = await request(`/v1/runs/${part(id)}`);
    if (stamp !== generation || !token) return;
    renderRun(run);
    await refreshEvents(id, stamp);
    $("poll-state").textContent = "Read-only polling · live";
  } catch (error) { $("poll-state").textContent = `Polling paused: ${error.message}`; }
  finally { polling = false; }
}, 3000);
setInterval(checkReady, 15000);
checkReady();

// The service owns workflow semantics. This module only presents DTOs and explicit commands.
const $ = (id) => document.getElementById(id);
let token = "";
let templates = [];
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
}

async function request(path, body, method = body === undefined ? "GET" : "POST") {
  const headers = { Accept: "application/json", Authorization: `Bearer ${token}` };
  if (body !== undefined) headers["Content-Type"] = "application/json";
  const response = await fetch(path, {
    method, headers, body: body === undefined ? undefined : JSON.stringify(body),
    cache: "no-store", credentials: "omit", redirect: "error",
  });
  let document;
  try { document = await response.json(); } catch { throw new Error("Server returned an invalid response."); }
  if (!response.ok) {
    if (response.status === 401) lock();
    throw new Error(`${document.error?.code || "request_failed"}: ${document.error?.message || "Request rejected"}`);
  }
  return document;
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
  starts.clear();
  creation = null;
  $("token").value = "";
  $("login").hidden = false;
  $("console").hidden = true;
  $("logout").hidden = true;
  for (const id of ["runs", "events", "stages"]) $(id).replaceChildren();
  for (const id of ["spec", "artifact-document", "artifact-id"]) $(id).value = "";
  for (const id of ["run-json", "plan-json", "outcome", "artifact-output", "catalog-json"]) $(id).textContent = "";
}

function node(tag, text, className) {
  const element = document.createElement(tag);
  if (text !== undefined) element.textContent = text;
  if (className) element.className = className;
  return element;
}

function controls() {
  const state = String(current?.state || "").toLowerCase();
  $("plan").disabled = busy || !current || !["draft", "created", "planned"].includes(state);
  $("approve").disabled = busy || !current?.plan || !["planned", "awaiting_approval"].includes(state);
  $("start").disabled = busy || !current?.approval?.id || state !== "approved";
  $("cancel").disabled = busy || !current || terminal.has(state);
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
  $("runs").replaceChildren();
  if (!runs.length) $("runs").append(node("p", "No runs yet. Start with a catalog template.", "empty"));
  for (const run of runs) {
    const button = node("button", undefined, `run-item${current?.id === run.id ? " selected" : ""}`);
    button.append(node("span", run.spec?.name || run.id), node("small", `${run.state} · ${run.id}`));
    button.type = "button";
    button.addEventListener("click", () => command(() => selectRun(run.id)));
    $("runs").append(button);
  }
}

function renderRun(run) {
  const previous = current;
  current = run;
  $("run-title").textContent = run.spec?.name || run.id;
  $("run-state").textContent = run.state;
  $("run-meta").textContent = `${run.id} · updated ${run.updated_at || "—"} · spec ${run.spec_digest || "—"}`;
  $("run-json").textContent = pretty(run);
  $("plan-json").textContent = run.plan ? pretty(run.plan) : "No plan yet. Generate and review a plan before approval.";
  $("outcome").textContent = run.outcome ? pretty(run.outcome) : "No outcome recorded. No quality or release claim.";
  $("approval-status").textContent = run.approval?.id
    ? `Approval ${run.approval.id} · plan ${run.approval.plan_digest || run.plan?.digest || "—"}`
    : "No approval. Approval and execution are separate actions.";
  if (previous?.id !== run.id || previous?.plan?.digest !== run.plan?.digest) {
    $("limits").value = pretty(run.approval?.limits || run.plan?.limits || run.spec?.limits || {});
  }
  $("stages").replaceChildren();
  if (!run.stages?.length) $("stages").append(node("p", "No stages recorded.", "empty"));
  for (const stage of run.stages || []) {
    const item = node("div", undefined, "stage");
    const header = node("div", undefined, "stage-header");
    header.append(node("strong", stage.id), node("span", stage.state));
    const attempts = Array.isArray(stage.attempts) ? stage.attempts.length : stage.attempts ?? 0;
    item.append(header, node("small", `${stage.operation} · attempts ${attempts}`));
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
      node("small", event.created_at || event.timestamp || ""), node("pre", pretty(event)));
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
      $("templates").replaceChildren(node("option", "Select a template…"));
      $("templates").firstElementChild.value = "";
      templates.forEach((template, index) => {
        const option = node("option", template.name || template.id);
        option.value = String(index);
        $("templates").append(option);
      });
      $("login").hidden = true;
      $("console").hidden = false;
      $("logout").hidden = false;
      notice("Session unlocked. Browsing and polling are read-only.");
      await refreshRuns();
    } catch (error) { lock(); throw error; }
  });
});
$("logout").addEventListener("click", () => { lock(); notice("Console locked. Server-side runs continue independently."); });
$("refresh").addEventListener("click", () => command(refreshRuns));
$("templates").addEventListener("change", () => {
  const template = templates[Number($("templates").value)];
  if ($("templates").value === "" || !template) return;
  $("spec").value = pretty(template.spec);
  $("template-description").textContent = template.description || "";
  creation = null;
});
$("spec").addEventListener("input", () => {
  creation = null;
  try { jsonObject("spec"); $("spec-validation").textContent = "Valid JSON object · server validates workflow semantics."; }
  catch (error) { $("spec-validation").textContent = error.message; }
});
$("create-form").addEventListener("submit", (event) => {
  event.preventDefault();
  command(async () => {
    const spec = jsonObject("spec");
    // Retain the key after uncertain network failures; a retry must not create another run.
    creation ||= crypto.randomUUID();
    const run = await request("/v1/runs", { spec, idempotency_key: creation });
    creation = null;
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
    const response = await fetch("/ready", { cache: "no-store", credentials: "omit" });
    $("health").textContent = response.ok ? "Database ready" : "Service not ready";
  } catch { $("health").textContent = "Connection unavailable"; }
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

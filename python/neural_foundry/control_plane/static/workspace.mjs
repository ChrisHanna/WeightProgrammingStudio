import { datasetDocument, evidenceRows, measurementRows, parameterValue } from "./workspace_logic.mjs";
import { nextAction } from "./workflow_builder.mjs";

const $ = (id) => document.getElementById(id);
const pretty = (value) => JSON.stringify(value, null, 2);
const part = (value) => encodeURIComponent(value);
const state = { token: "", ready: false, busy: false, generation: 0, workspace: null,
  projectId: null, snapshot: null, view: "overview", run: null, preview: null, actions: [], gymParent: null };
const requestKeys = new Map();
let dialogSubmit = null;
let previewGeneration = 0;
const views = new Set(["overview", "models", "datasets", "gyms", "evaluate", "fine_tune", "weight_edit", "merge", "experiments", "export"]);
const assets = (kind) => (state.snapshot?.assets || []).filter((asset) => asset.kind === kind);
const assetById = (id) => (state.snapshot?.assets || []).find((asset) => asset.id === id);
const assetName = (id) => { const asset = assetById(id); return asset ? `${asset.name} · v${asset.version}` : id || "Not recorded"; };
const selectedValues = (select) => [...select.selectedOptions].map((option) => option.value);
const projectPath = () => `/v1/projects/${part(state.projectId)}`;

function node(tag, text, cls) {
  const element = document.createElement(tag);
  if (text !== undefined) element.textContent = text;
  if (cls) element.className = cls;
  return element;
}

function notify(message, error = false) {
  $("notification").textContent = message;
  $("notification").hidden = false;
  $("notification").classList.toggle("error", error);
  $("notification").setAttribute("role", error ? "alert" : "status");
}

function controls() {
  const unavailable = !state.ready || state.busy;
  $("project-switch").disabled = unavailable || !state.token;
  $("new-project").disabled = unavailable || !state.token;
  $("first-project").disabled = unavailable;
  $("refresh-workspace").disabled = unavailable;
  for (const button of document.querySelectorAll("form button[type=submit], [data-mutation]")) button.disabled = unavailable;
  for (const button of $("navigation").querySelectorAll("button")) button.disabled = state.busy || !state.token || !state.projectId;
  if (state.run) {
    $("review-experiment").disabled = unavailable || !["created", "planned"].includes(state.run.state);
    $("approve-experiment").disabled = unavailable || state.run.state !== "planned";
    $("start-experiment").disabled = unavailable || state.run.state !== "approved";
    $("cancel-experiment").disabled = unavailable || ["completed", "failed", "cancelled"].includes(state.run.state);
  }
}

async function request(path, body) {
  const token = state.token;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 20000);
  try {
    let response;
    try {
      response = await fetch(path, { method: body === undefined ? "GET" : "POST",
        headers: { Accept: "application/json", Authorization: `Bearer ${token}`,
          ...(body === undefined ? {} : { "Content-Type": "application/json" }) },
        body: body === undefined ? undefined : JSON.stringify(body), signal: controller.signal,
        credentials: "omit", cache: "no-store", redirect: "error" });
    } catch {
      throw new Error("Connection interrupted. Refresh the workspace before retrying; an accepted operation may still exist.");
    }
    if (token !== state.token) throw new Error("Session changed. Unlock again to inspect the result.");
    let document;
    try { document = await response.json(); } catch { throw new Error("The server returned an invalid response."); }
    if (!response.ok) {
      if (response.status === 401) lock();
      throw new Error(`${document.error?.message || "Request rejected"} (${document.error?.code || response.status})`);
    }
    return document;
  } finally { clearTimeout(timer); }
}

async function persist(path, body) {
  const key = path + pretty(body);
  if (!requestKeys.has(key)) requestKeys.set(key, crypto.randomUUID());
  const result = await request(path, { ...body, idempotency_key: requestKeys.get(key) });
  return result;
}

async function action(callback) {
  if (state.busy) return;
  state.busy = true;
  controls();
  try { await callback(); }
  catch (error) {
    notify(error.message || "The operation failed.", true);
    if ($("workspace-dialog").open) {
      $("dialog-error").textContent = error.message || "The operation failed.";
      $("dialog-error").hidden = false;
    }
  } finally { state.busy = false; controls(); }
}

function lock() {
  state.token = "";
  state.workspace = state.snapshot = state.run = state.projectId = null;
  state.generation++;
  requestKeys.clear();
  resetPreview();
  $("workspace-token").value = "";
  $("workspace-login").hidden = false;
  $("product-workspace").hidden = true;
  $("project-switch").replaceChildren(node("option", "Select a project"));
  $("lock").hidden = $("refresh-workspace").hidden = true;
  $("workspace-dialog").close();
  dialogSubmit = null;
  for (const id of ["dialog-fields", "project-title", "project-description", "project-stats", "model-list",
    "dataset-list", "gym-list", "evidence-list", "experiment-list", "experiment-stages",
    "experiment-json", "experiment-events", "comparison-table", "recent-experiments",
    "fine-tune-form", "fine-tune-recipes", "weight-edit-form", "weight-edit-recipes", "merge-form", "merge-recipes",
    "gym-template", "gym-dataset", "preview-example", "evaluation-gym", "evaluation-evidence",
    "experiment-title", "experiment-state", "experiment-scope", "experiment-next", "experiment-plan"]) {
    $(id).replaceChildren();
  }
  $("gym-form").reset();
  $("evaluation-form").reset();
  controls();
}

function showView(view) {
  state.view = views.has(view) ? view : "overview";
  for (const section of document.querySelectorAll("[data-page]")) section.hidden = section.dataset.page !== state.view;
  for (const button of $("navigation").querySelectorAll("[data-view]")) {
    const active = button.dataset.view === state.view;
    button.classList.toggle("active", active);
    if (active) button.setAttribute("aria-current", "page"); else button.removeAttribute("aria-current");
  }
  if (state.projectId) history.replaceState(null, "", `#/${part(state.projectId)}/${state.view}`);
}

async function loadWorkspace(preferredProject) {
  const generation = state.generation;
  const document = await request("/v1/workspace");
  if (!state.token || generation !== state.generation) return;
  state.workspace = document;
  $("project-switch").replaceChildren(node("option", "Select a project"));
  $("project-switch").firstChild.value = "";
  for (const project of document.projects) {
    const option = node("option", project.name);
    option.value = project.id;
    $("project-switch").append(option);
  }
  const projectId = preferredProject || state.projectId || document.projects[0]?.id;
  if (projectId && document.projects.some((project) => project.id === projectId)) {
    await loadProject(projectId);
  } else {
    state.projectId = state.snapshot = null;
    $("no-project").hidden = false;
    $("project-pages").hidden = true;
    if (projectId) notify("That project is not available. Select a project from this workspace.", true);
  }
  controls();
}

async function loadProject(projectId) {
  const drafts = projectId === state.projectId
    ? [...$("project-pages").querySelectorAll("input[id],select[id],textarea[id]")].filter((input) => input.type !== "file")
      .map((input) => [input.id, input.multiple ? selectedValues(input) : input.value]) : null;
  const gymParent = drafts ? state.gymParent : null;
  const generation = ++state.generation;
  const document = await request(`/v1/projects/${part(projectId)}`);
  if (!state.token || generation !== state.generation) return;
  state.projectId = projectId;
  state.snapshot = document;
  state.run = null;
  state.gymParent = gymParent;
  if (!drafts) {
    $("gym-form").reset();
    $("evaluation-form").reset();
    $("preview-response").value = "";
  }
  $("project-switch").value = projectId;
  $("no-project").hidden = true;
  $("project-pages").hidden = false;
  $("experiment-detail").hidden = true;
  resetPreview();
  renderProject();
  if (drafts) {
    for (const [id, value] of drafts) {
      const input = $(id);
      if (input?.tagName === "SELECT" && !input.multiple) {
        input.value = value;
        if (id.endsWith("-method") || id === "gym-template") input.dispatchEvent(new Event("change"));
      }
    }
    renderEvaluationEvidence();
    for (const [id, value] of drafts) {
      const input = $(id);
      if (!input) continue;
      if (input.multiple) for (const option of input.options) option.selected = value.includes(option.value);
      else input.value = value;
    }
  }
  showView(state.view);
}

function stat(label, value) {
  const card = node("div", undefined, "stat");
  card.append(node("small", label), node("strong", String(value)));
  return card;
}

function button(text, callback, cls = "text-button") {
  const element = node("button", text, cls);
  element.type = "button";
  element.addEventListener("click", callback);
  return element;
}

function empty(container, message) {
  container.replaceChildren(node("div", message, "empty-state"));
}

function assetTable(container, values) {
  if (!values.length) { empty(container, "No assets yet. Add one to start this workstream."); return; }
  const table = node("table");
  const head = node("thead");
  const header = node("tr");
  for (const text of ["Asset", "Version", "Identity / context", ""]) header.append(node("th", text));
  head.append(header); table.append(head);
  const body = node("tbody");
  for (const asset of values) {
    const row = node("tr");
    const name = node("td"); name.append(node("strong", asset.name), node("small", asset.kind));
    const context = asset.kind === "model" ? `${asset.document.model_id} · ${asset.document.revision}`
      : asset.kind === "dataset" ? `${asset.document.examples.length} examples`
      : asset.kind === "gym" ? asset.document.template
      : asset.kind === "recipe" ? `${asset.document.method} · saved plan`
      : asset.document.source_description || "Recorded candidate answers";
    const actions = node("td");
    actions.append(button("Inspect", () => openAsset(asset)));
    row.append(name, node("td", `v${asset.version}`), node("td", context), actions);
    body.append(row);
  }
  table.append(body);
  container.replaceChildren(table);
  container.classList.add("table-wrap");
}

function fillAssetOptions(select, values, placeholder, preserve = true) {
  const selected = preserve ? new Set(selectedValues(select)) : new Set();
  select.replaceChildren();
  if (!select.multiple) {
    const option = node("option", placeholder);
    option.value = ""; select.append(option);
  }
  for (const asset of values) {
    const option = node("option", `${asset.name} · v${asset.version}`);
    option.value = asset.id; option.selected = selected.has(asset.id);
    select.append(option);
  }
}

function renderProject() {
  const { project, runs } = state.snapshot;
  $("project-title").textContent = project.name;
  $("project-description").textContent = project.description || "A versioned model-improvement project.";
  $("project-stats").replaceChildren(stat("Model references", assets("model").length),
    stat("Dataset versions", assets("dataset").length), stat("Gym versions", assets("gym").length),
    stat("Experiments", runs.length));
  assetTable($("model-list"), assets("model"));
  assetTable($("dataset-list"), assets("dataset"));
  assetTable($("gym-list"), assets("gym"));
  assetTable($("evidence-list"), assets("evidence"));
  renderRuns($("recent-experiments"), runs.slice(0, 5));
  renderRuns($("experiment-list"), runs);
  renderComparison();
  $("capability-list").replaceChildren();
  const names = { evaluate_saved: "Evaluate saved responses", fine_tune: "Fine-tuning execution",
    weight_edit: "Weight editing execution", merge: "Variant composition" };
  for (const [id, label] of Object.entries(names)) {
    const capability = state.workspace.capabilities[id];
    const row = node("div", undefined, "capability-row");
    row.append(node("span", label), node("span", capability?.enabled ? "Available" : "Runtime setup required",
      `badge${capability?.enabled ? " available" : ""}`));
    $("capability-list").append(row);
  }
  const selectedTemplate = $("gym-template").value;
  $("gym-template").replaceChildren();
  for (const template of state.workspace.gym_templates) {
    const option = node("option", template.name); option.value = template.id;
    $("gym-template").append(option);
  }
  if (state.workspace.gym_templates.some((item) => item.id === selectedTemplate)) $("gym-template").value = selectedTemplate;
  fillAssetOptions($("gym-dataset"), assets("dataset"), "Choose a dataset version");
  renderGymFields();
  fillAssetOptions($("evaluation-gym"), assets("gym").filter((asset) => asset.document.template === "json_exact"), "Choose an extraction gym");
  renderEvaluationEvidence();
  for (const workstream of ["fine_tune", "weight_edit", "merge"]) renderRecipe(workstream);
  controls();
}

function renderRuns(container, runs) {
  if (!runs.length) { empty(container, "No experiments yet. Evaluate recorded answers or prepare an improvement plan."); return; }
  container.replaceChildren();
  for (const run of runs) {
    const row = node("div", undefined, "list-row");
    const text = node("div");
    text.append(node("strong", run.spec?.name || run.id), node("p", `${run.state} · ${run.id}`));
    row.append(text, button("Inspect experiment", () => action(() => selectRun(run.id))));
    container.append(row);
  }
}

function renderComparison() {
  const rows = measurementRows(state.snapshot.runs);
  if (!rows.length) { empty($("comparison-table"), "Measured results will appear after an approved evaluation finishes."); return; }
  const table = node("table");
  const head = node("thead"); const header = node("tr");
  for (const text of ["Candidate", "Experiment", "Frozen context", "Metric", "Value", "Unit"]) header.append(node("th", text));
  head.append(header); table.append(head);
  const body = node("tbody");
  for (const row of rows) {
    const entry = node("tr");
    const context = row.comparisonKey ? `${assetName(row.gymId)} / ${assetName(row.datasetId)} / ${row.policy}` : "Context not recorded; do not compare";
    const metricNames = { json_exact_match: "Exact answer match", json_validity: "Valid JSON object" };
    const proportion = row.metric.unit === "ratio" && Object.hasOwn(metricNames, row.metric.name)
      && Number.isFinite(row.metric.value) && row.metric.value >= 0 && row.metric.value <= 1;
    const displayValue = proportion ? `${(row.metric.value * 100).toFixed(1)}%` : row.metric.value;
    for (const value of [assetName(row.modelId), row.runName, context,
      `${metricNames[row.metric.name] || row.metric.name} · ${row.metric.version}`,
      displayValue, proportion ? "% of validation examples" : row.metric.unit]) entry.append(node("td", String(value ?? "Not recorded")));
    body.append(entry);
  }
  table.append(body); $("comparison-table").replaceChildren(table);
}

function openDialog(title, submitText, build, onSubmit) {
  $("dialog-title").textContent = title;
  $("dialog-submit").textContent = submitText;
  $("dialog-fields").replaceChildren();
  $("dialog-error").hidden = true;
  dialogSubmit = onSubmit;
  build($("dialog-fields"));
  $("workspace-dialog").showModal();
  controls();
}

function field(container, id, label, { type = "text", value = "", required = false, help = "", options, rows = 4 } = {}) {
  const caption = node("label", label); caption.htmlFor = id;
  const input = node(options ? "select" : type === "textarea" ? "textarea" : "input");
  input.id = id;
  if (options) for (const [key, text] of options) { const option = node("option", text); option.value = key; input.append(option); }
  else if (type === "textarea") input.rows = rows;
  else input.type = type;
  if (!options || value !== "" || options.some(([key]) => key === "")) input.value = value;
  input.required = required;
  container.append(caption, input);
  if (help) container.append(node("p", help, "field-help"));
  return input;
}

function assetSelect(container, id, label, kind, required = true) {
  const input = field(container, id, label, { options: [], required });
  fillAssetOptions(input, assets(kind), `Choose ${kind} version`, false);
  return input;
}

function fileInput(container, target, help) {
  const input = field(container, `${target}-file`, "Choose a JSON / JSONL file", { type: "file", help });
  input.accept = ".json,.jsonl,application/json,text/plain";
  input.addEventListener("change", () => action(async () => {
    const file = input.files[0];
    if (!file) return;
    if (file.size > 240000) throw new Error("This inline import supports files up to 240 KB. Larger data needs a qualified artifact adapter.");
    $(target).value = await file.text();
    input.value = "";
    notify("File loaded for review. It has not been saved.");
  }));
}

async function saveAsset(kind, name, document, parent = null) {
  const result = await persist(`${projectPath()}/assets`, { kind, name, document, parent_asset_id: parent });
  $("workspace-dialog").close();
  await loadProject(state.projectId);
  notify(`${kind === "recipe" ? "Plan saved; no training was started." : "Asset version published."} ${result.name} · v${result.version}`);
  return result;
}

function createProject() {
  openDialog("Create a project", "Create project", (container) => {
    field(container, "new-project-name", "Project name", { required: true });
    field(container, "new-project-description", "What are you improving?", { type: "textarea", help: "Describe the model behavior or product outcome. You can connect multiple gyms to the same project." });
  }, async () => {
    const project = await persist("/v1/projects", { name: $("new-project-name").value, description: $("new-project-description").value });
    $("workspace-dialog").close();
    state.view = "overview";
    await loadWorkspace(project.id);
    notify("Project created. Add your model reference and define a gym.");
  });
}

function registerModel(parent = null) {
  openDialog(parent ? "Publish a model reference version" : "Register a model reference", parent ? "Publish next version" : "Register model", (container) => {
    field(container, "model-name", "Candidate name", { required: true, help: "For example: unchanged base or LoRA round one." });
    field(container, "model-id", "Model ID", { required: true, help: "A model/catalog identifier. This form does not read a server path or download weights." });
    field(container, "model-revision", "Revision / commit", { required: true });
    field(container, "model-source", "Identity source", { options: [
      ["reference", "Model reference (not checkpoint verified)"], ["recorded", "Candidate represented by recorded evidence"],
      ["software_fixture", "Software policy fixture"],
    ] });
    field(container, "model-digest", "Declared weights digest (optional)", { help: "A declaration is not an independent verification." });
    if (parent) {
      $("model-name").value = parent.name; $("model-name").readOnly = true;
      $("model-id").value = parent.document.model_id;
      $("model-revision").value = parent.document.revision;
      $("model-source").value = parent.document.source;
      $("model-digest").value = parent.document.weights_digest || "";
    }
  }, async () => {
    const document = { model_id: $("model-id").value, revision: $("model-revision").value, source: $("model-source").value };
    if ($("model-digest").value.trim()) document.weights_digest = $("model-digest").value.trim();
    await saveAsset("model", $("model-name").value, document, parent?.id || null);
  });
}

function importDataset(parent = null) {
  openDialog(parent ? "Publish a dataset version" : "Import a dataset", "Validate & publish version", (container) => {
    field(container, "dataset-name", "Dataset name", { required: true });
    const text = field(container, "dataset-text", "Examples · JSON or JSONL", { type: "textarea", required: true,
      help: 'Each example contains input text and an expected JSON object. Optional id and split fields are preserved. Example: {"input":"Total 42","expected":{"total":42},"split":"validation"}', rows: 7 });
    fileInput(container, text.id, "The file stays local until you explicitly publish it.");
    field(container, "dataset-input-field", "Input column", { value: "input", required: true });
    field(container, "dataset-expected-field", "Expected-answer column", { value: "expected", required: true });
    field(container, "dataset-default-split", "Missing split values use", { options: [["validation", "Validation"], ["train", "Training"]] });
    container.append(node("p", "Missing IDs are assigned in file order (example-1, example-2, ...). The server rejects duplicate IDs and normalized train/validation overlap. No sealed test data is created.", "field-help"));
    if (parent) {
      $("dataset-name").value = parent.name; $("dataset-name").readOnly = true;
      $("dataset-text").value = pretty(parent.document.examples);
      container.append(node("div", "Existing gyms and experiments stay pinned to the old dataset version. Publish a new gym version to evaluate the new data.", "callout"));
    }
  }, async () => {
    const document = datasetDocument($("dataset-text").value, { inputField: $("dataset-input-field").value,
      expectedField: $("dataset-expected-field").value, defaultSplit: $("dataset-default-split").value });
    await saveAsset("dataset", $("dataset-name").value, document, parent?.id || null);
  });
}

function importEvidence(parent = null) {
  openDialog(parent ? "Publish a recorded-evidence version" : "Import recorded candidate answers", "Publish evidence asset", (container) => {
    field(container, "evidence-name", "Evidence name", { required: true });
    assetSelect(container, "evidence-model", "Model / candidate version", "model");
    assetSelect(container, "evidence-dataset", "Dataset version", "dataset");
    field(container, "evidence-source", "Where did these responses come from?", { type: "textarea", required: true,
      help: "Record the original inference run, decoding settings and other relevant context. This remains user-supplied evidence, not fresh or independent inference." });
    const text = field(container, "evidence-text", "Saved responses · JSON or JSONL", { type: "textarea", required: true,
      help: 'One response per validation example: {"example_id":"example-1","response":"{\\"total\\":42}"}. Keep malformed model answers as strings; do not discard them.', rows: 7 });
    fileInput(container, text.id, "The server checks example IDs, dataset ownership and evaluation coverage.");
    if (parent) {
      $("evidence-name").value = parent.name; $("evidence-name").readOnly = true;
      $("evidence-model").value = parent.document.model_asset_id;
      $("evidence-dataset").value = parent.document.dataset_asset_id;
      $("evidence-source").value = parent.document.source_description;
      $("evidence-text").value = pretty(parent.document.rows);
    }
  }, async () => saveAsset("evidence", $("evidence-name").value, {
    model_asset_id: $("evidence-model").value, dataset_asset_id: $("evidence-dataset").value,
    source_description: $("evidence-source").value, rows: evidenceRows($("evidence-text").value),
  }, parent?.id || null));
}

function openAsset(asset) {
  openDialog(`${asset.name} · v${asset.version}`, "Close", (container) => {
    container.append(node("p", `${asset.kind} · immutable asset version`, "field-help"));
    container.append(node("p", `Content identity: ${asset.artifact_id}`, "contract"));
    if (asset.kind === "model") container.append(node("div", "Model identity is declared. This asset alone does not verify checkpoint bytes or enable execution.", "callout"));
    if (asset.kind === "dataset") {
      const table = node("table");
      const head = node("tr");
      for (const title of ["Example", "Split", "Input", "Expected"]) head.append(node("th", title));
      table.append(head);
      for (const example of asset.document.examples.slice(0, 20)) {
        const row = node("tr");
        for (const value of [example.id, example.split, example.input, pretty(example.expected)]) row.append(node("td", value));
        table.append(row);
      }
      const wrapper = node("div", undefined, "table-wrap"); wrapper.append(table); container.append(wrapper);
      container.append(node("p", "Showing up to 20 examples. The full immutable document is available below.", "field-help"));
    }
    if (asset.kind === "gym") container.append(button("Open in gym builder", () => {
      $("workspace-dialog").close(); state.gymParent = asset.id; $("gym-name").value = asset.name;
      $("gym-template").value = asset.document.template; $("gym-dataset").value = asset.document.dataset_asset_id || "";
      renderGymFields(asset.document.config); resetPreview(); showView("gyms");
    }, "quiet"));
    const latest = assets(asset.kind).filter((entry) => entry.name === asset.name)
      .every((entry) => entry.version <= asset.version);
    if (latest && ["model", "dataset", "evidence"].includes(asset.kind)) {
      container.append(button("Create next version", () => {
        $("workspace-dialog").close();
        ({ model: registerModel, dataset: importDataset, evidence: importEvidence })[asset.kind](asset);
      }, "quiet"));
    }
    if (asset.kind === "recipe") {
      const workstream = ["lora", "sft"].includes(asset.document.method) ? "fine_tune" : asset.document.method;
      const capability = state.workspace.capabilities[workstream];
      container.append(node("div", capability?.enabled ? "A qualified runtime is available. The server will still check this plan's exact inputs."
        : capability?.reason || "No qualified runtime is registered for this plan.", "callout"));
      const launch = button("Create experiment from this plan", () => action(async () => {
        const run = await persist(`${projectPath()}/experiments`, { workstream, name: asset.name,
          recipe_asset_id: asset.id });
        $("workspace-dialog").close(); await loadProject(state.projectId); await selectRun(run.id);
      }), "primary");
      launch.disabled = !capability?.enabled; container.append(launch);
    }
    const details = node("details"); details.append(node("summary", "Full asset document & lineage"), node("pre", pretty(asset)));
    container.append(details);
  }, async () => $("workspace-dialog").close());
}

function schemaField(container, id, label, schema, value) {
  const options = schema.enum ? schema.enum.map((entry) => [String(entry), String(entry)])
    : schema.type === "boolean" ? [["", "Use default"], ["true", "Yes"], ["false", "No"]] : undefined;
  const input = field(container, id, label.replaceAll("_", " "), { options,
    type: ["integer", "number"].includes(schema.type) ? "number" : "text",
    value: value ?? schema.default ?? "", help: schema.description || (schema.minimum !== undefined || schema.maximum !== undefined
      ? `Allowed: ${schema.minimum ?? "no minimum"} to ${schema.maximum ?? "no maximum"}.` : "Blank keeps the registered default.") });
  if (input.type === "number") {
    input.step = schema.type === "integer" ? "1" : "any";
    if (schema.minimum !== undefined) input.min = schema.minimum;
    if (schema.maximum !== undefined) input.max = schema.maximum;
  }
  input.dataset.parameter = label;
  return input;
}

function renderGymFields(config = {}) {
  const template = state.workspace.gym_templates.find((item) => item.id === $("gym-template").value);
  $("gym-config-fields").replaceChildren();
  if (!template) return;
  $("gym-template-description").textContent = template.description;
  const json = template.id === "json_exact";
  $("gym-dataset").hidden = $("gym-dataset-label").hidden = !json;
  $("gym-dataset").required = json;
  for (const [name, schema] of Object.entries(template.config_schema?.properties || {})) schemaField($("gym-config-fields"), `gym-config-${name}`, name, schema, config[name]);
  $("gym-contract").textContent = template.id === "json_exact"
    ? "Structured-answer task · uses this frozen dataset version. Manual preview does not call a model."
    : "Step-by-step task · trusted CPU preview with seeded reset and explicit episode termination. No external engine is launched.";
  $("preview-json-controls").hidden = !json;
  fillPreviewExamples();
}

function gymDefinition() {
  const template = state.workspace.gym_templates.find((item) => item.id === $("gym-template").value);
  if (!template) throw new Error("Choose an available environment template.");
  const config = {};
  for (const [name, schema] of Object.entries(template.config_schema?.properties || {})) {
    const input = $(`gym-config-${name}`);
    if (input.validity.badInput) throw new Error(`${name}: finish the numeric value.`);
    const value = parameterValue(input.value, schema, name);
    if (value !== undefined) config[name] = value;
  }
  const definition = { template: template.id, config };
  if (template.id === "json_exact") {
    if (!$("gym-dataset").value) throw new Error("Import and select a dataset version first.");
    definition.dataset_asset_id = $("gym-dataset").value;
  }
  return definition;
}

function fillPreviewExamples() {
  const dataset = assetById($("gym-dataset").value);
  $("preview-example").replaceChildren();
  for (const example of (dataset?.document.examples || []).filter((row) => row.split === "validation")) {
    const option = node("option", `${example.id} · ${example.split}`); option.value = example.id;
    $("preview-example").append(option);
  }
}

function resetPreview() {
  previewGeneration++;
  state.actions = []; state.preview = null;
  for (const id of ["preview-observation", "preview-board", "preview-actions", "preview-metrics"]) $(id).replaceChildren();
  $("preview-trace").textContent = "No preview yet.";
  $("preview-empty").hidden = false;
}

async function preview(actionValue) {
  const stamp = ++previewGeneration;
  const actions = actionValue === undefined ? [...state.actions] : [...state.actions, actionValue];
  const definition = gymDefinition();
  const seed = Number($("preview-seed").value);
  if (!Number.isSafeInteger(seed) || seed < 0 || seed > 2147483647) throw new Error("Choose a nonnegative integer seed.");
  const result = await request(`${projectPath()}/preview`, {
    definition, actions, seed, ...(definition.template === "json_exact"
      ? { example_id: $("preview-example").value, response: $("preview-response").value } : {}),
  });
  if (stamp !== previewGeneration) return;
  state.actions = actions; state.preview = result;
  $("preview-empty").hidden = true;
  $("preview-trace").textContent = pretty(result);
  const observation = node("div", undefined, "observation");
  observation.append(node("small", "Current observation · manual preview"), node("strong", typeof result.observation === "string"
    ? result.observation : pretty(result.observation)));
  if (result.terminated || result.truncated) observation.append(node("small", result.terminated ? "Episode terminated" : "Episode truncated"));
  $("preview-observation").replaceChildren(observation);
  $("preview-board").replaceChildren();
  if (definition.template === "stateful_counter") {
    const target = definition.config.target ?? 2;
    for (let position = 0; position <= Math.min(target, 32); position++) {
      const cell = node("div", position === target ? `${position} · goal` : String(position),
        `counter-cell${result.observation?.position === position ? " current" : ""}${position === target ? " goal" : ""}`);
      $("preview-board").append(cell);
    }
  }
  $("preview-actions").replaceChildren();
  for (const value of result.allowed_actions || []) {
    const control = button(typeof value === "number" ? value > 0 ? "Move +1" : "Move -1" : String(value),
      () => action(() => preview(value)), "quiet");
    control.disabled = result.terminated || result.truncated;
    $("preview-actions").append(control);
  }
  $("preview-metrics").replaceChildren();
  for (const metric of result.metrics || []) {
    const row = node("div", undefined, "measurement");
    row.append(node("span", `${metric.name} · v${metric.version}`), node("strong", `${metric.value} ${metric.unit}`));
    $("preview-metrics").append(row);
  }
}

function renderEvaluationEvidence() {
  const gym = assetById($("evaluation-gym").value);
  fillAssetOptions($("evaluation-evidence"), assets("evidence").filter((asset) =>
    !gym || asset.document.dataset_asset_id === gym.document.dataset_asset_id), "", false);
}

function renderRecipe(workstream) {
  const prefix = workstream === "fine_tune" ? "fine-tune" : workstream === "weight_edit" ? "weight-edit" : "merge";
  const capability = state.workspace.capabilities[workstream];
  $(`${prefix}-readiness`).textContent = capability?.enabled
    ? "Execution is available on a qualified runtime. Save a versioned plan, then review the actual experiment before authorizing it."
    : "Plan preparation is available. Model execution is not connected for this workstream in this deployment; saving a plan will not start a job.";
  $(`${prefix}-readiness`).classList.toggle("warning", !capability?.enabled);
  const methods = (state.workspace.recipe_methods || []).filter((method) =>
    workstream === "fine_tune" ? ["lora", "sft"].includes(method.id) : method.id === workstream);
  const host = $(`${prefix}-form`); host.replaceChildren();
  if (!methods.length) { empty(host, "No registered recipe schema is available. The server must publish a supported method before a plan can be prepared."); return; }
  const form = node("form", undefined, "panel"); form.id = `${prefix}-recipe-form`;
  host.append(form);
  if (!capability?.enabled) {
    const setup = node("details");
    setup.append(node("summary", "Runtime setup requirements"), node("p", capability?.reason || "No qualified runtime is registered.", "field-help"));
    form.append(setup);
  }
  field(form, `${prefix}-name`, "Plan name", { required: true });
  assetSelect(form, `${prefix}-model`, "Base model / candidate version", "model");
  if (workstream === "merge") {
    const sources = assetSelect(form, `${prefix}-sources`, "Source model versions (including the base)", "model");
    sources.multiple = true; sources.size = 5;
    form.append(node("p", methods[0].source_model_asset_ids_schema?.description
      || "Select two to sixteen distinct project models, including the base.", "field-help"));
  }
  field(form, `${prefix}-method`, "Method", { options: methods.map((method) => [method.id, method.name]) });
  if (workstream !== "merge") assetSelect(form, `${prefix}-dataset`, "Training / edit-example dataset version", "dataset", workstream === "fine_tune");
  const gyms = assetSelect(form, `${prefix}-gyms`, "Evaluation and preservation gyms", "gym");
  gyms.multiple = true; gyms.size = 4;
  const parameterHost = node("div", undefined, "form-grid"); form.append(parameterHost);
  const renderParameters = () => {
    parameterHost.replaceChildren();
    const method = methods.find((item) => item.id === $(`${prefix}-method`).value);
    for (const [name, schema] of Object.entries(method.parameters_schema?.properties || {})) schemaField(parameterHost, `${prefix}-parameter-${name}`, name, schema);
  };
  $(`${prefix}-method`).addEventListener("change", renderParameters);
  renderParameters();
  form.append(node("p", "The selected gyms declare evaluation requirements; they are not a guarantee of preservation or improvement. Compatibility, budgets and qualification are checked again at dispatch.", "field-help"));
  const advanced = node("details"); advanced.append(node("summary", "Advanced: existing qualified proof adapter inputs"));
  field(advanced, `${prefix}-proof`, "Existing proof-spec artifact ID", { help: "Optional content-addressed input for an already qualified adapter. No code or executable paths are accepted." });
  field(advanced, `${prefix}-role`, "Candidate role in that proof specification");
  field(advanced, `${prefix}-runtime`, "Qualified runtime profile ID", { help: "Entering a profile name does not qualify or enable it." });
  form.append(advanced);
  const save = node("button", "Save improvement plan", "primary"); save.type = "submit";
  const actions = node("div", undefined, "form-actions"); actions.append(save); form.append(actions);
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    action(async () => {
      const method = methods.find((item) => item.id === $(`${prefix}-method`).value);
      const parameters = {};
      for (const [name, schema] of Object.entries(method.parameters_schema?.properties || {})) {
        const input = $(`${prefix}-parameter-${name}`);
        if (input.validity.badInput) throw new Error(`${name}: finish the numeric value.`);
        const value = parameterValue(input.value, schema, name);
        if (value !== undefined) parameters[name] = value;
      }
      const document = { method: method.id, model_asset_id: $(`${prefix}-model`).value,
        gym_asset_ids: selectedValues(gyms).filter(Boolean), parameters };
      if (workstream === "merge") {
        const sources = selectedValues($(`${prefix}-sources`)).filter(Boolean);
        const schema = method.source_model_asset_ids_schema;
        if (sources.length < (schema?.minItems ?? 2) || sources.length > (schema?.maxItems ?? 16)
            || !sources.includes(document.model_asset_id)) {
          throw new Error("Choose two to sixteen source models, including the selected base.");
        }
        document.source_model_asset_ids = sources;
      }
      if (workstream !== "merge" && $(`${prefix}-dataset`).value) document.dataset_asset_id = $(`${prefix}-dataset`).value;
      for (const [fieldId, key] of [["proof", "proof_spec_artifact_id"], ["role", "role"], ["runtime", "runtime_profile"]]) {
        const value = $(`${prefix}-${fieldId}`).value.trim();
        if (value) document[key] = value;
      }
      const result = await saveAsset("recipe", $(`${prefix}-name`).value, document);
      openAsset(result);
    });
  });
  assetTable($(`${prefix}-recipes`), assets("recipe").filter((asset) =>
    workstream === "fine_tune" ? ["lora", "sft"].includes(asset.document.method) : asset.document.method === workstream));
}

async function selectRun(runId) {
  const run = await request(`/v1/runs/${part(runId)}`);
  if (!state.snapshot.runs.some((item) => item.id === run.id)) throw new Error("That experiment is not linked to the active project.");
  state.run = run;
  showView("experiments");
  renderRun();
  $("experiment-events").textContent = pretty(await request(`/v1/runs/${part(run.id)}/events`));
}

function renderRun() {
  const run = state.run;
  if (!run || !state.token) return;
  $("experiment-detail").hidden = false;
  $("experiment-title").textContent = run.spec.name;
  $("experiment-state").textContent = `${run.state} · ${run.id}`;
  $("experiment-scope").textContent = run.stages.some((stage) => stage.operation === "analysis.saved_json")
    ? "Recorded-output analysis" : "Inspect runtime evidence";
  $("experiment-next").textContent = nextAction(run);
  $("experiment-plan").replaceChildren(stat("Execution", run.outcome?.execution || run.state),
    stat("Measured quality", run.outcome?.quality || "Not evaluated"),
    stat("Release", run.outcome?.release_authorized === false ? "Not authorized" : "Not established"));
  if (run.plan) for (const [name, value] of Object.entries(run.plan.limits)) $("experiment-plan").append(stat(name.replaceAll("_", " "), value));
  $("experiment-json").textContent = pretty(run);
  $("experiment-stages").replaceChildren();
  for (const stage of run.stages) {
    const row = node("div", undefined, "stage-row");
    const labels = { "analysis.saved_json": "Score recorded answers", "proof.train": "Train candidate",
      "proof.export": "Prepare serving artifact", "proof.identity": "Verify candidate identity", "proof.score": "Evaluate candidate" };
    const definition = run.spec.stages.find((entry) => entry.id === stage.id);
    const modelId = stage.result?.model_asset_id || definition?.config?.model_asset_id;
    const candidate = run.spec.candidates.find((entry) => entry.id === stage.candidate_id);
    row.append(node("strong", `${labels[stage.operation] || stage.operation} · ${stage.state}`),
      node("p", `${modelId ? assetName(modelId) : candidate?.model.id || stage.candidate_id} · ${stage.attempts?.length || 0} attempts`));
    for (const attempt of stage.attempts || []) if (attempt.error) row.append(node("p", `${attempt.error.code}: ${attempt.error.message}`, "error-text"));
    if (stage.artifact_id) row.append(button("Inspect result artifact", () => action(async () => {
      const artifact = await request(`/v1/artifacts/${part(stage.artifact_id)}`);
      openDialog("Result evidence", "Close", (container) => container.append(node("pre", pretty(artifact))), async () => $("workspace-dialog").close());
    })));
    $("experiment-stages").append(row);
  }
  state.snapshot.runs = state.snapshot.runs.map((item) => item.id === run.id ? run : item);
  renderRuns($("experiment-list"), state.snapshot.runs);
  renderComparison();
  controls();
}

$("workspace-login-form").addEventListener("submit", (event) => {
  event.preventDefault();
  action(async () => {
    state.token = $("workspace-token").value.trim(); $("workspace-token").value = "";
    try {
      const hash = location.hash.split("/");
      state.view = views.has(hash[2]) ? hash[2] : "overview";
      await loadWorkspace(hash[1] ? decodeURIComponent(hash[1]) : undefined);
      $("workspace-login").hidden = true; $("product-workspace").hidden = false;
      $("lock").hidden = $("refresh-workspace").hidden = false;
      notify("Workspace unlocked. No model or training job has been started.");
    } catch (error) { lock(); throw error; }
  });
});
$("lock").addEventListener("click", () => { lock(); notify("Workspace locked. Server-side experiments are unchanged."); });
$("refresh-workspace").addEventListener("click", () => action(async () => {
  const runId = state.run?.id; await loadWorkspace(); if (runId) await selectRun(runId);
  notify("Workspace refreshed. Refresh never starts or retries experiments.");
}));
$("project-switch").addEventListener("change", () => {
  if ($("project-switch").value) action(() => loadProject($("project-switch").value));
});
for (const id of ["new-project", "first-project"]) $(id).addEventListener("click", createProject);
for (const element of document.querySelectorAll("[data-view], [data-go]")) element.addEventListener("click", () => showView(element.dataset.view || element.dataset.go));
$("add-model").addEventListener("click", () => registerModel());
$("add-dataset").addEventListener("click", () => importDataset());
$("add-evidence").addEventListener("click", () => importEvidence());
for (const id of ["close-dialog", "cancel-dialog"]) $(id).addEventListener("click", () => $("workspace-dialog").close());
$("dialog-form").addEventListener("submit", (event) => { event.preventDefault(); if (dialogSubmit) action(dialogSubmit); });
$("gym-template").addEventListener("change", () => { renderGymFields(); resetPreview(); });
$("gym-dataset").addEventListener("change", () => { fillPreviewExamples(); resetPreview(); });
$("gym-form").addEventListener("input", resetPreview);
$("preview-seed").addEventListener("input", resetPreview);
$("preview-example").addEventListener("change", resetPreview);
$("try-gym").addEventListener("click", () => action(() => preview()));
$("score-preview").addEventListener("click", () => action(() => preview()));
$("reset-preview").addEventListener("click", resetPreview);
$("gym-form").addEventListener("submit", (event) => {
  event.preventDefault();
  action(async () => {
    const name = $("gym-name").value;
    const parent = assetById(state.gymParent);
    const result = await saveAsset("gym", name, gymDefinition(), parent?.name === name ? parent.id : null);
    state.gymParent = result.id;
  });
});
$("evaluation-gym").addEventListener("change", renderEvaluationEvidence);
$("evaluation-form").addEventListener("submit", (event) => {
  event.preventDefault();
  action(async () => {
    const run = await persist(`${projectPath()}/experiments`, { workstream: "evaluate_saved",
      name: $("evaluation-name").value, gym_asset_id: $("evaluation-gym").value,
      evidence_asset_ids: selectedValues($("evaluation-evidence")) });
    await loadProject(state.projectId);
    await selectRun(run.id);
    notify("Evaluation draft created. Review and authorize the exact plan before explicitly starting it.");
  });
});
$("review-experiment").addEventListener("click", () => action(async () => {
  state.run = await request(`/v1/runs/${part(state.run.id)}/plan`, {}); renderRun();
  notify("Plan generated. Review the resource ceilings and frozen asset identities before authorizing.");
}));
$("approve-experiment").addEventListener("click", () => action(async () => {
  if (!window.confirm("Authorize this exact plan and its resource limits? This does not start execution.")) return;
  const id = state.run.id;
  await request(`/v1/runs/${part(id)}/approve`, { plan_digest: state.run.plan.digest, limits: state.run.plan.limits });
  state.run = await request(`/v1/runs/${part(id)}`); renderRun();
}));
$("start-experiment").addEventListener("click", () => action(async () => {
  if (!window.confirm("Start this approved experiment now? The worker will execute its planned operations.")) return;
  const run = await persist(`/v1/runs/${part(state.run.id)}/start`, { approval_id: state.run.approval.id });
  state.run = run; renderRun(); notify("Start accepted. Follow the durable result; execution completion is not release approval.");
}));
$("cancel-experiment").addEventListener("click", () => action(async () => {
  if (!window.confirm("Record cancellation? Already-issued work may still need to settle safely.")) return;
  state.run = await request(`/v1/runs/${part(state.run.id)}/cancel`, {}); renderRun();
}));
$("download-bundle").addEventListener("click", () => action(async () => {
  const bundle = await request(`${projectPath()}/bundle`);
  const url = URL.createObjectURL(new Blob([pretty(bundle)], { type: "application/json" }));
  const link = node("a"); link.href = url; link.download = "neural-foundry-project-evidence.json";
  document.body.append(link); link.click(); link.remove(); URL.revokeObjectURL(url);
  notify("Evidence manifest prepared; browser download requested. This is not a signed release or model-weight export.");
}));

async function checkReady() {
  try {
    const response = await fetch("/ready", { cache: "no-store", credentials: "omit", signal: AbortSignal.timeout(10000) });
    const result = await response.json(); state.ready = response.ok && result.ready === true;
    $("connection").textContent = state.ready ? "Workspace ready" : "Service not ready · actions paused";
  } catch { state.ready = false; $("connection").textContent = "Connection unavailable · actions paused"; }
  controls();
}
let polling = false;
setInterval(async () => {
  if (!state.token || !state.run || state.busy || polling || document.hidden) return;
  polling = true; const generation = state.generation; const id = state.run.id;
  try {
    const run = await request(`/v1/runs/${part(id)}`);
    if (generation !== state.generation || !state.token || state.run?.id !== id) return;
    state.run = run; renderRun();
    $("experiment-events").textContent = pretty(await request(`/v1/runs/${part(id)}/events`));
  } catch (error) { notify(`Live refresh paused: ${error.message}`, true); }
  finally { polling = false; }
}, 3000);
setInterval(checkReady, 15000);
controls();
checkReady();

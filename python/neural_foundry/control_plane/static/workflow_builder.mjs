export function parseSetting(raw, schema, label) {
  if (raw.trim() === "") return undefined;
  let value;
  if (schema.type === "array") {
    const items = raw.replace(/\r\n/g, "\n").split(schema.items.type === "string" ? "\n" : ",");
    value = items.map((item) => schema.items.type === "string"
      ? item : parseSetting(item, schema.items, label));
    if (value.some((item) => item === undefined)) throw new Error(`${label}: remove empty entries.`);
    if (value.length < (schema.minItems ?? 0) || value.length > (schema.maxItems ?? Infinity)) {
      throw new Error(`${label}: enter ${schema.minItems ?? 0} to ${schema.maxItems ?? "the allowed number of"} entries.`);
    }
    if (schema.items.type === "string" && value.some((item) => item.length > schema.items.maxLength)) {
      throw new Error(`${label}: an entry exceeds ${schema.items.maxLength} characters.`);
    }
  } else if (schema.type === "integer" || schema.type === "number") {
    value = Number(raw);
    if (!Number.isFinite(value) || (schema.type === "integer" && !Number.isSafeInteger(value))) {
      throw new Error(`${label}: enter a finite ${schema.type}.`);
    }
    if (value < (schema.minimum ?? -Infinity) || value > (schema.maximum ?? Infinity)
        || (schema.exclusiveMinimum !== undefined && value <= schema.exclusiveMinimum)) {
      throw new Error(`${label}: value is outside the allowed range.`);
    }
  } else {
    throw new Error(`${label}: this setting requires the advanced JSON editor.`);
  }
  return value;
}

export function supportsGuided(template, plugins) {
  const spec = template.spec;
  if (spec?.version !== 1 || spec.candidates?.length !== 1 || !spec.stages?.length) return false;
  return spec.stages.every((stage) => {
    const plugin = plugins.find((item) => item.id === stage.operation);
    if (!plugin?.enabled || stage.candidate_id !== spec.candidates[0].id) return false;
    return Object.values(plugin.config_schema?.properties || {}).every((schema) =>
      ["integer", "number"].includes(schema.type)
      || (schema.type === "array" && ["integer", "number", "string"].includes(schema.items?.type)));
  });
}

export function buildExperiment({ name, candidateIds, workloads, limits = {} }) {
  name = name.trim();
  if (!name || name.length > 128) throw new Error("Give your experiment a name (1-128 characters).");
  if (!workloads.length) throw new Error("Choose at least one available workload.");
  if (!candidateIds.length || candidateIds.length > 16) throw new Error("Add between 1 and 16 candidates.");
  const ids = candidateIds.map((id) => id.trim());
  if (ids.some((id) => !/^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$/.test(id))) {
    throw new Error("Candidate labels must start with a letter or number; use letters, numbers, _, ., :, or -.");
  }
  if (new Set(ids).size !== ids.length) throw new Error("Give each candidate a unique label.");
  const definition = workloads[0].template.spec.candidates[0];
  for (const { template } of workloads) {
    const other = template.spec.candidates[0];
    if (template.spec.candidates.length !== 1 || other.model.id !== definition.model.id
        || other.model.revision !== definition.model.revision
        || other.method !== definition.method) {
      throw new Error("These workloads need different candidate contracts. Use the advanced JSON editor.");
    }
  }
  const stages = [];
  ids.forEach((id, candidateIndex) => workloads.forEach(({ template, configs = {} }, workloadIndex) => {
    const mapping = new Map(template.spec.stages.map((stage, index) =>
      [stage.id, `c${candidateIndex + 1}-w${workloadIndex + 1}-s${index + 1}`]));
    for (const stage of template.spec.stages) {
      if (stage.depends_on.some((dependency) => !mapping.has(dependency))) {
        throw new Error("The workload has an unknown dependency. Inspect its advanced specification.");
      }
      stages.push({
        ...structuredClone(stage), id: mapping.get(stage.id), candidate_id: id,
        depends_on: stage.depends_on.map((dependency) => mapping.get(dependency)),
        config: { ...structuredClone(stage.config), ...configs[stage.id] },
        limits: { ...stage.limits, ...limits },
      });
    }
  }));
  if (stages.length > 64) throw new Error("This selection exceeds 64 stages. Use fewer candidates or workloads.");
  return { version: 1, name, purpose: "development",
    candidates: ids.map((id) => ({ ...structuredClone(definition), id })), stages };
}

export function nextAction(run) {
  if (!run) return "Choose an experiment, or create one from the workload library.";
  switch (run.state) {
    case "created": case "draft": return "Generate a plan to review the exact stages and resource limits. Nothing is running yet.";
    case "planned": return "Review the plan below, then approve it. Approval does not start execution.";
    case "approved": return "Your plan is approved. Select Start approved run when you are ready.";
    case "queued": return "Waiting for a worker. You can refresh safely; do not create a duplicate run.";
    case "running": return "The worker is executing your approved stages. Inspect progress and evidence below.";
    case "needs_attention": return "Execution is paused and will not retry automatically. Read the recorded reason; review it before creating a fresh experiment.";
    case "failed": return "Execution failed. Inspect the failed stage and its recorded error before trying a new experiment.";
    case "cancelled": return "Cancellation was recorded. Already-issued work may still need to settle; inspect attempts and events.";
    case "completed": return "Execution finished. Review measured quality separately; completion never authorizes release.";
    default: return `Inspect the recorded state: ${run.state}. No new action is authorized by this display.`;
  }
}

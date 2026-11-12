export function readRecords(text, collection) {
  if (!text.trim()) throw new Error("Choose a nonempty JSON or JSONL file.");
  let value;
  try {
    value = JSON.parse(text);
  } catch {
    value = text.split(/\r?\n/).filter((line) => line.trim()).map((line, index) => {
      try { return JSON.parse(line); }
      catch { throw new Error(`Line ${index + 1} is not valid JSON.`); }
    });
  }
  const rows = Array.isArray(value) ? value : value?.[collection] ?? [value];
  if (!Array.isArray(rows) || !rows.length || rows.some((row) => !row || Array.isArray(row) || typeof row !== "object")) {
    throw new Error("Import a nonempty array of example objects, or one object per line.");
  }
  return rows;
}

export function datasetDocument(text, { inputField = "input", expectedField = "expected", defaultSplit = "validation" } = {}) {
  const examples = readRecords(text, "examples").map((row, index) => {
    let expected = row[expectedField];
    if (typeof expected === "string") {
      try { expected = JSON.parse(expected); }
      catch { throw new Error(`Example ${index + 1}: the expected answer must be a JSON object.`); }
    }
    if (!expected || Array.isArray(expected) || typeof expected !== "object") {
      throw new Error(`Example ${index + 1}: the expected answer must be a JSON object.`);
    }
    if (typeof row[inputField] !== "string" || !row[inputField].trim()) {
      throw new Error(`Example ${index + 1}: "${inputField}" must contain the input text.`);
    }
    const split = row.split ?? defaultSplit;
    if (!["train", "validation"].includes(split)) {
      throw new Error(`Example ${index + 1}: choose train or validation. Sealed tests are not imported here.`);
    }
    return { id: row.id ?? `example-${index + 1}`, split, input: row[inputField], expected };
  });
  return { format: "json_examples_v1", examples };
}

export function evidenceRows(text) {
  return readRecords(text, "rows").map((row, index) => {
    if (typeof row.example_id !== "string" || typeof row.response !== "string") {
      throw new Error(`Response ${index + 1}: provide example_id and a response string. Invalid model JSON is retained as a string.`);
    }
    return { example_id: row.example_id, response: row.response };
  });
}

export function parameterValue(raw, schema, label) {
  if (raw === "") return undefined;
  if (schema.type === "boolean") {
    if (!["true", "false"].includes(raw)) throw new Error(`${label}: choose yes or no.`);
    return raw === "true";
  }
  if (schema.enum) {
    if (!schema.enum.includes(raw)) throw new Error(`${label}: choose a supported value.`);
    return raw;
  }
  if (schema.type === "integer" || schema.type === "number") {
    const value = Number(raw);
    if (!Number.isFinite(value) || (schema.type === "integer" && !Number.isSafeInteger(value))
        || value < (schema.minimum ?? -Infinity) || value > (schema.maximum ?? Infinity)
        || (schema.exclusiveMinimum !== undefined && value <= schema.exclusiveMinimum)) {
      throw new Error(`${label}: enter a number within the displayed bounds.`);
    }
    return value;
  }
  if (schema.type === "string") return raw;
  throw new Error(`${label}: the registered parameter type is not supported by this editor.`);
}

export function measurementRows(runs) {
  return runs.flatMap((run) => (run.stages || []).flatMap((stage) => {
    const result = stage.result;
    if (!Array.isArray(result?.metrics)) return [];
    return result.metrics.map((metric) => ({
      runId: run.id, runName: run.spec?.name || run.id, stageId: stage.id,
      modelId: result.model_asset_id || stage.candidate_id,
      gymId: result.gym_asset_id ?? null, datasetId: result.dataset_asset_id ?? null,
      policy: result.metric_policy ?? null, metric,
      comparisonKey: result.gym_asset_id && result.dataset_asset_id && result.metric_policy
        ? JSON.stringify([result.gym_asset_id, result.dataset_asset_id, result.metric_policy,
          metric.name, metric.version, metric.unit]) : null,
    }));
  }));
}

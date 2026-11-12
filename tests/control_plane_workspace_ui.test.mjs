import test from "node:test";
import assert from "node:assert/strict";
import { datasetDocument, evidenceRows, measurementRows, parameterValue } from "../python/neural_foundry/control_plane/static/workspace_logic.mjs";

test("dataset import maps columns and makes default IDs and splits explicit", () => {
  const document = datasetDocument('{"prompt":"Receipt A","answer":"{\\"total\\":42}"}\n{"prompt":"Receipt B","answer":{"total":8},"split":"train","id":"b"}',
    { inputField: "prompt", expectedField: "answer", defaultSplit: "validation" });
  assert.deepEqual(document.examples, [
    { id: "example-1", split: "validation", input: "Receipt A", expected: { total: 42 } },
    { id: "b", split: "train", input: "Receipt B", expected: { total: 8 } },
  ]);
  assert.throws(() => datasetDocument('[{"input":"x","expected":[]}]'), /JSON object/);
  assert.throws(() => datasetDocument('[{"input":"x","expected":{},"split":"test"}]'), /Sealed tests/);
});

test("saved invalid answers are preserved rather than discarded during import", () => {
  assert.deepEqual(evidenceRows('[{"example_id":"a","response":"not json"}]'),
    [{ example_id: "a", response: "not json" }]);
  assert.throws(() => evidenceRows('[{"example_id":"a","response":{}}]'), /response string/);
});

test("comparison identity requires the same frozen gym, dataset and metric policy", () => {
  const metric = { name: "exact", version: "1", value: 1, unit: "ratio" };
  const run = (gym, dataset) => ({ id: gym, stages: [{ id: "score", result: {
    gym_asset_id: gym, dataset_asset_id: dataset, metric_policy: "json_exact_v1", metrics: [metric],
  } }] });
  const rows = measurementRows([run("gym-1", "data-1"), run("gym-2", "data-1"), run("gym-1", "data-2"),
    { id: "legacy", stages: [{ result: { metrics: [metric] } }] }]);
  assert.notEqual(rows[0].comparisonKey, rows[1].comparisonKey);
  assert.notEqual(rows[0].comparisonKey, rows[2].comparisonKey);
  assert.equal(rows[3].comparisonKey, null);
});

test("recipe parameters respect registered numeric and enum constraints", () => {
  assert.equal(parameterValue("", { type: "integer" }, "Epochs"), undefined);
  assert.equal(parameterValue("8", { type: "integer", minimum: 1, maximum: 64 }, "Rank"), 8);
  assert.throws(() => parameterValue("NaN", { type: "number" }, "Rate"), /number/);
  assert.throws(() => parameterValue("1.2", { type: "integer" }, "Epochs"), /number/);
  assert.throws(() => parameterValue("unknown", { enum: ["ties", "dare"] }, "Method"), /supported/);
});

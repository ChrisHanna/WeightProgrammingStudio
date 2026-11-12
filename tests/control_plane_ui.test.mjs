import test from "node:test";
import assert from "node:assert/strict";
import { buildExperiment, parseSetting, supportsGuided, nextAction } from "../python/neural_foundry/control_plane/static/workflow_builder.mjs";

const template = { spec: { version: 1, candidates: [
  { id: "original", model: { id: "fixture", revision: "1" }, method: "fixture" },
], stages: [
  { id: "train", operation: "software.training", candidate_id: "original", depends_on: [], config: { steps: 4 }, limits: { steps: 8 } },
  { id: "export", operation: "software.export", candidate_id: "original", depends_on: ["train"], config: {}, limits: {} },
] } };

test("multiple workloads and candidates retain separate dependency graphs without changing templates", () => {
  const before = structuredClone(template);
  const spec = buildExperiment({ name: "Comparison", candidateIds: ["base", "variant"],
    workloads: [{ template, configs: { train: { steps: 6 } } }, { template }],
    limits: { wall_seconds: 10 } });
  assert.equal(spec.stages.length, 8);
  assert.equal(new Set(spec.stages.map((stage) => stage.id)).size, 8);
  assert.deepEqual(spec.stages[1].depends_on, [spec.stages[0].id]);
  assert.deepEqual(spec.stages[5].depends_on, [spec.stages[4].id]);
  assert.equal(spec.stages[4].candidate_id, "variant");
  assert.equal(spec.stages[0].config.steps, 6);
  assert.equal(spec.stages[2].config.steps, 4);
  assert.deepEqual(spec.stages[0].limits, { steps: 8, wall_seconds: 10 });
  assert.deepEqual(template, before);
});

test("ambiguous candidates, incompatible contracts and oversized graphs are rejected", () => {
  const args = { name: "Test", candidateIds: ["a"], workloads: [{ template }] };
  assert.throws(() => buildExperiment({ ...args, candidateIds: ["a", " a "] }), /unique/);
  assert.throws(() => buildExperiment({ ...args, candidateIds: ["not a valid id"] }), /Candidate labels/);
  assert.throws(() => buildExperiment({ ...args, workloads: [] }), /workload/);
  assert.throws(() => buildExperiment({ ...args, candidateIds: Array.from({ length: 16 }, (_, i) => `c${i}`),
    workloads: Array(3).fill({ template }) }), /64 stages/);
  const other = structuredClone(template);
  other.spec.candidates[0].model = { revision: "1", id: "fixture" };
  assert.equal(buildExperiment({ ...args, workloads: [{ template }, { template: other }] }).stages.length, 4);
  other.spec.candidates[0].model.revision = "2";
  assert.throws(() => buildExperiment({ ...args, workloads: [{ template }, { template: other }] }), /different candidate contracts/);
});

test("settings preserve omitted defaults and reject nonfinite, fractional and empty numeric values", () => {
  const schema = { type: "integer", minimum: 0, maximum: 3 };
  assert.equal(parseSetting("", schema, "Seeds"), undefined);
  assert.equal(parseSetting("0", schema, "Seeds"), 0);
  for (const value of ["Infinity", "NaN", "1.5", "4"]) {
    assert.throws(() => parseSetting(value, schema, "Seeds"), /Seeds/);
  }
  const array = { type: "array", minItems: 1, maxItems: 2, items: schema };
  assert.deepEqual(parseSetting("0, 2", array, "Seeds"), [0, 2]);
  assert.throws(() => parseSetting("0,", array, "Seeds"), /empty/);
  assert.throws(() => parseSetting("0,1,2", array, "Seeds"), /entries/);
  assert.throws(() => parseSetting("0", { type: "number", exclusiveMinimum: 0 }, "Rate"), /range/);
});

test("text rows are not silently trimmed or dropped", () => {
  const schema = { type: "array", minItems: 1, maxItems: 3, items: { type: "string", maxLength: 10 } };
  assert.deepEqual(parseSetting("  first\n\nlast", schema, "Rows"), ["  first", "", "last"]);
  assert.throws(() => parseSetting("a".repeat(11), schema, "Rows"), /characters/);
});

test("disabled or unsupported plugins cannot enter the guided builder", () => {
  const plugins = template.spec.stages.map((stage) => ({ id: stage.operation, enabled: true,
    config_schema: { properties: {} } }));
  assert.equal(supportsGuided(template, plugins), true);
  plugins[0].enabled = false;
  assert.equal(supportsGuided(template, plugins), false);
  plugins[0].enabled = true;
  plugins[0].config_schema.properties = { unsupported: { type: "object" } };
  assert.equal(supportsGuided(template, plugins), false);
  assert.match(nextAction({ state: "needs_attention" }), /will not retry automatically/);
});

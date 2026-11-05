# Weight IR v1

The authoritative parser is `nf_weight_ir::program::parse_program`. It accepts bounded canonical JSON v1 and returns an immutable validated program. Parsing never reads an artifact, opens a path, downloads a model, performs inference, or modifies a checkpoint. Valid structure is not execution permission or scientific acceptance.

## Required root fields

| Field | Wire contract |
| --- | --- |
| `ir_version`, `opset_version` | Integer 1; unknown versions are rejected |
| `base_model` | Exact digests for checkpoint, manifest, parameter set, architecture graph, config, and explicit nullable tokenizer |
| `objective` | Digests of frozen contract, scientific objective, experiment protocol, and Transformation Search Space |
| `operations` | Ordered instructions, each with a unique ID, distinct earlier dependencies, and a typed `operation` |
| `artifacts` | Logical IDs mapped to immutable Safetensors operand descriptors |
| `preconditions` | Exact source tensor name/role/shape/dtype/before-digest/storage-group declarations |
| `required_capabilities` | Explicit bounded logical capability names |
| `constraints` | Gradient-free update policy, shape policy, parameter/artifact/temporary-memory bounds, numerical policy |
| `effects` | Reads, parameter writes, buffer writes, graph/config/tokenizer changes |
| `preservation` | Experimental, bounded error, or exact under named conditions |
| `verification` | Portfolio/protocol, exact gym bindings with validation/scorer/split identities, smoke and parity suite digests |
| `lineage` | Parent candidate IDs, hypothesis ID, and full method/inspection/fitting record |
| `extensions` | Optional bounded namespaced strings; no executable or permission semantics |

Unknown fields are rejected in nested records and operation variants. A tensor region has `tensor`, `start`, and `shape`; starts and widths are unsigned 64-bit integers with checked ends and rank at most 32. Scalar regions have empty start/shape vectors. Element products cannot overflow. Actual shape compatibility and scope are checked against trusted inventory at compilation.

Each operand descriptor declares the exact artifact digest, tensor name within that Safetensors artifact, dtype, shape, tensor data bytes, artifact bytes, access class, and evidence provenance. Shape/dtype must agree with the declared byte count. Holdout operands and holdout provenance are forbidden. Declared metadata is not evidence that the actual referenced artifact exists or has those values; resolution validates it later.

## Operation set 1

All operations include a target region. No instruction accepts Python, shell expressions, URLs, or general filesystem paths.

| Type | Additional operands |
| --- | --- |
| `replace_slice` | `value` artifact ID |
| `add_delta` | `delta` artifact ID |
| `scale` | Finite `factor` |
| `mask` | `mask` artifact ID; binary values and dtype are checked during compilation/execution |
| `permute_axis` | Explicit axis and bijective permutation of that target axis |
| `linear_map` | `matrix` artifact ID and left/right side |
| `subspace_projection` | `basis` artifact ID, side, finite strength, nonnegative orthonormal tolerance |
| `low_rank_update` | Left/right artifact IDs and finite strength |
| `construct_recipe` | Named versioned recipe, specification digest, explicit input artifact IDs |

An operation must have its target declared in write effects; operations using old target values also require a matching read effect. All effect regions refer to source preconditions. Compiler passes additionally derive actual read/write effects, resolve aliases, validate operand dimensions/values, enforce budgets, and reject unsupported constructors/backends. Extra declarations do not grant rights beyond the frozen contract or search space.

`gradient_free` concerns candidate-producing updates, not a claim that the entire research pipeline used no derivatives. Non-writing attribution may be declared separately. Fitted/unknown candidate lineage cannot qualify as a gradient-free direct update. A fully gradient-free pipeline additionally requires the contract's strict inspection policy.

The numerical policy explicitly selects f32/f64 accumulation, nearest-ties-even output rounding, non-finite rejection, positive finite maximum parameter magnitude, and nonnegative finite backend tolerance. The contract/compiler can impose tighter bounds; a proposal cannot weaken a verifier by choosing permissive numbers. No implicit dtype casts or approximation claims are introduced by parsing.

## Preservation, verification, and controls

An exact claim names a versioned rule and condition artifact IDs. A bounded-error claim names a metric, tolerance, and protocol. These are unverified claims until the trusted compiler and evaluator supply evidence. Experimental claims assert no equivalence.

Verification uses immutable binding IDs, not late-resolved gym names. Mandatory study checks cannot be removed by editing this field. Final held-out results never become proposal inputs. The standard five-gym profile is enforced by the study contract rather than by requiring five bindings in every standalone IR document.

An explicit no-op may have no instructions or artifacts, no writes, and zero changed parameters in its method record. Its verification bindings and memory bounds remain required. It cannot satisfy a positive measured improvement threshold merely because parsing or replay succeeded.

## Bounds and compatibility

Default parsing limits are canonical JSON v1 limits. Additionally, v1 permits at most 1,024 instructions/artifacts/preconditions/effect regions/gym bindings/parent IDs, 128 capabilities, and 16 namespaced extension strings of at most 1,024 bytes each. A schema-valid program outside a frozen transformation domain is rejected by search-space enforcement. Unknown effectful instructions are never skipped. Registered recipe and backend support is checked separately; schema support does not advertise an implemented scientific engine.

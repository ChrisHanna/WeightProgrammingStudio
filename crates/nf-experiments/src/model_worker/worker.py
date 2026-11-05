"""Fixed trusted CPU model adapter. Never receives gym source or callbacks.

Invoked with isolated Python (-I -S); only operator-approved dependencies and
native extension paths are added. Checkpoints/configs remain inert data.
"""

# The real exec child cannot import dependencies, read a job or run model work
# until Rust owns its pidfd and commits any required coordinator registration.
# EOF (including controller death) is fail-closed, never permission to proceed.
import os

if os.read(0, 1) != b"G":
    os._exit(125)
os.close(0)

import importlib.util
import json
import resource
import sys
import time
from pathlib import Path

NATIVE_PATH, SITE, JOB = sys.argv[1:]
sys.path.insert(0, SITE)
spec = importlib.util.spec_from_file_location("_native", NATIVE_PATH)
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)
# Trusted selected adapter dependency; never a coordinator import.
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.set_grad_enabled(False)
if torch.__version__ != "2.9.1+cpu" or torch.cuda.is_available():
    raise RuntimeError("only pinned actual CPU Torch runtime is qualified")
root = Path(JOB)
request = json.loads((root / "request.json").read_bytes())


def probe():
    return {
        "schema_version": 1,
        "kind": "trusted_torch_cpu_probe",
        "torch_version": torch.__version__,
        "native_version": native.version(),
        "python": list(sys.version_info[:3]),
        "threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "kernel_threads": len(list(Path("/proc/self/task").iterdir())),
        "cpu_limit": list(resource.getrlimit(resource.RLIMIT_CPU)),
        "as_limit": list(resource.getrlimit(resource.RLIMIT_AS)),
        "fd_limit": list(resource.getrlimit(resource.RLIMIT_NOFILE)),
        "core_limit": list(resource.getrlimit(resource.RLIMIT_CORE)),
        "gradient_enabled": torch.is_grad_enabled(),
        "credential_marker_present": any(
            k in os.environ
            for k in (
                "AWS_ACCESS_KEY_ID",
                "AWS_SECRET_ACCESS_KEY",
                "COORDINATOR_TOKEN",
                "NF_HOST_SECRET",
            )
        ),
    }


class Linear(torch.nn.Module):
    def __init__(self, rows, columns, dtype, buffered):
        super().__init__()
        self.w = torch.nn.Parameter(torch.empty((rows, columns), dtype=dtype), requires_grad=False)
        self.tied = self.w
        if buffered:
            self.register_buffer("offset", torch.empty(rows, dtype=dtype))
            self.register_buffer("offset_alias", self.offset)
        self.buffered = buffered

    def forward(self, inputs):
        result = torch.mv(self.w, inputs)
        return result + self.offset if self.buffered else result


class Arithmetic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = torch.nn.Linear(2, 6, bias=False, dtype=torch.float64)
        self.heads = torch.nn.Linear(6, 5, bias=False, dtype=torch.float64)
        self.requires_grad_(False)

    def forward(self, inputs):
        return self.heads(torch.relu(self.hidden(inputs)))


def tensors(model):
    return {
        **{
            name: ("parameter", value)
            for name, value in model.named_parameters(remove_duplicate=False)
        },
        **{name: ("buffer", value) for name, value in model.named_buffers(remove_duplicate=False)},
    }


def observations(model):
    result = []
    groups = {}
    for name, (role, tensor) in sorted(tensors(model).items()):
        if tensor.device.type != "cpu" or not tensor.is_contiguous() or tensor.numel() == 0:
            raise RuntimeError("unqualified runtime tensor packing/device")
        key = (tensor.data_ptr(), tensor.numel() * tensor.element_size())
        group = groups.setdefault(key, f"physical_group_{len(groups)}")
        # Raw addresses NEVER cross IPC. Only equality-group labels do.
        data = bytes(tensor.detach().reshape(-1).view(torch.uint8).tolist())
        result.append(
            {
                "name": name,
                "role": role,
                "dtype": "f64" if tensor.dtype == torch.float64 else "f32",
                "shape": list(tensor.shape),
                "strides": list(tensor.stride()),
                "digest": native.content_digest(data),
                "group": group,
                "device": "cpu",
                "requires_grad": tensor.requires_grad,
                "grad_is_none": tensor.grad is None,
            }
        )
    return result


def edit(model, source, before, copies):
    """Fixed typed operations only. Never eval/exec, backward, optimizers or callbacks."""
    accumulation = {"f32": torch.float32, "f64": torch.float64}[
        request["numerical"]["accumulation_dtype"]
    ]
    allocated = 0
    trace_bytes = 0
    operand_bytes = 0
    staging_bytes = copies["logical_group_bytes"] * 4 + sum(
        len(v["hex"]) // 2 for v in request["operands"].values()
    )

    def allocate(value):
        nonlocal allocated
        allocated += value.numel() * value.element_size()
        if allocated + staging_bytes > request["allocation_budget"]:
            raise RuntimeError("explicit tensor allocation budget exceeded")
        return value

    def add(a, b):
        return allocate(torch.add(a, b))

    def mul(a, b):
        return allocate(torch.mul(a, b))

    def matrix(a, b):
        # Explicit accumulation order, no BLAS/FMA tolerance fallback.
        result = allocate(torch.zeros((a.shape[0], b.shape[1]), dtype=accumulation))
        for k in range(a.shape[1]):
            result = add(result, mul(a[:, k, None], b[k, None, :]))
        return result

    with torch.inference_mode():
        inference_mode = torch.is_inference_mode_enabled()
        operands = {}
        for name, spec in request["operands"].items():
            data = bytearray.fromhex(spec["hex"])
            operand_bytes += len(data)
            dtype = {"f32": torch.float32, "f64": torch.float64, "u8": torch.uint8}[spec["dtype"]]
            operands[name] = allocate(
                torch.frombuffer(data, dtype=dtype).clone().reshape(spec["shape"])
            )
        steps = []
        for instruction in request["instructions"]:
            op = instruction["operation"]
            target = op["target"]
            role, tensor = tensors(model)[target["tensor"]]
            if role != "parameter" or tensor.requires_grad or tensor.grad is not None:
                raise RuntimeError("only gradient-free parameter edits are registered")
            slices = tuple(slice(s, s + n) for s, n in zip(target["start"], target["shape"]))
            view = tensor[slices]
            old = allocate(view.to(dtype=accumulation, copy=True))
            kind = op["type"]

            def operand(name, operation=op):
                return allocate(operands[operation[name]].to(dtype=accumulation, copy=True))

            if kind == "replace_slice":
                result = operand("value")
            elif kind == "add_delta":
                result = add(old, operand("delta"))
            elif kind == "scale":
                result = mul(old, allocate(torch.tensor(op["factor"], dtype=accumulation)))
            elif kind == "mask":
                result = mul(old, operand("mask"))
            elif kind == "permute_axis":
                indices = allocate(torch.tensor(op["permutation"], dtype=torch.int64))
                result = allocate(torch.index_select(old, op["axis"], indices))
            elif kind == "linear_map":
                transform = operand("matrix")
                result = matrix(transform, old) if op["side"] == "left" else matrix(old, transform)
            elif kind == "low_rank_update":
                delta = matrix(operand("left"), operand("right"))
                result = add(
                    old, mul(allocate(torch.tensor(op["strength"], dtype=accumulation)), delta)
                )
            elif kind == "subspace_projection":
                basis = operand("basis")
                projected = (
                    matrix(basis, matrix(basis.T, old))
                    if op["side"] == "left"
                    else matrix(matrix(old, basis), basis.T)
                )
                strength = allocate(torch.tensor(op["strength"], dtype=accumulation))
                result = allocate(torch.sub(old, mul(strength, projected)))
            else:
                raise RuntimeError("unregistered operation; no training/update callback API")
            # Both precision conversions are EXPLICIT and bound to native policy.
            result = allocate(result.to(dtype=tensor.dtype, copy=True))
            if result.shape != view.shape or not torch.isfinite(result).all():
                raise RuntimeError("same-shape finite output required")
            view.copy_(result)
            state = observations(model)
            values = {}
            for name, (_, value) in sorted(tensors(model).items()):
                data = bytes(value.detach().reshape(-1).view(torch.uint8).tolist())
                trace_bytes += len(data)
                values[name] = data.hex()
            steps.append({"instruction": instruction["id"], "state": state, "values": values})
        outputs = [
            model(allocate(torch.tensor(vector, dtype=next(model.parameters()).dtype))).tolist()
            for vector in request["inputs"]
        ]
    return {
        "schema_version": 1,
        "kind": "trusted_torch_cpu_edit",
        "plan": request["plan"],
        "manifest": source.manifest_digest,
        "checkpoint": native.content_digest((root / "checkpoint.safetensors").read_bytes()),
        "inputs": native.content_digest(
            native.canonical_json(json.dumps(request["inputs"]).encode())
        ),
        "outputs": outputs,
        "numerical": request["numerical"],
        "torch_version": torch.__version__,
        "before": before,
        "after": observations(model),
        "steps": steps,
        "inference_mode": inference_mode,
        "gradient_enabled": torch.is_grad_enabled(),
        "training": model.training,
        "threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "kernel_threads": len(list(Path("/proc/self/task").iterdir())),
        "copies": copies,
        "edit_allocations": {
            "scope": "explicit_tensor_results_and_casts;not_all_interpreter_or_observation_allocations",
            "explicit_tensor_allocation_bytes": allocated,
            "operand_clone_bytes": operand_bytes,
            "trace_tensor_bytes": trace_bytes,
        },
    }


def load_model(directory, asset_map, seed):
    manifest_bytes = (directory / "manifest.json").read_bytes()
    checkpoint = (directory / "checkpoint.safetensors").read_bytes()
    assets = {
        name: (directory / "assets" / filename).read_bytes() for name, filename in asset_map.items()
    }
    source = native.load_source(manifest_bytes, checkpoint, assets)
    manifest = json.loads(source.manifest_json())
    factory = manifest["architecture"]["id"]
    torch.manual_seed(seed)
    specs = {
        **{name: {**value, "role": "parameter"} for name, value in manifest["parameters"].items()},
        **{name: {**value, "role": "buffer"} for name, value in manifest["buffers"].items()},
    }
    dtype = torch.float64 if next(iter(specs.values()))["dtype"] == "f64" else torch.float32
    if factory == "nf.arithmetic.relu":
        model = Arithmetic()
    elif factory in ("fixture.cpu", "nf.fixture.linear_buffered"):
        rows, columns = manifest["parameters"]["w"]["shape"]
        model = Linear(rows, columns, dtype, factory == "nf.fixture.linear_buffered")
    else:
        raise RuntimeError("factory is not registered")
    if set(tensors(model)) != set(specs):
        raise RuntimeError("actual factory tensor inventory differs")
    factory_spans = {
        (tensor.data_ptr(), tensor.numel() * tensor.element_size())
        for _, tensor in tensors(model).values()
    }
    factory_bytes = sum(span[1] for span in factory_spans)
    logical_groups = {}
    group_roles = {}
    group_bytes = 0
    for name, spec in sorted(specs.items()):
        group = spec["storage_group"]
        group_roles.setdefault(group, set()).add(spec["role"])
        if group not in logical_groups:
            data = source.tensor_bytes(name)
            group_bytes += len(data)
            # Both mutable staging and clone are owned copies, not source views.
            staging = bytearray(data)
            logical_groups[group] = (
                torch.frombuffer(staging, dtype=dtype).clone().reshape(spec["shape"])
            )
        loaded = logical_groups[group]
        if spec["role"] == "parameter":
            parameter = torch.nn.Parameter(loaded, requires_grad=False)
            if name in ("w", "tied"):
                # The trusted factory ALWAYS ties these slots, not a checkpoint claim.
                if name == "w":
                    model.w = parameter
                    model.tied = model.w
            elif name == "hidden.weight":
                model.hidden.weight = parameter
            elif name == "heads.weight":
                model.heads.weight = parameter
            else:
                raise RuntimeError("unknown parameter slot")
        elif name == "offset":
            model.offset = loaded
            model.offset_alias = model.offset
        elif name != "offset_alias":
            raise RuntimeError("unknown buffer slot")
    if any(len(roles) != 1 for roles in group_roles.values()):
        raise RuntimeError("cross-role alias packing is unsupported")
    model.eval()
    copies = {
        "scope": "explicit_owned_transfer_staging_clone_and_factory_copies;not_all_interpreter_allocations",
        "logical_group_bytes": group_bytes,
        "factory_tensor_bytes": factory_bytes,
        "native_tensor_bytes_copy": group_bytes,
        "mutable_staging_bytes": group_bytes,
        "torch_clone_bytes": group_bytes,
        "input_tensor_bytes": sum(len(v) for v in request["inputs"])
        * (8 if dtype == torch.float64 else 4),
        "zero_copy": False,
    }
    return source, model, copies, dtype


def execute():
    seeds = request.get("seeds", [0] * len(request["inputs"]))
    if len(seeds) != len(request["inputs"]):
        raise RuntimeError("one seed per input required")
    source, model, copies, dtype = load_model(root, request["assets"], seeds[0])
    before = observations(model)
    if request["mode"] == "edit":
        return edit(model, source, before, copies)
    if request["mode"] != "infer":
        raise RuntimeError("unregistered trusted worker request mode")
    start = time.clock_gettime(time.CLOCK_BOOTTIME)
    with torch.inference_mode():
        inference_mode = torch.is_inference_mode_enabled()
        outputs = []
        for vector, seed in zip(request["inputs"], seeds):
            torch.manual_seed(seed)
            outputs.append(model(torch.tensor(vector, dtype=dtype, device="cpu")).tolist())
    after = observations(model)
    if before != after:
        raise RuntimeError("inference changed parameters, buffers or aliases")
    return {
        "schema_version": 2,
        "kind": "trusted_torch_cpu_inference",
        "seeds": seeds,
        "factory": json.loads(source.manifest_json())["architecture"]["id"],
        "manifest": source.manifest_digest,
        "parameters": source.parameter_digest,
        "checkpoint": native.content_digest((root / "checkpoint.safetensors").read_bytes()),
        "inputs": native.content_digest(
            native.canonical_json(json.dumps(request["inputs"]).encode())
        ),
        "torch_version": torch.__version__,
        "outputs": outputs,
        "before": before,
        "after": after,
        "inference_mode": inference_mode,
        "gradient_enabled": torch.is_grad_enabled(),
        "training": model.training,
        "threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "kernel_threads": len(list(Path("/proc/self/task").iterdir())),
        "forward_wall_ms": int((time.clock_gettime(time.CLOCK_BOOTTIME) - start) * 1000),
        "copies": copies,
        "assurance": "trusted_fixed_model_worker_not_untrusted_gym_sandbox",
    }


def measurement():
    seeds = request["seeds"]
    original, left, left_copies, _ = load_model(root, request["assets"], seeds[0])
    candidate, right, right_copies, _ = load_model(
        Path(request["candidate_job"]), request["candidate_assets"], seeds[0]
    )
    models = [left, right]
    before = [observations(model) for model in models]
    samples = []

    def clock():
        return time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)

    with torch.inference_mode():
        inference_mode = torch.is_inference_mode_enabled()
        for ordinal, arm in enumerate(request["schedule"]):
            outputs = []
            forward_ns = 0
            start = clock()
            for vector, seed in zip(request["inputs"], seeds):
                torch.manual_seed(seed)
                inputs = torch.tensor(vector, dtype=torch.float64, device="cpu")
                begin = clock()
                output = models[arm](inputs)
                end = clock()
                if end <= begin:
                    raise RuntimeError("nonpositive monotonic forward interval")
                forward_ns += end - begin
                outputs.append(output.tolist())
            finish = clock()
            output_digest = native.content_digest(
                native.canonical_json(json.dumps(outputs, allow_nan=False).encode())
            )
            status = Path("/proc/self/status").read_text()
            rss = next(line for line in status.splitlines() if line.startswith("VmRSS:"))
            if rss.split()[2] != "kB":
                raise RuntimeError("unknown kernel RSS unit")
            samples.append(
                {
                    "ordinal": ordinal,
                    "arm": arm,
                    "warmup": ordinal // 2 < request["warmup"],
                    "forward_ns": forward_ns,
                    "end_to_end_ns": finish - start,
                    "process_rss_bytes": int(rss.split()[1]) * 1024,
                    "outputs": output_digest,
                }
            )
    after = [observations(model) for model in models]
    if before != after:
        raise RuntimeError("measurement mutated model state")
    affinity = next(
        line.split(":", 1)[1].strip()
        for line in Path("/proc/self/status").read_text().splitlines()
        if line.startswith("Cpus_allowed_list:")
    )
    return {
        "schema_version": 1,
        "kind": "trusted_cpu_measurement",
        "protocol": request["protocol"],
        "manifests": [original.manifest_digest, candidate.manifest_digest],
        "before": before,
        "after": after,
        "samples": samples,
        "clock": "linux_clock_monotonic_raw_ns",
        "clock_resolution_ns": max(
            1, int(time.clock_getres(time.CLOCK_MONOTONIC_RAW) * 1000000000)
        ),
        "cpu_affinity": affinity,
        "torch_version": torch.__version__,
        "inference_mode": inference_mode,
        "gradient_enabled": torch.is_grad_enabled(),
        "training": [m.training for m in models],
        "threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "kernel_threads": len(list(Path("/proc/self/task").iterdir())),
        "process_high_water_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "device_allocator_peak_bytes": None,
        "device_peak_bytes": None,
        "copies": [left_copies, right_copies],
        "input_tensor_allocation_bytes": len(samples) * sum(len(v) for v in request["inputs"]) * 8,
        "memory_scope": "whole_paired_process_co_resident_models_interpreter_and_runtime;not_per_model",
        "forward_boundary": "sum_model_call_intervals_excludes_seed_input_tensor_output_copy_and_between_call_loop",
        "end_to_end_boundary": "in_process_numeric_request_includes_seed_input_tensor_forward_output_copy;excludes_model_load_and_process_startup",
    }


result = (
    probe()
    if request["mode"] == "probe"
    else measurement()
    if request["mode"] == "measure"
    else execute()
)
print(json.dumps(result, allow_nan=False, separators=(",", ":")), flush=True)

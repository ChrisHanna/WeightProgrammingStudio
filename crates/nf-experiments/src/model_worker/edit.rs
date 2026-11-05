//! Paired, strict-bit-parity trusted Torch execution of immutable compiled plans.
//! No executable DTO, callback, optimizer, or training request is accepted here.
use super::*;
use nf_core::schemas::Validated;
use nf_surgeon::transaction::{
    execute_cancellable, ExecutionLimits, SmokeSuite, VerifiedCandidate,
};
use nf_weight_ir::compilation::{CompiledPlan, TORCH_CPU_BACKEND_ID};
use nf_weight_ir::program::{OperationKind, TensorRegion};

#[derive(Debug, Clone, Serialize)]
pub struct EditReceipt {
    pub schema_version: u32,
    pub backend: String,
    pub runtime: ContentDigest,
    pub plan: ContentDigest,
    pub candidate: ContentDigest,
    pub status: String,
    pub parity: String,
    pub outputs_eligible: bool,
    pub promotion_status: String,
    pub kernel_cpu_ms: u64,
    pub kernel_peak_rss_bytes: u64,
    pub wall_ms: u64,
    pub measurement_scope: String,
    pub native_verification: nf_surgeon::transaction::TransactionReceipt,
    pub child_report: Value,
}
/// Cannot be deserialized or constructed from a self-reported receipt.
pub struct VerifiedEdit {
    candidate: VerifiedCandidate,
    receipt: EditReceipt,
}
impl VerifiedEdit {
    pub fn into_parts(self) -> (VerifiedCandidate, EditReceipt) {
        (self.candidate, self.receipt)
    }
}
fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8] = b"0123456789abcdef";
    bytes
        .iter()
        .flat_map(|b| {
            [
                DIGITS[(b >> 4) as usize] as char,
                DIGITS[(b & 15) as usize] as char,
            ]
        })
        .collect()
}
fn unhex(value: &Value, bytes: usize) -> Result<Vec<u8>> {
    let text = value
        .as_str()
        .ok_or_else(|| error(ErrorCode::InvalidValue, "hex bytes missing"))?;
    if text.len() != bytes * 2 {
        return Err(error(
            ErrorCode::IntegrityMismatch,
            "returned tensor byte length differs",
        ));
    }
    let nibble = |b| match b {
        b'0'..=b'9' => Ok(b - b'0'),
        b'a'..=b'f' => Ok(b - b'a' + 10),
        _ => Err(error(ErrorCode::InvalidValue, "noncanonical tensor hex")),
    };
    text.as_bytes()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|p| Ok(nibble(p[0])? * 16 + nibble(p[1])?))
        .collect()
}
fn inside(mut index: usize, shape: &[u64], region: &TensorRegion) -> bool {
    for axis in (0..shape.len()).rev() {
        let coordinate = index as u64 % shape[axis];
        index /= shape[axis] as usize;
        if coordinate < region.start[axis] || coordinate >= region.start[axis] + region.shape[axis]
        {
            return false;
        }
    }
    true
}
impl QualifiedTorch {
    pub fn edit(
        &self,
        plan: &CompiledPlan,
        source: &SourceState,
        smoke: &Validated<SmokeSuite>,
        limits: Limits,
        cancel: &CancellationToken,
    ) -> Result<VerifiedEdit> {
        limits.validate()?;
        cancel.check()?;
        if plan.backend().id != TORCH_CPU_BACKEND_ID || plan.backend().version != "1" {
            return Err(error(
                ErrorCode::PermissionDenied,
                "compiled policy must explicitly authorize nf.trusted_torch_cpu v1",
            ));
        }
        if plan.instructions().len() > 16
            || plan
                .instructions()
                .iter()
                .any(|i| i.operation.kind() == OperationKind::ConstructRecipe)
        {
            return Err(error(ErrorCode::UnsupportedOperation, "Torch profile supports at most 16 same-shape tensor instructions, not constructors/training"));
        }
        let names = source
            .manifest()
            .parameters
            .iter()
            .chain(source.manifest().buffers.iter());
        let logical_bytes: u64 = names
            .map(|(_, spec)| spec.byte_len())
            .collect::<Result<Vec<_>>>()?
            .iter()
            .sum();
        if logical_bytes.saturating_mul(plan.instructions().len() as u64 + 1) > 65_536 {
            return Err(error(
                ErrorCode::LimitExceeded,
                "bounded full-state step trace exceeds 64 KiB",
            ));
        }
        let inputs: Vec<_> = smoke.probes.iter().map(|p| p.input.clone()).collect();
        preflight(source, &inputs)?;
        // Independent native execution performs stale-source, smoke, method,
        // ancestry and final coordinate checks. Nothing is published yet.
        let reference =
            execute_cancellable(plan, source, smoke, ExecutionLimits::default(), cancel)?;
        let operand_bytes: usize = plan.operands().values().map(|v| v.data().len()).sum();
        if operand_bytes > 65_536 {
            return Err(error(
                ErrorCode::LimitExceeded,
                "Torch operand transfer exceeds 64 KiB",
            ));
        }
        let operands: BTreeMap<_,_> = plan.operands().iter().map(|(id,v)| (id, json!({"dtype":v.descriptor().dtype,"shape":v.descriptor().shape,"hex":hex(v.data())}))).collect();
        let (job, assets) = self.stage_source(source)?;
        let request = json!({"schema_version":1,"mode":"edit","assets":assets,"inputs":inputs,"plan":plan.digest(),"instructions":plan.instructions(),"numerical":plan.numerical(),"allocation_budget":plan.program().constraints.max_temporary_bytes.min(67_108_864),"operands":operands});
        record(&job.join("request.json"), &request)?;
        record(
            &job.join("intent.json"),
            &json!({"plan":plan.digest(),"runtime":self.identity(),"status":"incomplete","outputs_eligible":false}),
        )?;
        let captured = self.launch(&job, &limits, cancel)?;
        record(
            &job.join("execution.json"),
            &json!({"status":captured.status,"cleanup_confirmed":true,"cpu_ms":captured.cpu_ms,"peak_rss_bytes":captured.rss_bytes,"wall_ms":captured.wall_ms,"stdout_digest":ContentDigest::of_bytes(&captured.stdout),"stderr_digest":ContentDigest::of_bytes(&captured.stderr),"outputs_eligible":false}),
        )?;
        if !captured.success {
            return Err(error(
                ErrorCode::TransactionAborted,
                "Torch edit process failed or exceeded its bounds; no candidate admitted",
            ));
        }
        let report = CanonicalDocument::parse(
            &captured.stdout,
            JsonLimits {
                max_bytes: limits.output_bytes,
                max_depth: 32,
                max_nodes: 20000,
            },
        )?;
        verify_edit(plan, source, &reference, &inputs, report.value())?;
        cancel.check()?;
        source.verify_integrity()?;
        self.verify_runtime()?;
        let receipt = EditReceipt {
            schema_version:1,backend:TORCH_CPU_BACKEND_ID.into(),runtime:self.identity.clone(),plan:plan.digest().clone(),candidate:reference.digest().clone(),
            status:"verified_torch_edit".into(),parity:"strict_full_bytes_against_independent_native_reference;no_tolerance_fallback".into(),outputs_eligible:false,promotion_status:"not_evaluated".into(),
            kernel_cpu_ms:captured.cpu_ms,kernel_peak_rss_bytes:captured.rss_bytes,wall_ms:captured.wall_ms,
            measurement_scope:"wait4_whole_single_trusted_process_all_threads;AS_hard_cap_not_RSS_limit;native_reference_and_runtime_setup_cost_additional_unknown".into(),
            native_verification:reference.receipt().clone(),child_report:report.value().clone(),
        };
        record(&job.join("completion.json"), &receipt)?;
        Ok(VerifiedEdit {
            candidate: reference,
            receipt,
        })
    }
}
fn verify_edit(
    plan: &CompiledPlan,
    source: &SourceState,
    reference: &VerifiedCandidate,
    inputs: &[Vec<f64>],
    report: &Value,
) -> Result<()> {
    if report["schema_version"] != 1
        || report["kind"] != "trusted_torch_cpu_edit"
        || report["plan"].as_str() != Some(plan.digest().as_str())
        || report["manifest"].as_str() != Some(source.manifest_digest().as_str())
        || report["checkpoint"].as_str()
            != Some(ContentDigest::of_bytes(source.checkpoint_bytes()).as_str())
        || report["torch_version"] != "2.9.1+cpu"
        || report["inference_mode"] != true
        || report["gradient_enabled"] != false
        || report["training"] != false
        || report["threads"] != 1
        || report["interop_threads"] != 1
        || report["kernel_threads"] != 1
        || report["numerical"] != serde_json::to_value(plan.numerical()).unwrap()
        || report["copies"]["zero_copy"] != false
    {
        return Err(error(
            ErrorCode::IntegrityMismatch,
            "Torch edit identity/mode/numerical receipt differs",
        ));
    }
    verify_state(source, &report["before"], None)?;
    let steps = report["steps"]
        .as_array()
        .ok_or_else(|| error(ErrorCode::IntegrityMismatch, "Torch step trace missing"))?;
    if steps.len() != plan.instructions().len() {
        return Err(error(
            ErrorCode::IntegrityMismatch,
            "Torch instruction trace differs",
        ));
    }
    let specs: BTreeMap<_, _> = source
        .manifest()
        .parameters
        .iter()
        .chain(source.manifest().buffers.iter())
        .collect();
    let mut previous: BTreeMap<String, Vec<u8>> = specs
        .keys()
        .map(|name| Ok(((*name).clone(), source.tensor_bytes(name)?)))
        .collect::<Result<_>>()?;
    let mut trace_bytes = 0;
    for (instruction, step) in plan.instructions().iter().zip(steps) {
        if step["instruction"] != instruction.id {
            return Err(error(
                ErrorCode::IntegrityMismatch,
                "Torch step order differs",
            ));
        }
        let values = step["values"]
            .as_object()
            .ok_or_else(|| error(ErrorCode::InvalidValue, "Torch state bytes missing"))?;
        if values.len() != specs.len() {
            return Err(error(
                ErrorCode::IntegrityMismatch,
                "Torch returned extra or missing slots",
            ));
        }
        let mut next = BTreeMap::new();
        let mut hashes = BTreeMap::new();
        let target = instruction.operation.target();
        let target_spec = specs
            .get(&target.tensor)
            .ok_or_else(|| error(ErrorCode::SourceMismatch, "target disappeared"))?;
        for (name, spec) in &specs {
            let data = unhex(&values[name.as_str()], spec.byte_len()? as usize)?;
            trace_bytes += data.len();
            let width = spec.dtype.byte_width() as usize;
            for (index, (a, b)) in previous[name.as_str()]
                .chunks_exact(width)
                .zip(data.chunks_exact(width))
                .enumerate()
            {
                let value = if spec.dtype == DType::F32 {
                    f32::from_le_bytes(b.try_into().unwrap()) as f64
                } else {
                    f64::from_le_bytes(b.try_into().unwrap())
                };
                if !value.is_finite() || value.abs() > plan.numerical().max_abs_parameter {
                    return Err(error(
                        ErrorCode::NonFinite,
                        "Torch intermediate violates finite/magnitude invariants",
                    ));
                }
                if a != b
                    && (source.manifest().buffers.contains_key(*name)
                        || spec.storage_group != target_spec.storage_group
                        || !inside(index, &spec.shape, target))
                {
                    return Err(error(
                        ErrorCode::ScopeViolation,
                        "Torch step mutated coordinates outside its own effective alias mask",
                    ));
                }
            }
            hashes.insert((*name).clone(), ContentDigest::of_bytes(&data));
            next.insert((*name).clone(), data);
        }
        verify_state(source, &step["state"], Some(&hashes))?;
        previous = next;
    }
    if &report["after"]
        != steps
            .last()
            .map(|s| &s["state"])
            .unwrap_or(&report["before"])
    {
        return Err(error(
            ErrorCode::IntegrityMismatch,
            "Torch forward changed edited model state",
        ));
    }
    let actual_outputs: Vec<Vec<f64>> =
        serde_json::from_value(report["outputs"].clone()).map_err(|_| {
            error(
                ErrorCode::InvalidValue,
                "Torch edited smoke outputs missing",
            )
        })?;
    if report["inputs"].as_str() != Some(ContentDigest::of_bytes(&canonical(&inputs)?).as_str()) {
        return Err(error(
            ErrorCode::IntegrityMismatch,
            "edited smoke input identity differs",
        ));
    }
    let materialized =
        reference.materialize_source(nf_surgeon::transaction::MaterializationLimits::default())?;
    let expected_outputs = preflight(&materialized, inputs)?;
    let tolerance = if specs.values().any(|s| s.dtype == DType::F32) {
        1e-5
    } else {
        1e-10
    };
    if actual_outputs.len() != expected_outputs.len()
        || actual_outputs.iter().zip(&expected_outputs).any(|(a, b)| {
            a.len() != b.len()
                || a.iter()
                    .zip(b)
                    .any(|(a, b)| !a.is_finite() || (a - b).abs() > tolerance)
        })
    {
        return Err(error(
            ErrorCode::SmokeFailed,
            "actual edited Torch forward differs from independent native smoke",
        ));
    }
    for (name, data) in previous {
        if data != reference.tensor_bytes(&name)? {
            return Err(error(ErrorCode::IntegrityMismatch,"Torch candidate differs from independent native reference bytes; no silent tolerance/cast fallback"));
        }
    }
    let mut groups = BTreeMap::new();
    for spec in specs.values() {
        groups.insert(&spec.storage_group, spec.byte_len()?);
    }
    let group_bytes: u64 = groups.values().sum();
    for field in [
        "native_tensor_bytes_copy",
        "mutable_staging_bytes",
        "torch_clone_bytes",
        "factory_tensor_bytes",
        "logical_group_bytes",
    ] {
        if report["copies"][field].as_u64() != Some(group_bytes) {
            return Err(error(
                ErrorCode::IntegrityMismatch,
                "Torch source copy receipt differs",
            ));
        }
    }
    if report["edit_allocations"]["trace_tensor_bytes"].as_u64() != Some(trace_bytes as u64)
        || report["edit_allocations"]["operand_clone_bytes"].as_u64()
            != Some(
                plan.operands()
                    .values()
                    .map(|v| v.data().len() as u64)
                    .sum(),
            )
        || report["edit_allocations"]["explicit_tensor_allocation_bytes"]
            .as_u64()
            .is_none_or(|n| {
                n.saturating_add(group_bytes * 4).saturating_add(
                    plan.operands()
                        .values()
                        .map(|v| v.data().len() as u64)
                        .sum::<u64>(),
                ) > plan
                    .program()
                    .constraints
                    .max_temporary_bytes
                    .min(67_108_864)
            })
        || report["edit_allocations"]["scope"]
            != "explicit_tensor_results_and_casts;not_all_interpreter_or_observation_allocations"
    {
        return Err(error(
            ErrorCode::IntegrityMismatch,
            "Torch edit allocation/copy receipt differs",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn full_byte_decoder_and_coordinate_masks_are_strict() {
        assert_eq!(unhex(&json!("00ff01"), 3).unwrap(), vec![0, 255, 1]);
        for value in [json!("0F"), json!("x0"), json!([0]), json!("0000")] {
            assert!(unhex(&value, 1).is_err());
        }
        let mask = TensorRegion {
            tensor: "w".into(),
            start: vec![0, 1],
            shape: vec![2, 1],
        };
        assert!(!inside(0, &[2, 2], &mask));
        assert!(inside(1, &[2, 2], &mask));
        assert!(!inside(2, &[2, 2], &mask));
        assert!(inside(3, &[2, 2], &mask));
    }
}

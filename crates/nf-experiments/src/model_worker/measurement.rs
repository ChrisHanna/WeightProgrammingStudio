//! Frozen CPU timing/RSS protocols. Measurement is not scientific acceptance.
use super::*;
use nf_core::study::{MeasurementOrder, MeasurementScope, MetricUnit, Statistic};
use serde::{Deserialize, Serialize};

fn bounded_read(path: &str, max: u64) -> Result<String> {
    let mut bytes = Vec::new();
    fs::File::open(path)
        .map_err(io)?
        .take(max + 1)
        .read_to_end(&mut bytes)
        .map_err(io)?;
    if bytes.len() as u64 > max {
        return Err(error(
            ErrorCode::LimitExceeded,
            "CPU inventory exceeds bound",
        ));
    }
    String::from_utf8(bytes)
        .map_err(|_| error(ErrorCode::InvalidValue, "CPU inventory is not text"))
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Environment {
    pub schema_version: u32,
    pub runtime: ContentDigest,
    pub device: String,
    pub physical_device: ContentDigest,
    pub driver: ContentDigest,
    pub affinity: String,
    pub backend: String,
    pub framework: String,
    pub clock: String,
    pub physical_identity_scope: String,
}
impl QualifiedTorch {
    pub fn measurement_environment(&self) -> Result<Environment> {
        self.verify_runtime()?;
        let cpu = bounded_read("/proc/cpuinfo", 1_048_576)?;
        let properties: Vec<_> = cpu
            .split("\n\n")
            .next()
            .unwrap_or("")
            .lines()
            .filter(|l| {
                [
                    "vendor_id",
                    "cpu family",
                    "model\t",
                    "model name",
                    "stepping",
                    "flags",
                ]
                .iter()
                .any(|k| l.starts_with(k))
            })
            .collect();
        if properties.is_empty() {
            return Err(error(
                ErrorCode::UnsupportedCapability,
                "CPU inventory unavailable",
            ));
        }
        let status = bounded_read("/proc/self/status", 65536)?;
        let affinity = status
            .lines()
            .find_map(|s| s.strip_prefix("Cpus_allowed_list:"))
            .ok_or_else(|| error(ErrorCode::UnsupportedCapability, "CPU affinity unavailable"))?
            .trim()
            .to_owned();
        let physical_device = ContentDigest::of_bytes(&canonical(&(
            properties,
            &affinity,
            bounded_read("/proc/sys/kernel/random/boot_id", 4096)?,
        ))?);
        let driver = ContentDigest::of_bytes(&canonical(&(
            std::env::consts::ARCH,
            std::env::consts::OS,
            bounded_read("/proc/sys/kernel/osrelease", 4096)?,
        ))?);
        Ok(Environment {
            schema_version: 1,
            runtime: self.identity.clone(),
            device: "cpu".into(),
            physical_device,
            driver,
            affinity,
            backend: "nf.trusted_torch_cpu.v1".into(),
            framework: "torch2.9.1+cpu".into(),
            clock: "linux_clock_monotonic_raw_ns".into(),
            physical_identity_scope:
                "boot_visible_cpu_inventory_and_affinity_not_unique_chip_serial".into(),
        })
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Metric {
    pub name: String,
    pub scope: MeasurementScope,
    pub unit: MetricUnit,
    pub statistic: Statistic,
    pub required: bool,
    pub ceiling: Option<f64>,
    pub max_regression: Option<f64>,
    pub min_gain: Option<f64>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Protocol {
    pub schema_version: u32,
    pub environment: Environment,
    pub workload: ContentDigest,
    pub inputs: Vec<Vec<f64>>,
    pub seeds: Vec<u64>,
    pub batch_size: u32,
    pub concurrency: u32,
    pub precision: String,
    pub cache: String,
    pub state_reset: String,
    pub warmup: u32,
    pub repetitions: u32,
    pub order: MeasurementOrder,
    pub order_seed: u64,
    pub metrics: Vec<Metric>,
}
impl Protocol {
    fn validate(&self) -> Result<()> {
        if self.schema_version != 1
            || self.inputs.is_empty()
            || self.inputs.len() > 16
            || self.seeds.len() != self.inputs.len()
            || self.batch_size != 1
            || self.concurrency != 1
            || self.precision != "f64"
            || self.cache != "disabled"
            || self.state_reset != "stateless_checked_before_after"
            || self.warmup > 32
            || !(3..=128).contains(&self.repetitions)
            || self.metrics.is_empty()
            || self.metrics.len() > 16
        {
            return Err(error(
                ErrorCode::InvalidValue,
                "bounded explicit CPU workload/warmup/repetition/settings required",
            ));
        }
        if self.workload
            != ContentDigest::of_bytes(&canonical(&(
                &self.inputs,
                &self.seeds,
                self.batch_size,
                self.concurrency,
                &self.precision,
                &self.cache,
                &self.state_reset,
            ))?)
        {
            return Err(error(
                ErrorCode::SourceMismatch,
                "workload identity differs from exact inputs/seeds/settings",
            ));
        }
        let mut names = BTreeSet::new();
        for m in &self.metrics {
            if m.name.is_empty()
                || m.name.len() > 128
                || !names.insert(&m.name)
                || m.name.chars().any(char::is_control)
                || [m.ceiling, m.max_regression, m.min_gain]
                    .into_iter()
                    .flatten()
                    .any(|v| !v.is_finite() || v < 0.0)
                || m.min_gain == Some(0.0)
                || (m.required
                    && m.ceiling.is_none()
                    && m.max_regression.is_none()
                    && m.min_gain.is_none())
                || m.statistic == Statistic::Exact
            {
                return Err(error(
                    ErrorCode::InvalidValue,
                    "distinct finite metric/statistic/positive gain/hard threshold required",
                ));
            }
            match m.scope {
                MeasurementScope::ForwardLatency | MeasurementScope::EndToEndLatency => {
                    if !matches!(
                        m.unit,
                        MetricUnit::Nanoseconds | MetricUnit::Milliseconds | MetricUnit::Seconds
                    ) {
                        return Err(error(
                            ErrorCode::InvalidValue,
                            "latency needs exact time units",
                        ));
                    }
                }
                _ => {
                    if m.unit != MetricUnit::Bytes {
                        return Err(error(
                            ErrorCode::InvalidValue,
                            "RSS/device memory requires bytes, not a latency or allocation proxy",
                        ));
                    }
                }
            }
        }
        Ok(())
    }
    fn schedule(&self) -> Vec<u8> {
        let mut state = self.order_seed ^ 0x9e3779b97f4a7c15;
        let mut schedule = Vec::new();
        for round in 0..self.warmup + self.repetitions {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let first = match self.order {
                MeasurementOrder::PairedInterleaved => (round % 2) as u8,
                MeasurementOrder::RandomizedInterleaved => (state & 1) as u8,
            };
            schedule.extend([first, 1 - first]);
        }
        schedule
    }
}
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Sample {
    ordinal: usize,
    arm: u8,
    warmup: bool,
    forward_ns: u64,
    end_to_end_ns: u64,
    process_rss_bytes: u64,
    outputs: ContentDigest,
}
/// Native observed process completion is required; serialized samples cannot mint this.
pub struct Measurement {
    id: ContentDigest,
    report: Value,
}
impl Measurement {
    pub fn id(&self) -> &ContentDigest {
        &self.id
    }
    pub fn report(&self) -> &Value {
        &self.report
    }
}
fn statistic(values: &[u64], kind: Statistic) -> Result<f64> {
    if values.is_empty() {
        return Err(error(ErrorCode::InvalidValue, "empty measurement support"));
    }
    let mut values = values.to_vec();
    values.sort_unstable();
    let n = values.len();
    let value = match kind {
        Statistic::Mean => values.iter().map(|n| u128::from(*n)).sum::<u128>() as f64 / n as f64,
        Statistic::Median => {
            if n.is_multiple_of(2) {
                (u128::from(values[n / 2 - 1]) + u128::from(values[n / 2])) as f64 / 2.0
            } else {
                values[n / 2] as f64
            }
        }
        Statistic::P95 => values[(n * 95).div_ceil(100) - 1] as f64,
        Statistic::P99 => values[(n * 99).div_ceil(100) - 1] as f64,
        Statistic::Maximum => *values.last().unwrap() as f64,
        Statistic::Exact => {
            return Err(error(
                ErrorCode::UnsupportedCapability,
                "exact is not a repeated runtime statistic",
            ))
        }
    };
    Ok(value)
}
fn assess_metric(
    m: &Metric,
    original: Option<f64>,
    candidate: Option<f64>,
    identical: bool,
) -> Value {
    let gain = original.zip(candidate).map(|(a, b)| a - b);
    let status = if candidate.is_none() || original.is_none() {
        "unavailable"
    } else if identical && m.min_gain.is_some() {
        "no_op_cannot_satisfy_positive_gain"
    } else if m.scope == MeasurementScope::ProcessRss
        && (m.min_gain.is_some() || m.max_regression.is_some())
    {
        "inconclusive_co_resident_memory_attribution"
    } else if candidate.is_some_and(|v| m.ceiling.is_some_and(|n| v > n))
        || gain.is_some_and(|g| {
            m.max_regression.is_some_and(|n| g < -n) || m.min_gain.is_some_and(|n| g < n)
        })
    {
        "failed"
    } else {
        "passed"
    };
    json!({"name":m.name,"scope":m.scope,"unit":m.unit,"statistic":m.statistic,"required":m.required,"original":original,"candidate":candidate,"gain":gain,"thresholds":m,"status":status,"point_estimate_only_requires_later_uncertainty_gate":true})
}
fn scaled(value: f64, unit: MetricUnit) -> f64 {
    match unit {
        MetricUnit::Milliseconds => value / 1_000_000.0,
        MetricUnit::Seconds => value / 1_000_000_000.0,
        _ => value,
    }
}
impl QualifiedTorch {
    pub fn measure(
        &self,
        original: &SourceState,
        candidate: &SourceState,
        bytes: &[u8],
        limits: Limits,
        cancel: &CancellationToken,
    ) -> Result<Measurement> {
        cancel.check()?;
        limits.validate()?;
        original.verify_integrity()?;
        candidate.verify_integrity()?;
        let p: Protocol = serde_json::from_value(
            CanonicalDocument::parse(bytes, JsonLimits::default())?
                .value()
                .clone(),
        )
        .map_err(|_| {
            error(
                ErrorCode::InvalidValue,
                "invalid frozen measurement protocol",
            )
        })?;
        p.validate()?;
        if p.environment != self.measurement_environment()? {
            return Err(error(
                ErrorCode::SourceMismatch,
                "runtime/physical CPU/driver/affinity/timer environment changed",
            ));
        }
        for source in [original, candidate] {
            if source.manifest().architecture.id != "nf.arithmetic.relu"
                || source.manifest().architecture.version != "1"
            {
                return Err(error(
                    ErrorCode::UnsupportedCapability,
                    "initial actual measurement profile is registered f64 arithmetic only",
                ));
            }
        }
        let expected = [
            preflight(original, &p.inputs)?,
            preflight(candidate, &p.inputs)?,
        ];
        let digests = [
            ContentDigest::of_bytes(&canonical(&expected[0])?),
            ContentDigest::of_bytes(&canonical(&expected[1])?),
        ];
        let schedule = p.schedule();
        let id = ContentDigest::of_bytes(&canonical(
            &json!({"protocol":p,"source":original.manifest_digest(),"candidate":candidate.manifest_digest(),"source_checkpoint":ContentDigest::of_bytes(original.checkpoint_bytes()),"candidate_checkpoint":ContentDigest::of_bytes(candidate.checkpoint_bytes()),"measurement_code":ContentDigest::of_bytes(include_bytes!("measurement.rs"))}),
        )?);
        let (job, assets) = self.stage_source(original)?;
        let (other, candidate_assets) = self.stage_source(candidate)?;
        record(
            &job.join("request.json"),
            &json!({"schema_version":1,"mode":"measure","assets":assets,"candidate_job":other,"candidate_assets":candidate_assets,"inputs":p.inputs,"seeds":p.seeds,"schedule":schedule,"warmup":p.warmup,"protocol":id}),
        )?;
        record(
            &job.join("intent.json"),
            &json!({"protocol":id,"status":"incomplete","outputs_eligible":false}),
        )?;
        let captured = self.launch(&job, &limits, cancel)?;
        record(
            &job.join("execution.json"),
            &json!({"status":captured.status,"cleanup_confirmed":true,"stdout":ContentDigest::of_bytes(&captured.stdout),"stderr":ContentDigest::of_bytes(&captured.stderr),"cpu_ms":captured.cpu_ms,"peak_rss_bytes":captured.rss_bytes,"wall_ms":captured.wall_ms,"outputs_eligible":false}),
        )?;
        if !captured.success {
            return Err(error(
                ErrorCode::TransactionAborted,
                "measurement worker failed or exceeded bounds; no measurement witness",
            ));
        }
        let child = CanonicalDocument::parse(
            &captured.stdout,
            JsonLimits {
                max_bytes: limits.output_bytes,
                max_depth: 32,
                max_nodes: 20000,
            },
        )?;
        let r = child.value();
        if r["schema_version"]!=1 || r["kind"]!="trusted_cpu_measurement" || r["protocol"]!=json!(id)
            || r["manifests"]!=json!([original.manifest_digest(),candidate.manifest_digest()]) || r["before"]!=r["after"]
            || r["clock"]!=p.environment.clock || r["cpu_affinity"]!=p.environment.affinity || r["torch_version"]!="2.9.1+cpu"
            || r["inference_mode"]!=true || r["gradient_enabled"]!=false || r["training"]!=json!([false,false])
            || r["threads"]!=1 || r["interop_threads"]!=1 || r["kernel_threads"]!=1 || r["clock_resolution_ns"].as_u64().is_none_or(|v|v==0)
            || !r["device_allocator_peak_bytes"].is_null() || !r["device_peak_bytes"].is_null()
            || r["memory_scope"]!="whole_paired_process_co_resident_models_interpreter_and_runtime;not_per_model"
            || r["forward_boundary"]!="sum_model_call_intervals_excludes_seed_input_tensor_output_copy_and_between_call_loop"
            || r["end_to_end_boundary"]!="in_process_numeric_request_includes_seed_input_tensor_forward_output_copy;excludes_model_load_and_process_startup" {
            return Err(error(ErrorCode::IntegrityMismatch,"measurement protocol/environment/mode/boundary receipt mismatch"));
        }
        verify_state(original, &r["before"][0], None)?;
        verify_state(candidate, &r["before"][1], None)?;
        let samples: Vec<Sample> = serde_json::from_value(r["samples"].clone())
            .map_err(|_| error(ErrorCode::InvalidValue, "malformed actual samples"))?;
        if samples.len() != schedule.len() {
            return Err(error(
                ErrorCode::IntegrityMismatch,
                "warmup/measured schedule length differs",
            ));
        }
        let mut forward = [Vec::new(), Vec::new()];
        let mut end_to_end = [Vec::new(), Vec::new()];
        let mut rss = [Vec::new(), Vec::new()];
        for (i, s) in samples.iter().enumerate() {
            if s.ordinal != i
                || s.arm != schedule[i]
                || s.warmup != (i / 2 < p.warmup as usize)
                || s.forward_ns == 0
                || s.end_to_end_ns < s.forward_ns
                || s.process_rss_bytes == 0
                || s.outputs != digests[s.arm as usize]
            {
                return Err(error(
                    ErrorCode::IntegrityMismatch,
                    "actual sample order/timer/outputs/RSS differs from frozen workload",
                ));
            }
            if !s.warmup {
                let a = s.arm as usize;
                forward[a].push(s.forward_ns);
                end_to_end[a].push(s.end_to_end_ns);
                rss[a].push(s.process_rss_bytes);
            }
        }
        if forward.iter().any(|v| v.len() != p.repetitions as usize) {
            return Err(error(
                ErrorCode::IntegrityMismatch,
                "insufficient measured repetitions",
            ));
        }
        let mut copies = 0;
        for (index, source) in [original, candidate].iter().enumerate() {
            let bytes = source.payload_bytes()?;
            copies += bytes * 4;
            for field in [
                "logical_group_bytes",
                "factory_tensor_bytes",
                "native_tensor_bytes_copy",
                "mutable_staging_bytes",
                "torch_clone_bytes",
            ] {
                if r["copies"][index][field].as_u64() != Some(bytes) {
                    return Err(error(
                        ErrorCode::IntegrityMismatch,
                        "measurement model copy receipt differs",
                    ));
                }
            }
            if r["copies"][index]["zero_copy"] != false {
                return Err(error(
                    ErrorCode::IntegrityMismatch,
                    "unqualified measurement zero-copy claim",
                ));
            }
        }
        let input_bytes =
            p.inputs.iter().map(|v| v.len() as u64 * 8).sum::<u64>() * schedule.len() as u64;
        if r["input_tensor_allocation_bytes"].as_u64() != Some(input_bytes)
            || r["process_high_water_rss_bytes"]
                .as_u64()
                .is_none_or(|v| v == 0 || v > captured.rss_bytes + 4_194_304)
        {
            return Err(error(
                ErrorCode::IntegrityMismatch,
                "measurement allocation/kernel RSS receipt differs",
            ));
        }
        // This fixed factory has exactly two dense matrices/no buffers. Opaque
        // provenance assets or storage-group labels cannot turn identical
        // executed weights/config into an efficiency-improvement candidate.
        let identical = original.manifest().parameter_digest
            == candidate.manifest().parameter_digest
            && original.manifest().architecture == candidate.manifest().architecture;
        let mut metrics = Vec::new();
        for metric in &p.metrics {
            let values = match metric.scope {
                MeasurementScope::ForwardLatency => Some(&forward),
                MeasurementScope::EndToEndLatency => Some(&end_to_end),
                MeasurementScope::ProcessRss => Some(&rss),
                _ => None,
            };
            let (a, b) = match values {
                Some(values) => (
                    Some(scaled(
                        statistic(&values[0], metric.statistic)?,
                        metric.unit,
                    )),
                    Some(scaled(
                        statistic(&values[1], metric.statistic)?,
                        metric.unit,
                    )),
                ),
                None => (None, None),
            };
            metrics.push(assess_metric(metric, a, b, identical));
        }
        original.verify_integrity()?;
        candidate.verify_integrity()?;
        self.verify_runtime()?;
        if self.measurement_environment()? != p.environment {
            return Err(error(
                ErrorCode::SourceMismatch,
                "measurement environment changed during run",
            ));
        }
        cancel.check()?;
        let eligible = metrics
            .iter()
            .all(|m| m["required"] != true || m["status"] == "passed");
        let report = json!({"schema_version":1,"protocol_id":id,"protocol":p,"source":original.manifest_digest(),"candidate":candidate.manifest_digest(),"identical_model":identical,"metrics":metrics,"measurement_conditions_satisfied":eligible,"scientific_acceptance":false,"outputs_eligible":false,"promotion_status":"not_evaluated","kernel_cpu_ms":captured.cpu_ms,"kernel_peak_rss_bytes":captured.rss_bytes,"process_wall_ms":captured.wall_ms,"explicit_model_transfer_staging_clone_factory_bytes":copies,"measured_input_allocation_bytes":input_bytes,"additional_cost_scope":"native_staging_hashing_reference_and_setup_outside_child_unknown_not_zero","child_report":r});
        let result = Measurement {
            id: ContentDigest::of_bytes(&canonical(&report)?),
            report,
        };
        record(&job.join("completion.json"), result.report())?;
        Ok(result)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn metric() -> Metric {
        Metric {
            name: "latency".into(),
            scope: MeasurementScope::ForwardLatency,
            unit: MetricUnit::Nanoseconds,
            statistic: Statistic::Median,
            required: true,
            ceiling: Some(50.0),
            max_regression: None,
            min_gain: None,
        }
    }
    #[test]
    fn measurement_protocol_controlled_statistics_exclude_no_samples_and_define_tails() {
        let values = [100, 10, 30, 20];
        assert_eq!(statistic(&values, Statistic::Mean).unwrap(), 40.0);
        assert_eq!(statistic(&values, Statistic::Median).unwrap(), 25.0);
        for kind in [Statistic::P95, Statistic::P99, Statistic::Maximum] {
            assert_eq!(statistic(&values, kind).unwrap(), 100.0);
        }
        assert!(statistic(&[], Statistic::Median).is_err());
        assert!(statistic(&[1], Statistic::Exact).is_err());
        assert_eq!(scaled(1_000_000.0, MetricUnit::Milliseconds), 1.0);
        assert_eq!(scaled(1_000_000_000.0, MetricUnit::Seconds), 1.0);
    }
    #[test]
    fn measurement_protocol_controlled_gate_noisy_noop_and_unavailable_are_not_passes() {
        let mut m = metric();
        assert_eq!(
            assess_metric(&m, Some(40.0), Some(60.0), false)["status"],
            "failed"
        );
        assert_eq!(
            assess_metric(&m, None, None, false)["status"],
            "unavailable"
        );
        m.min_gain = Some(1.0);
        assert_eq!(
            assess_metric(&m, Some(40.0), Some(10.0), true)["status"],
            "no_op_cannot_satisfy_positive_gain"
        );
        m.scope = MeasurementScope::ProcessRss;
        assert_eq!(
            assess_metric(&m, Some(40.0), Some(10.0), false)["status"],
            "inconclusive_co_resident_memory_attribution"
        );
    }
    #[test]
    fn measurement_protocol_statistics_do_not_overflow_integer_accumulation() {
        assert_eq!(
            statistic(&[u64::MAX, u64::MAX], Statistic::Mean).unwrap(),
            u64::MAX as f64
        );
        assert_eq!(
            statistic(&[u64::MAX, u64::MAX], Statistic::Median).unwrap(),
            u64::MAX as f64
        );
    }
}

//! Coordinator-owned broker for the fixed static computation. No receipt import.
//! Trusted edit/export/CAS/ledger code stays here; the isolated image receives
//! only four verified weight vectors, one input and an exact dispatch commitment.
use super::*;
use std::io::{Read, Write};
use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt};
use std::path::PathBuf;
use std::time::Duration;
#[path = "../../fixture_supervisor/mod.rs"]
#[allow(dead_code)]
mod supervisor;
const IMAGE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/nf-final-linear-worker"));
const PYTHON: &str = "/usr/bin/python3";
const HEAD: &[u8] = b"NFLIN001";
const WORKER_MEMORY: u64 = 32 << 20;
const GUARDIAN_MEMORY: u64 = 96 << 20;

/// The public fixture API requires a dedicated controller process. Temporarily
/// narrow its AS admission so the sum of controller+guardian+worker AS limits
/// cannot exceed the frozen host-memory allowance. Never raise an existing
/// limit. The old soft limit is restored only after the external child is gone
/// (or left unknown, in which case retain this lower limit until process exit).
pub(super) struct MemoryEnvelope {
    old: rustix::process::Rlimit,
    restore: bool,
}
impl MemoryEnvelope {
    pub fn enter(host: u64) -> Result<Self> {
        let old = rustix::process::getrlimit(rustix::process::Resource::As);
        let cap = host
            .checked_sub(GUARDIAN_MEMORY + WORKER_MEMORY)
            .ok_or_else(|| deny("aggregate memory envelope unavailable"))?;
        let soft = old
            .current
            .ok_or_else(|| deny("finite controller AS limit required"))?
            .min(cap);
        if soft < bounds::MEMORY {
            return Err(deny(
                "aggregate controller/guardian/worker memory too small",
            ));
        }
        rustix::process::setrlimit(
            rustix::process::Resource::As,
            rustix::process::Rlimit {
                current: Some(soft),
                maximum: old.maximum,
            },
        )
        .map_err(|e| deny(&format!("controller AS admission unavailable: {e}")))?;
        bounds::hard_memory_limit(host)?;
        Ok(Self {
            old,
            restore: false,
        })
    }
    pub fn cleanup_confirmed(&mut self) {
        self.restore = true;
    }
}
impl Drop for MemoryEnvelope {
    fn drop(&mut self) {
        if self.restore {
            let _ = rustix::process::setrlimit(rustix::process::Resource::As, self.old);
        }
    }
}

fn io(e: std::io::Error) -> CoreError {
    failure(ErrorCode::Io, &format!("supervised fixture: {e}"))
}
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
fn unhex(s: &str) -> Result<Vec<u8>> {
    if !s.len().is_multiple_of(2) || s.len() > 65536 {
        return Err(deny("bounded worker hex required"));
    }
    s.as_bytes()
        .chunks_exact(2)
        .map(|p| {
            std::str::from_utf8(p)
                .ok()
                .and_then(|v| u8::from_str_radix(v, 16).ok())
                .ok_or_else(|| deny("invalid worker hex"))
        })
        .collect()
}
struct Image {
    directory: PathBuf,
    path: PathBuf,
}
impl Image {
    fn new(nonce: &str) -> Result<Self> {
        let directory =
            std::env::temp_dir().join(format!("nf-linear-{}-{nonce}", std::process::id()));
        std::fs::DirBuilder::new()
            .mode(0o700)
            .create(&directory)
            .map_err(io)?;
        let image = Self {
            path: directory.join("worker"),
            directory,
        };
        let mut file = std::fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .mode(0o500)
            .open(&image.path)
            .map_err(io)?;
        file.write_all(IMAGE).map_err(io)?;
        file.sync_all().map_err(io)?;
        Ok(image)
    }
}
impl Drop for Image {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.directory);
    }
}

pub(super) fn image_storage_bytes() -> u64 {
    IMAGE.len() as u64
}
pub(super) fn image_digest() -> ContentDigest {
    ContentDigest::of_bytes(IMAGE)
}
pub(super) fn profile() -> ContentDigest {
    hash(&(
        ContentDigest::of_bytes(IMAGE),
        ContentDigest::of_bytes(include_bytes!("linear_worker.c")),
        ContentDigest::of_bytes(include_bytes!("supervised.rs")),
        ContentDigest::of_bytes(include_bytes!("../../fixture_supervisor/mod.rs")),
        ContentDigest::of_bytes(include_bytes!("../../fixture_supervisor/boundary.py")),
        ContentDigest::of_bytes(include_bytes!("../../fixture_supervisor/guardian.py")),
    ))
    .expect("bounded code identities")
}
fn names(cell: &ContentDigest) -> (String, String, String) {
    let suffix = cell.as_str().trim_start_matches("sha256:");
    (
        format!("worker_intent_{suffix}"),
        format!("worker_prepared_{suffix}"),
        format!("worker_receipt_{suffix}"),
    )
}
fn persist(ledger: &mut Ledger, lease: &Lease, kind: &str, body: &impl Serialize) -> Result<()> {
    let tx = ledger
        .connection
        .transaction_with_behavior(TransactionBehavior::Immediate)
        .map_err(sql)?;
    check_lease(&tx, &ledger.store, lease, &files::moment()?)?;
    record(&tx, lease, kind, body)?;
    ensure_live(lease)?;
    tx.commit().map_err(sql)
}
fn deadline_ns(ms: i64) -> Result<u64> {
    u64::try_from(ms)
        .ok()
        .and_then(|m| m.checked_mul(1_000_000))
        .ok_or_else(|| deny("absolute supervisor deadline overflow"))
}

pub(super) struct Measured {
    pub outputs: Vec<[f64; 2]>,
    pub arithmetic_outputs: Option<Vec<[f64; 5]>>,
    pub samples: Vec<Value>,
    pub latencies: [f64; 4],
    /// Controller-wide high-water RSS with all four arms resident.
    pub process_rss_bytes: u64,
    pub evidence: Value,
}
impl Ledger {
    pub(super) fn supervised_forwards(
        &mut self,
        lease: &Lease,
        reservation: &budgets::Reservation,
        cancel: &CancellationToken,
        cpu_start: u64,
        dispatch: &ContentDigest,
        models: [&SourceState; 4],
        requests: &[(&ContentDigest, &[f64])],
    ) -> Result<Vec<Measured>> {
        self.supervised_forwards_kind::<false>(
            lease,
            reservation,
            cancel,
            cpu_start,
            dispatch,
            models,
            requests,
        )
    }
    pub(super) fn supervised_arithmetic_forwards(
        &mut self,
        lease: &Lease,
        reservation: &budgets::Reservation,
        cancel: &CancellationToken,
        cpu_start: u64,
        dispatch: &ContentDigest,
        models: [&SourceState; 4],
        requests: &[(&ContentDigest, &[f64])],
    ) -> Result<Vec<Measured>> {
        self.supervised_forwards_kind::<true>(
            lease,
            reservation,
            cancel,
            cpu_start,
            dispatch,
            models,
            requests,
        )
    }
    fn supervised_forwards_kind<const ARITHMETIC: bool>(
        &mut self,
        lease: &Lease,
        reservation: &budgets::Reservation,
        cancel: &CancellationToken,
        cpu_start: u64,
        dispatch: &ContentDigest,
        models: [&SourceState; 4],
        requests: &[(&ContentDigest, &[f64])],
    ) -> Result<Vec<Measured>> {
        if requests.is_empty() || requests.len() > 32 {
            return Err(deny("bounded supervised panel required"));
        }
        let head: &[u8] = if ARITHMETIC { b"NFARI002" } else { HEAD };
        let output_size = if ARITHMETIC { 808 } else { 432 };
        let keys: Vec<_> = requests.iter().map(|(key, _)| *key).collect();
        let batch = hash(&(dispatch, &keys))?;
        let cell = &batch;
        self.allocation_fence(lease, reservation, cancel, cpu_start)?;
        let (_, intent) = read(&self.connection, &lease.contract, "intent")?
            .ok_or_else(|| deny("native intent absent"))?;
        let deadline = deadline_ns(
            intent["operation_deadline"]
                .as_i64()
                .ok_or_else(|| deny("deadline absent"))?,
        )?;
        let mut random = [0; 32];
        std::fs::File::open("/dev/urandom")
            .and_then(|mut f| f.read_exact(&mut random))
            .map_err(io)?;
        let nonce = hex(&random);
        let image_database_cap = if ARITHMETIC {
            self.connection
                .query_row("PRAGMA max_page_count", [], |r| r.get(0))
                .map_err(sql)?
        } else {
            bounds::image_scratch_cap(&self.connection)?
        };
        let image = Image::new(&nonce)?;
        let mut input_bytes = head.to_vec();
        input_bytes.extend((requests.len() as u32).to_le_bytes());
        for model in models {
            model.verify_integrity()?;
            let tensors: &[(&str, usize)] = if ARITHMETIC {
                &[("hidden.weight", 12), ("heads.weight", 30)]
            } else {
                &[("w", 4)]
            };
            for (name, length) in tensors {
                let weights = model
                    .tensor(name)
                    .and_then(|t| t.f64_values())
                    .ok_or_else(|| deny("f64 worker weights required"))?;
                if weights.len() != *length {
                    return Err(deny("worker weight shape differs"));
                }
                for w in weights {
                    input_bytes.extend(w.to_le_bytes());
                }
            }
        }
        for (key, input) in requests {
            if input.len() != 2 {
                return Err(deny("worker input shape differs"));
            }
            input_bytes.extend(unhex(key.as_str().trim_start_matches("sha256:"))?);
            for x in *input {
                input_bytes.extend(x.to_le_bytes());
            }
        }
        let policy = supervisor::Policy {
            profile: supervisor::PROFILE.into(),
            boot_id: lease.boot.clone(),
            deadline_boottime_ns: deadline,
            memory_bytes: WORKER_MEMORY,
            cpu_seconds: 1,
            output_bytes: (output_size * requests.len()) as u64,
            grace_ms: 40,
            reap_ms: 200,
            executable_sha256: ContentDigest::of_bytes(IMAGE)
                .as_str()
                .trim_start_matches("sha256:")
                .into(),
            argv: vec![image
                .path
                .to_str()
                .ok_or_else(|| deny("worker path encoding"))?
                .into()],
            input_hex: hex(&input_bytes),
            binding: supervisor::LaunchBinding {
                contract: lease.contract.to_string(),
                reservation: reservation.id().to_string(),
                attempt: cell.to_string(),
                lease: hash(&(&lease.owner, lease.epoch, lease.deadline))?.to_string(),
                fencing_epoch: u64::try_from(lease.epoch)
                    .map_err(|_| deny("invalid fencing epoch"))?,
                nonce,
                lease_deadline_boottime_ns: deadline_ns(lease.deadline)?,
            },
        };
        let (intent_name, prepared_name, receipt_name) = names(cell);
        // Persist even before preparation: a crash in the preparation gap cannot
        // refund an unknown guardian/worker allocation or silently retry it.
        persist(
            self,
            lease,
            &intent_name,
            &json!({"cell":cell,"dispatch":dispatch,"cells":keys,"policy":policy,"profile":profile()}),
        )?;
        fault("supervisor_before_prepare")?;
        let mut child = supervisor::Supervisor::prepare(Path::new(PYTHON), policy).map_err(io)?;
        let prepared = loop {
            self.allocation_fence(lease, reservation, cancel, cpu_start)?;
            if let Some(prepared) = child.poll_prepared().map_err(io)? {
                break prepared;
            }
            std::thread::sleep(Duration::from_millis(2));
        };
        // Prepared means the guardian copied, sealed and closed the source file.
        // Remove named scratch BEFORE restoring the larger database allowance.
        std::fs::remove_dir_all(&image.directory).map_err(io)?;
        drop(image);
        self.connection
            .pragma_update(None, "max_page_count", image_database_cap)
            .map_err(sql)?;
        persist(self, lease, &prepared_name, &prepared)?;
        fault("supervisor_after_prepare")?;
        self.allocation_fence(lease, reservation, cancel, cpu_start)?;
        child.authorize_after_persist(&prepared).map_err(io)?;
        fault("supervisor_after_authorize")?;
        let mut stop_sent = false;
        let receipt = loop {
            // Durable stop precedes normal cancellation. If persistence fails,
            // EOF on Drop and the original absolute guardian deadline still act.
            let fence = if stop_sent {
                Ok(())
            } else {
                self.allocation_fence(lease, reservation, cancel, cpu_start)
            };
            if let Err(error) = fence {
                let tx = self
                    .connection
                    .transaction_with_behavior(TransactionBehavior::Immediate)
                    .map_err(sql)?;
                record(
                    &tx,
                    lease,
                    &format!("worker_stop_{}", prepared.policy.binding.nonce),
                    &json!({"prepared":prepared,"reason":format!("{:?}",error.code)}),
                )?;
                tx.commit().map_err(sql)?;
                child.cancel_after_durable_stop().map_err(io)?;
                stop_sent = true;
            }
            if let Some(receipt) = child.poll().map_err(io)? {
                break receipt;
            }
            if files::moment()?.ms >= (deadline / 1_000_000) as i64 + 2500 {
                return Err(deny(
                    "supervisor receipt missing; cleanup unknown; retain allocation",
                ));
            }
            std::thread::sleep(Duration::from_millis(2));
        };
        let process_rss_bytes = receipt
            .controller_observation
            .as_ref()
            .map(|observation| observation.process_peak_rss_bytes)
            .filter(|value| *value > 0)
            .ok_or_else(|| deny("controller process RSS measurement unavailable"))?;
        let observation = serde_json::to_value(&receipt.controller_observation)
            .map_err(|_| deny("controller observation encoding"))?;
        persist(
            self,
            lease,
            &receipt_name,
            &json!({"cell":cell,"receipt":receipt,"controller":observation}),
        )?;
        fault("supervisor_after_receipt")?;
        self.allocation_fence(lease, reservation, cancel, cpu_start)?;
        if receipt.status != "completed"
            || !receipt.cleanup_confirmed
            || !receipt.stderr_hex.is_empty()
            || receipt.guardian_peak_rss_bytes > GUARDIAN_MEMORY
            || rss()?
                .saturating_add(receipt.guardian_peak_rss_bytes)
                .saturating_add(WORKER_MEMORY)
                > intent["resources"]["host_bytes"].as_u64().unwrap_or(0)
        {
            return Err(deny(
                "supervised computation failed resource/cleanup checks",
            ));
        }
        let output = unhex(&receipt.stdout_hex)?;
        if output.len() != output_size * requests.len() {
            return Err(deny("worker batch output length differs"));
        }
        let receipt_id = hash(&json!({"cell":cell,"receipt":receipt,"controller":observation}))?;
        requests.iter().zip(output.chunks_exact(output_size)).enumerate().map(|(index, ((key, input), output))| {
            let (outputs, arithmetic_outputs, samples, latencies) = if ARITHMETIC {
                let (values, samples, latencies) = decode_arithmetic(key, output)?;
                for (model, observed) in models.iter().zip(&values) {
                    let expected = model.infer_arithmetic_fixture(input, ExecutionLimits::default())?;
                    if observed.as_slice() != expected {
                        return Err(deny("arithmetic worker output differs from independent native recomputation"));
                    }
                }
                (Vec::new(), Some(values), samples, latencies)
            } else {
                let (outputs, samples, latencies) = decode(key, output)?;
                verify_math(&input_bytes, index, &outputs)?;
                (outputs, None, samples, latencies)
            };
            Ok(Measured { outputs, arithmetic_outputs, samples, latencies, process_rss_bytes,
                evidence: json!({"cell":key,"batch":batch,"index":index,"prepared_record":prepared_name,
                    "receipt_record":receipt_name,"receipt_digest":receipt_id}) })
        }).collect()
    }
}
/// Four interleaved arms, one warmup and three timed arithmetic forwards.
/// The timings describe only the fixed worker forward, not scoring or I/O.
fn decode_arithmetic(
    key: &ContentDigest,
    bytes: &[u8],
) -> Result<(Vec<[f64; 5]>, Vec<Value>, [f64; 4])> {
    if bytes.len() != 808
        || &bytes[..8] != b"NFARI002"
        || hex(&bytes[8..40]) != key.as_str().trim_start_matches("sha256:")
    {
        return Err(deny("arithmetic measurement ABI/context differs"));
    }
    let mut outputs = Vec::new();
    let mut samples = Vec::new();
    let mut latencies = [0.0_f64; 4];
    for (index, sample) in bytes[40..].chunks_exact(48).enumerate() {
        let ns = u64::from_le_bytes(sample[..8].try_into().unwrap());
        let values: [f64; 5] = std::array::from_fn(|i| {
            f64::from_le_bytes(sample[8 + i * 8..16 + i * 8].try_into().unwrap())
        });
        if ns >= 1_000_000_000 || values.iter().any(|value| !value.is_finite()) {
            return Err(deny("invalid arithmetic worker measurement"));
        }
        let arm = index % 4;
        let repeat = index / 4;
        if repeat == 0 {
            outputs.push(values);
        } else if outputs[arm] != values {
            return Err(deny("arithmetic output varied across repeats"));
        }
        let ms = ns as f64 / 1_000_000.0;
        if repeat > 0 {
            latencies[arm] = latencies[arm].max(ms);
        }
        samples.push(json!({"arm":arm,"repeat":repeat,"warmup":repeat==0,
            "wall_ms":ms,"output":hash(&values)?}));
    }
    if outputs[0] != outputs[1] {
        return Err(deny("supervised arithmetic no-op parity failed"));
    }
    Ok((outputs, samples, latencies))
}
fn decode(key: &ContentDigest, bytes: &[u8]) -> Result<(Vec<[f64; 2]>, Vec<Value>, [f64; 4])> {
    if bytes.len() != 432
        || &bytes[..8] != HEAD
        || hex(&bytes[8..40]) != key.as_str().trim_start_matches("sha256:")
    {
        return Err(deny("worker output exact study identity/ABI differs"));
    }
    let mut outputs = Vec::new();
    let mut samples = Vec::new();
    let mut latencies = [0.0_f64; 4];
    for (index, sample) in bytes[40..424].chunks_exact(24).enumerate() {
        let ns = u64::from_le_bytes(sample[..8].try_into().unwrap());
        let y = [
            f64::from_le_bytes(sample[8..16].try_into().unwrap()),
            f64::from_le_bytes(sample[16..24].try_into().unwrap()),
        ];
        if ns >= 1_000_000_000 || y.iter().any(|v| !v.is_finite()) {
            return Err(deny("invalid worker measurement"));
        }
        let arm = index % 4;
        let repeat = index / 4;
        if repeat == 0 {
            outputs.push(y);
        } else if outputs[arm] != y {
            return Err(deny("worker output varied across repeats"));
        }
        let ms = ns as f64 / 1_000_000.0;
        if repeat > 0 {
            latencies[arm] = latencies[arm].max(ms);
        }
        samples.push(
            json!({"arm":arm,"repeat":repeat,"warmup":repeat==0,"wall_ms":ms,"output":hash(&y)?}),
        );
    }
    if outputs[0] != outputs[1] {
        return Err(deny("supervised no-op parity failed"));
    }
    Ok((outputs, samples, latencies))
}

// Independent Rust recomputation outside the measured forward interval. A
// worker's repeated, self-consistent outputs are not a mathematical witness.
fn verify_math(input: &[u8], index: usize, outputs: &[[f64; 2]]) -> Result<()> {
    if input.len() < 140 || outputs.len() != 4 {
        return Err(deny("invalid linear math witness"));
    }
    let count = u32::from_le_bytes(input[8..12].try_into().unwrap()) as usize;
    if count == 0 || count > 32 || index >= count || input.len() != 140 + count * 48 {
        return Err(deny("invalid linear math input size"));
    }
    let value = |offset| f64::from_le_bytes(input[offset..offset + 8].try_into().unwrap());
    let x = [value(172 + index * 48), value(180 + index * 48)];
    for (arm, observed) in outputs.iter().enumerate() {
        let w = [
            value(12 + arm * 32),
            value(20 + arm * 32),
            value(28 + arm * 32),
            value(36 + arm * 32),
        ];
        let expected = [
            (0.0 + w[0] * x[0]) + w[1] * x[1],
            (0.0 + w[2] * x[0]) + w[3] * x[1],
        ];
        if w.iter().chain(x.iter()).any(|n| !n.is_finite())
            || expected.iter().any(|n| !n.is_finite())
            || *observed != expected
        {
            return Err(failure(
                ErrorCode::IntegrityMismatch,
                "worker output differs from trusted linear math",
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod math_tests {
    use super::*;
    #[test]
    fn arithmetic_measurements_require_matched_repeats() {
        let key = ContentDigest::of_bytes(b"arithmetic-cell");
        let mut output = b"NFARI002".to_vec();
        output.extend(unhex(key.as_str().trim_start_matches("sha256:")).unwrap());
        for repeat in 0..4_u64 {
            for arm in 0..4_u64 {
                output.extend((1000 + repeat * 100 + arm).to_le_bytes());
                for value in [if arm < 2 { 0.0 } else { arm as f64 }, 2.0, 3.0, 4.0, 5.0] {
                    output.extend(value.to_le_bytes());
                }
            }
        }
        let (values, samples, latencies) = decode_arithmetic(&key, &output).unwrap();
        assert_eq!(values.len(), 4);
        assert_eq!(samples.len(), 16);
        assert_eq!(latencies[0], 0.0013);
        assert_eq!(latencies[3], 0.001303);
        let mut wrong = output.clone();
        wrong[40 + 4 * 48 + 8] ^= 1;
        assert!(decode_arithmetic(&key, &wrong).is_err());
        assert!(decode_arithmetic(&key, &output[..output.len() - 1]).is_err());
    }
    #[test]
    fn arithmetic_worker_executes_and_times_real_forwards() {
        use std::process::{Command, Stdio};
        let key = ContentDigest::of_bytes(b"real-arithmetic-cell");
        let mut input = b"NFARI002".to_vec();
        input.extend(1u32.to_le_bytes());
        for arm in 0..4 {
            let mut weights = [0.0_f64; 42];
            weights[0] = 1.0;
            weights[3] = 1.0;
            for head in 0..5 {
                weights[12 + head * 6] = 1.0;
                weights[13 + head * 6] = if arm == 3 { 2.0 } else { 1.0 };
            }
            for weight in weights {
                input.extend(weight.to_le_bytes());
            }
        }
        input.extend(unhex(key.as_str().trim_start_matches("sha256:")).unwrap());
        input.extend(2.0_f64.to_le_bytes());
        input.extend(3.0_f64.to_le_bytes());
        let mut child = Command::new(concat!(env!("OUT_DIR"), "/nf-final-linear-worker"))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        child.stdin.take().unwrap().write_all(&input).unwrap();
        let result = child.wait_with_output().unwrap();
        assert!(result.status.success(), "{:?}", result.stderr);
        let (values, samples, latencies) = decode_arithmetic(&key, &result.stdout).unwrap();
        assert_eq!(values[0], [5.0; 5]);
        assert_eq!(values[1], values[0]);
        assert_eq!(values[3], [8.0; 5]);
        assert_eq!(samples.len(), 16);
        assert!(latencies
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0));
    }
    #[test]
    fn consistent_wrong_worker_math_is_not_a_witness() {
        let key = ContentDigest::of_bytes(b"cell");
        let mut input = HEAD.to_vec();
        input.extend(1u32.to_le_bytes());
        for _ in 0..4 {
            for w in [1.0_f64, 0.0, 0.0, 1.0] {
                input.extend(w.to_le_bytes());
            }
        }
        input.extend(unhex(key.as_str().trim_start_matches("sha256:")).unwrap());
        for x in [2.0_f64, 3.0] {
            input.extend(x.to_le_bytes());
        }
        let mut output = HEAD.to_vec();
        output.extend(unhex(key.as_str().trim_start_matches("sha256:")).unwrap());
        for _ in 0..16 {
            output.extend(1u64.to_le_bytes());
            for y in [9.0_f64, 9.0] {
                output.extend(y.to_le_bytes());
            }
        }
        output.extend(4096u64.to_le_bytes());
        let (wrong, _, _) = decode(&key, &output).unwrap();
        assert!(verify_math(&input, 0, &wrong).is_err());
        assert!(verify_math(&input, 0, &[[2.0, 3.0]; 4]).is_ok());
        assert!(verify_math(&input, usize::MAX, &wrong).is_err());
        assert!(verify_math(&input[..139], 0, &wrong).is_err());
    }
}

/// Reopened native observations; old arithmetic ABI has no timing samples.
pub(super) struct ArithmeticEvidence {
    pub outputs: Vec<[f64; 5]>,
    pub latencies: Option<[f64; 4]>,
    pub process_rss_bytes: Option<u64>,
}
/// Resolve actual model artifacts and reopen native launch records. Old ABI
/// records retain their outputs but cannot supply newly required timing evidence.
pub(super) fn qualify_arithmetic_inputs(
    connection: &Connection,
    contract: &ContentDigest,
    models: [&SourceState; 4],
) -> Result<BTreeMap<ContentDigest, ArithmeticEvidence>> {
    if !cleanup_known(connection, contract)? {
        return Err(deny("arithmetic cleanup unknown"));
    }
    let mut weights = Vec::new();
    for model in models {
        arithmetic_source::identity(model)?;
        for name in ["hidden.weight", "heads.weight"] {
            for v in model
                .tensor(name)
                .and_then(|t| t.f64_values())
                .ok_or_else(|| deny("arithmetic weights absent"))?
            {
                weights.extend(v.to_le_bytes());
            }
        }
    }
    let mut query = connection.prepare("SELECT kind FROM final_fixture_records WHERE contract=?1 AND kind LIKE 'worker_intent_%' ORDER BY kind").map_err(sql)?;
    let names = query
        .query_map([contract.as_str()], |r| r.get::<_, String>(0))
        .map_err(sql)?;
    let mut result = BTreeMap::new();
    for name in names {
        let (_, launch) = read(connection, contract, &name.map_err(sql)?)?
            .ok_or_else(|| deny("launch missing"))?;
        let input = unhex(
            launch["policy"]["input_hex"]
                .as_str()
                .ok_or_else(|| deny("input missing"))?,
        )?;
        let timed = input.len() >= 8 && &input[..8] == b"NFARI002";
        if input.len() < 1356
            || (!timed && &input[..8] != b"NFARI001")
            || input[12..1356] != weights
        {
            return Err(deny(
                "arithmetic launch not bound to independently resolved artifacts",
            ));
        }
        let count = u32::from_le_bytes(input[8..12].try_into().unwrap()) as usize;
        if count == 0 || count > 32 || input.len() != 1356 + count * 48 {
            return Err(deny("arithmetic launch size differs"));
        }
        let cell: ContentDigest =
            serde_json::from_value(launch["cell"].clone()).map_err(|_| deny("batch missing"))?;
        let (_, receipt) = read(connection, contract, &self::names(&cell).2)?
            .ok_or_else(|| deny("receipt missing"))?;
        if receipt["receipt"]["status"] != "completed"
            || receipt["receipt"]["exit_code"] != 0
            || receipt["receipt"]["stderr_hex"] != ""
        {
            return Err(deny("arithmetic launch did not complete"));
        }
        let output = unhex(
            receipt["receipt"]["stdout_hex"]
                .as_str()
                .ok_or_else(|| deny("output missing"))?,
        )?;
        let width = if timed { 808 } else { 200 };
        if output.len() != count * width {
            return Err(deny("partial arithmetic output"));
        }
        for (index, wire) in output.chunks_exact(width).enumerate() {
            let start = 1356 + index * 48;
            if wire[8..40] != input[start..start + 32] {
                return Err(deny("arithmetic cell identity differs"));
            }
            let key: ContentDigest = format!("sha256:{}", hex(&wire[8..40])).parse()?;
            let x = [
                f64::from_le_bytes(input[start + 32..start + 40].try_into().unwrap()),
                f64::from_le_bytes(input[start + 40..start + 48].try_into().unwrap()),
            ];
            let (values, latencies) = if timed {
                let (values, _, latencies) = decode_arithmetic(&key, wire)?;
                (values, Some(latencies))
            } else {
                if &wire[..8] != b"NFARI001" {
                    return Err(deny("arithmetic output version differs"));
                }
                (
                    wire[40..]
                        .chunks_exact(40)
                        .map(|arm| {
                            std::array::from_fn(|i| {
                                f64::from_le_bytes(arm[i * 8..i * 8 + 8].try_into().unwrap())
                            })
                        })
                        .collect::<Vec<[f64; 5]>>(),
                    None,
                )
            };
            for (model, y) in models.iter().zip(&values) {
                if y.iter().any(|v| !v.is_finite())
                    || y.as_slice()
                        != model.infer_arithmetic_fixture(&x, ExecutionLimits::default())?
                {
                    return Err(deny("reopened arithmetic native math differs"));
                }
            }
            let process_rss_bytes = receipt["controller"]["process_peak_rss_bytes"]
                .as_u64()
                .filter(|value| *value > 0);
            let evidence = ArithmeticEvidence {
                outputs: values,
                latencies,
                process_rss_bytes,
            };
            if evidence.outputs[0] != evidence.outputs[1] || result.insert(key, evidence).is_some()
            {
                return Err(deny("duplicate arithmetic cell/noop differs"));
            }
        }
    }
    if result.is_empty() || result.len() > 289 {
        return Err(deny("complete bounded arithmetic native launches required"));
    }
    Ok(result)
}
pub(super) fn qualify_inputs(
    connection: &Connection,
    contract: &ContentDigest,
    models: [&SourceState; 4],
) -> Result<Vec<ContentDigest>> {
    let mut weights = Vec::new();
    for model in models {
        model.verify_integrity()?;
        let values = model
            .tensor("w")
            .and_then(|t| t.f64_values())
            .ok_or_else(|| deny("qualification requires f64 weights"))?;
        if values.len() != 4 {
            return Err(deny("qualification weight shape differs"));
        }
        for value in values {
            weights.extend(value.to_le_bytes());
        }
    }
    let mut query = connection.prepare("SELECT kind FROM final_fixture_records WHERE contract=?1 AND kind LIKE 'worker_intent_%' ORDER BY kind").map_err(sql)?;
    let names = query
        .query_map([contract.as_str()], |r| r.get::<_, String>(0))
        .map_err(sql)?;
    let mut receipts = Vec::new();
    for name in names {
        let (_, launch) = read(connection, contract, &name.map_err(sql)?)?
            .ok_or_else(|| deny("qualification launch absent"))?;
        let input = unhex(
            launch["policy"]["input_hex"]
                .as_str()
                .ok_or_else(|| deny("qualification input absent"))?,
        )?;
        if input.len() < 140 || input[12..140] != weights {
            return Err(failure(
                ErrorCode::IntegrityMismatch,
                "launch weights differ from resolved model artifacts",
            ));
        }
        let cell: ContentDigest = serde_json::from_value(launch["cell"].clone())
            .map_err(|_| deny("qualification batch absent"))?;
        receipts.push(
            read(connection, contract, &self::names(&cell).2)?
                .ok_or_else(|| deny("qualification receipt absent"))?
                .0,
        );
        if receipts.len() > 1 {
            return Err(deny("fixed fixture requires one bounded batch"));
        }
    }
    if receipts.len() != 1 {
        return Err(deny("qualification requires executed batch"));
    }
    Ok(receipts)
}

/// Historical receipts are checked against coordinator-owned immutable launch
/// records, never accepted from an API argument. Missing cleanup retains holds.
pub(super) fn cleanup_known(connection: &Connection, contract: &ContentDigest) -> Result<bool> {
    let mut query = connection.prepare("SELECT kind,body FROM final_fixture_records WHERE contract=?1 AND kind LIKE 'worker_intent_%'").map_err(sql)?;
    let rows = query
        .query_map([contract.as_str()], |r| {
            Ok((r.get::<_, String>(0)?, r.get::<_, Vec<u8>>(1)?))
        })
        .map_err(sql)?;
    for row in rows {
        let (name, bytes) = row.map_err(sql)?;
        let doc = CanonicalDocument::parse(&bytes, Limits::default())?;
        let cell: ContentDigest = serde_json::from_value(doc.value()["cell"].clone())
            .map_err(|_| deny("invalid worker intent"))?;
        if name != names(&cell).0 {
            return Err(deny("worker intent identity differs"));
        }
        let Some((_, prepared)) = read(connection, contract, &names(&cell).1)? else {
            return Ok(false);
        };
        let Some((_, received)) = read(connection, contract, &names(&cell).2)? else {
            return Ok(false);
        };
        let receipt: supervisor::Receipt = serde_json::from_value(received["receipt"].clone())
            .map_err(|_| deny("invalid cleanup receipt schema"))?;
        if received["cell"] != json!(cell)
            || prepared != json!(receipt.identity)
            || doc.value()["policy"] != json!(receipt.identity.policy)
            || receipt.identity.policy.binding.contract != contract.as_str()
            || receipt.identity.policy.binding.attempt != cell.as_str()
            || receipt.boot_id != receipt.identity.policy.boot_id
            || receipt.profile != supervisor::PROFILE
            || receipt.deadline_boottime_ns != receipt.identity.policy.deadline_boottime_ns
        {
            return Err(deny("cleanup receipt launch identity differs"));
        }
        if !receipt.cleanup_confirmed || !receipt.events.contains_key("cleanup_confirmed_ns") {
            return Ok(false);
        }
    }
    Ok(true)
}
pub(super) fn cpu_ms(connection: &Connection, contract: &ContentDigest) -> Result<u64> {
    let mut query = connection.prepare("SELECT body FROM final_fixture_records WHERE contract=?1 AND kind LIKE 'worker_receipt_%'").map_err(sql)?;
    let rows = query
        .query_map([contract.as_str()], |r| r.get::<_, Vec<u8>>(0))
        .map_err(sql)?;
    let mut cpu = 0u64;
    for row in rows {
        let bytes = row.map_err(sql)?;
        let doc = CanonicalDocument::parse(&bytes, Limits::default())?;
        let r = &doc.value()["receipt"];
        cpu = cpu
            .checked_add(
                r["guardian_cpu_used_ns"]
                    .as_u64()
                    .ok_or_else(|| deny("guardian CPU unknown"))?,
            )
            .and_then(|v| v.checked_add(r["all_children_cpu_used_ns"].as_u64()?))
            .ok_or_else(|| deny("supervised CPU overflow/unknown"))?;
    }
    Ok(cpu.div_ceil(1_000_000))
}
pub(super) fn verify(
    connection: &Connection,
    contract: &ContentDigest,
    cell: &Value,
    intent: &Value,
    dispatch: &ContentDigest,
) -> Result<()> {
    let invalid = || {
        failure(
            ErrorCode::IntegrityMismatch,
            "supervised evidence identity/resource/output differs",
        )
    };
    let key = hash(&(
        dispatch,
        &cell["binding"],
        &cell["item"],
        &cell["arm_identities"],
        &cell["seed"],
    ))?;
    let batch: ContentDigest =
        serde_json::from_value(cell["supervisor"]["batch"].clone()).map_err(|_| invalid())?;
    let (intent_name, prepared_name, receipt_name) = names(&batch);
    let (_, launch) = read(connection, contract, &intent_name)?.ok_or_else(invalid)?;
    let (_, prepared) = read(connection, contract, &prepared_name)?.ok_or_else(invalid)?;
    let (receipt_id, received) = read(connection, contract, &receipt_name)?.ok_or_else(invalid)?;
    let receipt: supervisor::Receipt =
        serde_json::from_value(received["receipt"].clone()).map_err(|_| invalid())?;
    let policy = &receipt.identity.policy;
    let events = &receipt.events;
    let began = events
        .get("supervisor_started_ns")
        .copied()
        .ok_or_else(invalid)?;
    let started = events
        .get("worker_started_ns")
        .copied()
        .ok_or_else(invalid)?;
    let cleaned = events
        .get("cleanup_confirmed_ns")
        .copied()
        .ok_or_else(invalid)?;
    let created = events
        .get("receipt_created_ns")
        .copied()
        .ok_or_else(invalid)?;
    let external_cpu_ms = receipt
        .guardian_cpu_used_ns
        .checked_add(receipt.all_children_cpu_used_ns)
        .ok_or_else(invalid)?
        .div_ceil(1_000_000);
    let (_, execution) = read(connection, contract, "execution")?.ok_or_else(invalid)?;
    if !(began <= started && started <= cleaned && cleaned <= created)
        || receipt.profile != policy.profile
        || receipt.boot_id != policy.boot_id
        || receipt.deadline_boottime_ns != policy.deadline_boottime_ns
        || receipt.guardian_cpu_used_ns >= 2_000_000_000
        || receipt
            .cpu_used_ns
            .is_none_or(|n| n > receipt.all_children_cpu_used_ns)
        || execution["cpu_ms"]
            .as_u64()
            .is_none_or(|n| n < external_cpu_ms)
        || received["controller"]["process_cpu_delta_ns"]
            .as_u64()
            .is_none()
        || received["controller"]["process_peak_rss_bytes"]
            .as_u64()
            .is_none_or(|n| {
                n.saturating_add(GUARDIAN_MEMORY)
                    .saturating_add(WORKER_MEMORY)
                    > intent["resources"]["host_bytes"].as_u64().unwrap_or(0)
            })
        || receipt.peak_rss_bytes.is_none_or(|n| n > WORKER_MEMORY)
    {
        return Err(invalid());
    }
    let keys = launch["cells"].as_array().ok_or_else(invalid)?;
    let count = keys.len();
    let index = cell["supervisor"]["index"].as_u64().ok_or_else(invalid)? as usize;
    if count == 0
        || count > 32
        || index >= count
        || keys[index] != json!(key)
        || hash(&(dispatch, keys))? != batch
        || launch["dispatch"] != json!(dispatch)
        || received["cell"] != json!(batch)
        || launch["cell"] != json!(batch)
        || prepared != json!(receipt.identity)
        || launch["policy"] != json!(policy)
        || launch["profile"] != intent["supervised_profile"]
        || format!("sha256:{}", policy.executable_sha256)
            != intent["worker_image"].as_str().unwrap_or("")
        || policy.profile != supervisor::PROFILE
        || policy.binding.contract != contract.as_str()
        || policy.binding.attempt != batch.as_str()
        || policy.binding.reservation != intent["reservation"].as_str().unwrap_or("")
        || json!(policy.binding.fencing_epoch) != intent["epoch"]
        || json!(policy.boot_id) != intent["boot"]
        || policy.deadline_boottime_ns
            != deadline_ns(intent["operation_deadline"].as_i64().ok_or_else(invalid)?)?
        || receipt.status != "completed"
        || !receipt.cleanup_confirmed
        || receipt.exit_code != Some(0)
        || receipt.cleanup_overshoot_ns != Some(0)
        || receipt.cpu_used_ns.is_none_or(|n| n >= 1_000_000_000)
        || policy.memory_bytes != WORKER_MEMORY
        || policy.cpu_seconds != 1
        || policy.output_bytes != 432 * count as u64
        || !receipt.stderr_hex.is_empty()
        || receipt.guardian_peak_rss_bytes > GUARDIAN_MEMORY
        || receipt
            .events
            .get("cleanup_confirmed_ns")
            .is_none_or(|t| *t >= policy.deadline_boottime_ns)
        || receipt
            .events
            .get("worker_started_ns")
            .is_none_or(|t| *t >= policy.deadline_boottime_ns)
        || cell["supervisor"]["cell"] != json!(key)
        || cell["supervisor"]["receipt_digest"] != json!(receipt_id)
        || cell["supervisor"]["prepared_record"] != prepared_name
        || cell["supervisor"]["receipt_record"] != receipt_name
    {
        return Err(invalid());
    }
    let input = unhex(&policy.input_hex)?;
    if input.len() != 140 + count * 48
        || &input[..8] != HEAD
        || u32::from_le_bytes(input[8..12].try_into().unwrap()) as usize != count
        || hex(&input[140 + index * 48..172 + index * 48])
            != key.as_str().trim_start_matches("sha256:")
    {
        return Err(invalid());
    }
    let vector = &input[172 + index * 48..188 + index * 48];
    let x = [
        f64::from_le_bytes(vector[..8].try_into().unwrap()),
        f64::from_le_bytes(vector[8..].try_into().unwrap()),
    ];
    if json!(hash(&x)?) != cell["item"]["input"] {
        return Err(invalid());
    }
    let output = unhex(&receipt.stdout_hex)?;
    if output.len() != count * 432 {
        return Err(invalid());
    }
    let (outputs, samples, _) = decode(&key, &output[index * 432..(index + 1) * 432])?;
    verify_math(&input, index, &outputs)?;
    if json!(outputs) != cell["outputs"] || json!(samples) != cell["samples"] {
        return Err(invalid());
    }
    Ok(())
}

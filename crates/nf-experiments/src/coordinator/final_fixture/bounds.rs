//! Qualification of a finite native fixture in a kernel-memory-limited process.
//! RLIMIT_AS's irreversible hard ceiling bounds ALL mappings, not sampled RSS.
//! The caller must launch a dedicated native coordinator under that limit; we
//! never lower a shared controller's process limit or silently run unlimited.
use super::*;
use nf_checkpoint::export::ExportLimits;
use nf_checkpoint::serialization::SerializationLimits;
use nf_surgeon::transaction::MaterializationLimits;

pub(super) const MEMORY: u64 = 128 * 1024 * 1024;
/// Linux enforces address-space admission for heap, stacks and mmap before
/// allocation. This is stronger than a resident-memory ceiling. Trusted native
/// fixture code does not fork or raise limits. An unlimited/too-large hard limit
/// is unsupported, even when current RSS is tiny. No envelope assertion is used.
pub(super) fn hard_memory_limit(requested: u64) -> Result<u64> {
    let limit = rustix::process::getrlimit(rustix::process::Resource::As);
    match (limit.current, limit.maximum) {
        (Some(soft), Some(hard)) if soft >= MEMORY && soft <= hard && hard <= requested => {
            // Lowering a limit is not retroactive. Reject a caller that already
            // mapped beyond the ceiling before installing it.
            use std::io::Read;
            let mut stat = String::new();
            std::fs::File::open("/proc/self/statm")
                .and_then(|f| f.take(1024).read_to_string(&mut stat))
                .map_err(|e| failure(ErrorCode::Io, &e.to_string()))?;
            let pages = stat
                .split_whitespace()
                .next()
                .and_then(|n| n.parse::<u64>().ok())
                .ok_or_else(|| deny("native address-space accounting unavailable"))?;
            // SAFETY: sysconf has no pointer preconditions.
            let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
            if page <= 0
                || pages
                    .checked_mul(page as u64)
                    .is_none_or(|bytes| bytes > soft)
            {
                return Err(deny("native mappings exceed kernel memory ceiling"));
            }
            Ok(hard)
        }
        _ => Err(deny(
            "dedicated native process with finite hard RLIMIT_AS <= contract host memory required",
        )),
    }
}
pub(super) const STORAGE: u64 = 4 * 1024 * 1024;
const MIB: u64 = 1024 * 1024;
pub(super) const EXECUTION: ExecutionLimits = ExecutionLimits {
    max_verification_bytes: 4096,
    max_smoke_work: 4096,
    max_smoke_vector_elements: 2,
};
pub(super) const LOAD: StateLoadLimits = StateLoadLimits {
    max_checkpoint_bytes: 4096,
    max_tensor_bytes: 32,
    max_tensors: 2,
    max_asset_bytes: 2,
};
pub(super) fn export_limits() -> CoreExportLimits {
    CoreExportLimits {
        materialization: MaterializationLimits {
            load: LOAD,
            serialization: SerializationLimits {
                max_checkpoint_bytes: 4096,
                max_tensor_bytes: 32,
                max_header_bytes: 1024,
                max_staging_bytes: 4096,
            },
        },
        export: ExportLimits {
            max_checkpoint_bytes: 4096,
            max_asset_bytes: 2,
            max_total_bytes: 65536,
            max_tensors: 2,
        },
        reload: LOAD,
        smoke: EXECUTION,
    }
}
/// A checked upper bound, including six smoke forwards per probe: two
/// transactions plus materialized/reloaded checks for each of two exports.
pub(super) fn calls(c: &StudyContract, probes: usize) -> Result<u64> {
    c.portfolio
        .bindings
        .iter()
        .filter(|b| b.schedule.final_required)
        .try_fold(
            (probes as u64)
                .checked_mul(6)
                .ok_or_else(|| deny("forward count overflow"))?,
            |n, b| {
                n.checked_add((b.splits.holdout.items.len() as u64) * 16)
                    .ok_or_else(|| deny("forward count overflow"))
            },
        )
}
/// No payload resolution, native edit, export or holdout disclosure precedes
/// this qualification. Broader shapes, programs and smaller hard envelopes
/// require a separately limited worker, not a best-effort unlimited fallback.
pub(super) fn preflight(
    ledger: &Ledger,
    c: &StudyContract,
    edit: &CompiledPlan,
    noop: &CompiledPlan,
    smoke: &Validated<SmokeSuite>,
) -> Result<()> {
    hard_memory_limit(c.limits.host_memory_bytes)?;
    if super::supervised::image_storage_bytes() > MIB {
        return Err(deny("static image exceeds reserved scratch envelope"));
    }
    smoke.validate()?;
    if c.portfolio.bindings.len() > 8
        || c.portfolio.bindings.iter().any(|b| b.metrics.len() > 4)
        || c.portfolio
            .bindings
            .iter()
            .filter(|b| b.schedule.final_required)
            .map(|b| b.splits.holdout.items.len())
            .sum::<usize>()
            > 32
        || canonical(c)?.len() > 65536
        || smoke.adapter != "nf.linear_fixture.v1"
        || smoke
            .probes
            .iter()
            .any(|p| p.input.len() != 2 || p.expected.as_ref().is_some_and(|v| v.len() != 2))
        || c.limits.host_memory_bytes < MEMORY
        || c.limits.storage_bytes < STORAGE
        || c.limits.cpu_ms < 30_000
        || c.limits.wall_ms < 30_000
    {
        return Err(deny(
            "contract outside bounded native fixture resource profile",
        ));
    }
    for plan in [edit, noop] {
        let a = &plan.summary().allocations;
        if plan.program().operations.len() > 8
            || canonical(plan.program())?.len() > 65536
            || a.estimated_working_bytes > MIB
            || a.validated_input_bytes > 4096
            || a.kernel_work_units > 65536
            || a.validation_scalar_work > 65536
        {
            return Err(deny(
                "compiled program outside bounded native fixture profile",
            ));
        }
    }
    let (page, count): (u64, u64) = ledger
        .connection
        .query_row(
            "SELECT page_size,page_count FROM pragma_page_size(),pragma_page_count()",
            [],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .map_err(sql)?;
    if page.checked_mul(count).is_none_or(|n| n > MIB) {
        return Err(deny("native fixture requires a ledger of at most 1 MiB"));
    }
    Ok(())
}
/// SQLite checks this cap before extending the database. With an initial DB
/// <=1 MiB, growth <=1 MiB, rollback journal <=2 MiB + headers, and two bounded
/// exports (<256 KiB including manifests), 4 MiB covers retained and scratch
/// storage. While a named worker image exists, image_scratch_cap reserves its
/// space by narrowing database growth; the image is removed before restoring
/// this cap. CAS enforces each blob write and the executor bounds their count.
/// The caller restores the connection's previous cap even on execution error.
pub(super) fn cap_database(connection: &Connection, max_bytes: u64) -> Result<u64> {
    let old = connection
        .query_row("PRAGMA max_page_count", [], |r| r.get(0))
        .map_err(sql)?;
    let page: u64 = connection
        .query_row("PRAGMA page_size", [], |r| r.get(0))
        .map_err(sql)?;
    let count: u64 = connection
        .query_row("PRAGMA page_count", [], |r| r.get(0))
        .map_err(sql)?;
    let cap = (count + MIB / page).min(old).min(max_bytes / page);
    let actual: u64 = connection
        .query_row(&format!("PRAGMA max_page_count={cap}"), [], |r| r.get(0))
        .map_err(sql)?;
    if actual != cap {
        return Err(deny("SQLite write cap unavailable"));
    }
    Ok(old)
}
/// During image preparation reserve 1 MiB for its file and 256 KiB for CAS,
/// journal headers/directory overhead. DB growth+journal is at most twice its
/// cap: cap <=1.375 MiB leaves <=2.75 MiB, fitting the 4 MiB reservation.
/// If existing pages do not fit, fail BEFORE creating the scratch image.
pub(super) fn image_scratch_cap(connection: &Connection) -> Result<u64> {
    let old: u64 = connection
        .query_row("PRAGMA max_page_count", [], |r| r.get(0))
        .map_err(sql)?;
    let (page, count): (u64, u64) = connection
        .query_row(
            "SELECT page_size,page_count FROM pragma_page_size(),pragma_page_count()",
            [],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .map_err(sql)?;
    let cap = old.min((MIB + 384 * 1024) / page);
    if count > cap {
        return Err(deny("insufficient bounded scratch space for worker image"));
    }
    let actual: u64 = connection
        .query_row(&format!("PRAGMA max_page_count={cap}"), [], |r| r.get(0))
        .map_err(sql)?;
    if actual != cap {
        return Err(deny("image scratch database cap unavailable"));
    }
    Ok(old)
}

/// A private immutable copy of the exact verified 2x2 weights. Preparation
/// (integrity/shape/type checks and allocation) is OUTSIDE ForwardLatency.
#[allow(dead_code)]
pub(super) struct Forward([f64; 4]);
#[allow(dead_code)]
impl Forward {
    pub(super) fn prepare(model: &SourceState) -> Result<Self> {
        model.verify_integrity()?;
        let values = model
            .tensor("w")
            .and_then(|t| t.f64_values())
            .ok_or_else(|| deny("qualified f64 forward required"))?;
        let weights = values
            .try_into()
            .map_err(|_| deny("qualified 2x2 forward required"))?;
        Ok(Self(weights))
    }
    pub(super) fn measure(
        &self,
        x: &[f64],
        mut clock: impl FnMut() -> std::time::Duration,
    ) -> ([f64; 2], f64) {
        let start = clock();
        let output = self.run(x);
        let elapsed = clock().saturating_sub(start);
        (output, elapsed.as_secs_f64() * 1000.0)
    }
    pub(super) fn run(&self, x: &[f64]) -> [f64; 2] {
        let x = std::hint::black_box(x);
        let w = std::hint::black_box(&self.0);
        // Preserve the registered kernel's left-to-right accumulation, including
        // the initial +0. No validation, hashing, adapter or Vec allocation.
        std::hint::black_box([
            (0.0 + w[0] * x[0]) + w[1] * x[1],
            (0.0 + w[2] * x[0]) + w[3] * x[1],
        ])
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn acceptance_gate_forward_latency_has_exact_clock_boundaries() {
        let prepared = Forward([1.0, 2.0, 3.0, 4.0]);
        let mut ticks = [
            std::time::Duration::from_millis(100),
            std::time::Duration::from_millis(107),
        ]
        .into_iter();
        let (output, ms) = prepared.measure(&[5.0, 2.0], || ticks.next().unwrap());
        assert_eq!(output, [9.0, 23.0]);
        assert_eq!(ms, 7.0);
        assert!(ticks.next().is_none());
    }
}

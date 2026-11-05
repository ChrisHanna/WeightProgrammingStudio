//! A durable audit runs only in an already hard-limited dedicated controller.
//! Never lower a shared process limit here. This covers all controller mappings
//! (including baseline), not Docker daemon/CLI helpers or other host processes.
use super::*;

pub(super) fn qualify_controller(host_bytes: u64) -> Result<u64> {
    let maximum = host_bytes
        .checked_sub(crate::sandbox::MEMORY_BYTES)
        .unwrap_or(0)
        .min(crate::sandbox::CONTROLLER_MEMORY_ALLOWANCE_BYTES);
    let limit = rustix::process::getrlimit(rustix::process::Resource::As);
    let (Some(soft), Some(hard)) = (limit.current, limit.maximum) else {
        return Err(failure(ErrorCode::UnsupportedCapability,
            "durable audit requires dedicated controller with finite hard RLIMIT_AS; unlimited whole-controller contracts unsupported"));
    };
    if soft == 0 || soft > hard || hard > maximum {
        return Err(failure(
            ErrorCode::UnsupportedCapability,
            "controller hard RLIMIT_AS plus sandbox cgroup exceeds audit host envelope",
        ));
    }
    use std::io::Read;
    let mut stat = String::new();
    std::fs::File::open("/proc/self/statm")
        .and_then(|f| f.take(1024).read_to_string(&mut stat))
        .map_err(|e| failure(ErrorCode::Io, &e.to_string()))?;
    let pages = stat
        .split_whitespace()
        .next()
        .and_then(|n| n.parse::<u64>().ok())
        .ok_or_else(|| {
            failure(
                ErrorCode::UnsupportedCapability,
                "controller mappings unavailable",
            )
        })?;
    // SAFETY: sysconf has no pointer preconditions.
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if page <= 0 || pages.checked_mul(page as u64).is_none_or(|n| n > soft) {
        return Err(failure(
            ErrorCode::UnsupportedCapability,
            "controller mappings exceed preinstalled hard memory ceiling",
        ));
    }
    Ok(hard)
}

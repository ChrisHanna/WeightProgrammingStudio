//! Identity-only recovery of synchronous native work; never signal a reused PID.
use super::*;
use std::io::Read;
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Owner {
    pid: u32,
    boot: String,
    start_ticks: u64,
}
fn process(pid: u32) -> Result<Option<(u64, bool)>> {
    let file = match std::fs::File::open(format!("/proc/{pid}/stat")) {
        Ok(file) => file,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(failure(ErrorCode::Io, &e.to_string())),
    };
    let mut text = String::new();
    file.take(8193)
        .read_to_string(&mut text)
        .map_err(|e| failure(ErrorCode::Io, &e.to_string()))?;
    if text.len() > 8192 {
        return Err(deny("bounded native process identity required"));
    }
    let (_, tail) = text
        .rsplit_once(')')
        .ok_or_else(|| deny("native process identity malformed"))?;
    let mut fields = tail.split_whitespace();
    let zombie = fields.next() == Some("Z");
    // Field 22, after consuming field 3 (state).
    let start_ticks = fields
        .nth(18)
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| deny("native process start identity missing"))?;
    Ok(Some((start_ticks, zombie)))
}
impl Owner {
    pub(super) fn current() -> Result<Self> {
        let pid = std::process::id();
        let (start_ticks, _) = process(pid)?.ok_or_else(|| deny("native process absent"))?;
        Ok(Self {
            pid,
            boot: files::moment()?.boot,
            start_ticks,
        })
    }
    pub(super) fn gone(&self) -> Result<bool> {
        if self.pid <= 1 {
            return Err(deny("invalid native process identity"));
        }
        if files::moment()?.boot != self.boot {
            return Ok(true);
        }
        // A zombie leader alone is NOT proof that sibling threads have exited.
        // Require disappearance/reaping or an actually different start identity.
        Ok(process(self.pid)?.is_none_or(|(start, _)| start != self.start_ticks))
    }
}

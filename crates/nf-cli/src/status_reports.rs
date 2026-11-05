//! Internal diagnostic reports only; never export approval or execution authority.
use nf_core::error::{CoreError, ErrorCode, Result};
use nf_experiments::status_reports::CommittedSnapshot;
use std::path::Path;

pub(crate) fn dispatch(args: &[String]) -> Option<Result<String>> {
    if !matches!(args, [group, action, ..] if group == "study" && (action == "status" || action == "report"))
    {
        return None;
    }
    Some(render(args))
}

fn render(args: &[String]) -> Result<String> {
    let invalid = || {
        CoreError::new(ErrorCode::InvalidValue, "", "usage: nf study status|report DATABASE_FILE CONTRACT_DIGEST [--format json|markdown|both]; internal diagnostics only")
    };
    let (database, contract, format) = match args {
        [_, _, database, contract] => (database, contract, "json"),
        [_, _, database, contract, flag, format] if flag == "--format" => {
            (database, contract, format.as_str())
        }
        _ => return Err(invalid()),
    };
    // Reject malformed requests before any store observation.
    if !matches!(format, "json" | "markdown" | "both") {
        return Err(invalid());
    }
    let id = contract.parse()?;
    let snapshot = CommittedSnapshot::capture(Path::new(database), &id)?;
    match format {
        "json" => Ok(snapshot.render_json()),
        "markdown" => snapshot.render_markdown(),
        "both" => Ok(format!(
            "{}\n",
            serde_json::json!({
                "json": snapshot.render_json(),
                "markdown": snapshot.render_markdown()?,
            })
        )),
        _ => unreachable!(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(tail: &[&str]) -> Vec<String> {
        ["study", "report"]
            .into_iter()
            .chain(tail.iter().copied())
            .map(str::to_owned)
            .collect()
    }

    #[test]
    fn status_reports_validate_before_capture_and_do_not_create_stores() {
        let database = format!(
            "/tmp/nf-status-missing-{}/events.sqlite3",
            std::process::id()
        );
        let digest = format!("sha256:{}", "0".repeat(64));
        let result = dispatch(&args(&[&database, &digest, "--format", "public"])).unwrap();
        assert_eq!(result.unwrap_err().code, ErrorCode::InvalidValue);
        assert!(!Path::new(&database).exists());
        assert!(dispatch(&args(&[&database, &digest])).unwrap().is_err());
        assert!(!Path::new(&database).exists());
        assert!(dispatch(&args(&[&database])).unwrap().is_err());
        assert!(dispatch(&["study".into(), "run".into()]).is_none());
    }
}

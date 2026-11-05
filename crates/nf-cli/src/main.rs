// Native JSON, digest and protocol commands share the Rust core.
// Manual planning/control is exposed; research execution remains unavailable.
mod commands;
mod evals;
mod status_reports;
mod studies;
mod studio;
use nf_core::error::{CoreError, ErrorCode};
use std::process::ExitCode;

fn report(error: CoreError) -> ExitCode {
    let exit = match error.code {
        ErrorCode::PermissionDenied => 3,
        ErrorCode::Io => 4,
        ErrorCode::IntegrityMismatch
        | ErrorCode::SourceMismatch
        | ErrorCode::ScopeViolation
        | ErrorCode::AliasScopeEscape
        | ErrorCode::SmokeFailed => 5,
        ErrorCode::LimitExceeded
        | ErrorCode::QuotaExceeded
        | ErrorCode::ParameterBudgetExceeded => 6,
        _ => 2,
    };
    eprintln!(
        "{}",
        serde_json::json!({"schema_version":1,"ok":false,"error":error})
    );
    ExitCode::from(exit)
}
fn main() -> ExitCode {
    let args = std::env::args_os()
        .skip(1)
        .map(|arg| {
            arg.into_string().map_err(|_| {
                CoreError::new(
                    ErrorCode::InvalidValue,
                    "",
                    "UTF-8 command arguments required",
                )
            })
        })
        .collect::<Result<Vec<_>, _>>();
    let args = match args {
        Ok(args) => args,
        Err(error) => return report(error),
    };
    if let Some(result) = studies::dispatch(&args)
        .or_else(|| status_reports::dispatch(&args))
        .or_else(|| studio::dispatch(&args))
        .or_else(|| evals::dispatch(&args))
    {
        return match result {
            Ok(text) => {
                print!("{text}");
                ExitCode::SUCCESS
            }
            Err(error) => report(error),
        };
    }
    match args.as_slice() {
        [flag] if flag == "--version" || flag == "-V" => {
            println!("nf {}", nf_core::VERSION);
            ExitCode::SUCCESS
        }
        [] => {
            help();
            ExitCode::SUCCESS
        }
        [flag] if flag == "--help" || flag == "-h" => {
            help();
            ExitCode::SUCCESS
        }
        _ => match commands::dispatch(&args) {
            Ok(value) => {
                println!(
                    "{}",
                    serde_json::json!({"schema_version":1,"ok":true,"result":value})
                );
                ExitCode::SUCCESS
            }
            Err(error) => report(error),
        },
    }
}
fn help() {
    println!("Neural Programming Foundry {}\nUsage: nf --version | --help\n  nf core capabilities\n  nf json canonicalize JSON\n  nf blob digest FILE\n  nf protocol negotiate REQUEST.json\n  nf model inspect CHECKPOINT.safetensors\n  nf ir validate PROGRAM.json\n  nf search-space validate|show SPACE.json\n  nf ir compile|explain|apply REQUEST.json\n  nf patch replay REQUEST.json\n  nf checkpoint restore REQUEST.json\n  nf eval import|run|compare|verify ...\n  nf studio solve-shape|lower-structural|compose-patches|scientific-graph|library REQUEST.json\n  nf study plan|validate|freeze|run|stop|resume REQUEST.json\n  nf study status|report DATABASE_FILE CONTRACT_DIGEST [--format json|markdown|both]\nStudy reports are internal diagnostics, not sanitized exports or acceptance.\nManual freeze persists inputs/contract, not verified approval. Run unavailable; resume is recovery-only.\nCore requests use schema_version=1. Read commands do not execute studies.\nApply exports unaccepted weights; restore never changes an active reference.\nJSON errors: exit 2 validation/unsupported, 3 permission, 4 I/O, 5 integrity/scope/smoke, 6 budget.",nf_core::VERSION);
}

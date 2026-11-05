//! Controller-owned CPU fixture only. Not a launcher for Python or arbitrary code.
use std::io::{Read, Write};
use std::time::Duration;
fn millis() -> i64 {
    let t = rustix::time::clock_gettime(rustix::time::ClockId::Boottime);
    t.tv_sec * 1000 + t.tv_nsec / 1_000_000
}
fn ignore_term() {
    // SAFETY: this dedicated fixture process installs the libc constant handler,
    // never a Rust/Python callback. No live coordinator/model lives here.
    unsafe {
        libc::signal(libc::SIGTERM, libc::SIG_IGN);
    }
}
fn controller_crash(contract_path: &str, root: &str) -> ! {
    use nf_core::schemas::Validated;
    use nf_core::study::StudyContract;
    use nf_experiments::coordinator::budgets::{Request, Resources, WorkKind};
    use nf_experiments::coordinator::supervision::{FixtureMode, NativeJob, TrustedNativeWorker};
    use nf_experiments::coordinator::{Ledger, LedgerLimits};
    let mut bytes = Vec::new();
    std::fs::File::open(contract_path)
        .unwrap()
        .take(1_048_577)
        .read_to_end(&mut bytes)
        .unwrap();
    let contract = Validated::<StudyContract>::parse(&bytes).unwrap();
    let path = std::env::current_exe().unwrap();
    let digest = nf_core::canonical::ContentDigest::of_bytes(&std::fs::read(&path).unwrap());
    let worker = TrustedNativeWorker::qualify(&path, digest).unwrap();
    let mut ledger = Ledger::open(std::path::Path::new(root), LedgerLimits::default()).unwrap();
    let id = ledger.freeze(&contract).unwrap();
    let lease = ledger.acquire(&id, "crash-fixture", 1000).unwrap();
    let reservation = ledger
        .reserve(
            &lease,
            Request {
                schema_version: 1,
                operation: nf_core::canonical::ContentDigest::of_bytes(b"crash-worker"),
                attempt: 0,
                parent: None,
                kind: WorkKind::Metadata,
                resources: Resources {
                    cpu_ms: 8000,
                    host_bytes: 134217728,
                    storage_bytes: 1048576,
                    workers: 1,
                    ..Resources::default()
                },
                wall_ms: 5000,
                external: None,
            },
        )
        .unwrap();
    let running = ledger
        .launch_native(
            &lease,
            &reservation,
            &worker,
            NativeJob {
                mode: FixtureMode::Sleep,
                maximum_output_bytes: 256,
                termination_grace_ms: 20,
            },
        )
        .unwrap();
    println!(
        "{}",
        serde_json::json!({"contract":id,"worker":running.id(),"reservation":reservation.id()})
    );
    std::io::stdout().flush().unwrap();
    std::process::exit(86)
}
fn main() {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() == 3 && args[0] == "--controller-crash" {
        controller_crash(&args[1], &args[2]);
    }
    if args.as_slice() == ["--probe"] {
        let as_limit = rustix::process::getrlimit(rustix::process::Resource::As);
        let cpu = rustix::process::getrlimit(rustix::process::Resource::Cpu);
        let fsize = rustix::process::getrlimit(rustix::process::Resource::Fsize);
        println!(
            "{}",
            serde_json::json!({"protocol":"nf.trusted-fixture-worker.v1","address_space":as_limit.current,"cpu_seconds":cpu.current,"file_bytes":fsize.current,"pid":std::process::id(),"pgrp":rustix::process::getpgrp().as_raw_nonzero().get()})
        );
        return;
    }
    let [flag, deadline] = args.as_slice() else {
        std::process::exit(64)
    };
    if flag != "--worker" && flag != "--child" {
        std::process::exit(64);
    }
    let deadline: i64 = deadline.parse().unwrap();
    let leader = rustix::process::getpgrp().as_raw_nonzero().get() as u32 == std::process::id();
    if flag == "--worker" && !leader {
        std::process::exit(67);
    }
    // This trusted watchdog survives a coordinator crash. It kills only this
    // dedicated process group, whose leader was created by the supervisor.
    std::thread::spawn(move || loop {
        if millis() >= deadline {
            if leader {
                let group = rustix::process::getpgrp();
                let _ = rustix::process::kill_process_group(group, rustix::process::Signal::KILL);
            }
            std::process::exit(124);
        }
        std::thread::sleep(Duration::from_millis(2));
    });
    if flag == "--child" {
        ignore_term();
        loop {
            std::thread::sleep(Duration::from_millis(5));
        }
    }
    let mut input = Vec::new();
    std::io::stdin().take(4097).read_to_end(&mut input).unwrap();
    if input.len() > 4096 || input.is_empty() {
        std::process::exit(65);
    }
    let value: serde_json::Value = serde_json::from_slice(&input).unwrap();
    let mode = value["mode"].as_str().unwrap();
    match mode {
        "complete" => println!(
            "{}",
            serde_json::json!({"result":4,"kind":"trusted_cpu_fixture"})
        ),
        "sleep" => loop {
            std::thread::sleep(Duration::from_millis(5));
        },
        "ignore_term" => {
            ignore_term();
            loop {
                std::thread::sleep(Duration::from_millis(5));
            }
        }
        "spawn_child" => {
            ignore_term();
            let mut child = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--child", &deadline.to_string()])
                .spawn()
                .unwrap();
            let _ = child.wait();
        }
        "busy" => loop {
            std::hint::black_box(123u64.wrapping_mul(456));
        },
        "memory" => {
            let limit = rustix::process::getrlimit(rustix::process::Resource::As)
                .current
                .unwrap();
            let mut bytes = Vec::<u8>::new();
            if bytes.try_reserve_exact(limit as usize + 1).is_err() {
                std::process::exit(72);
            }
            bytes.resize(limit as usize + 1, 1);
            std::hint::black_box(bytes);
        }
        "output" => {
            let mut out = std::io::stdout().lock();
            loop {
                if out.write_all(&[b'x'; 1024]).is_err() {
                    std::process::exit(73);
                }
            }
        }
        "partial" => {
            println!("{{\"partial_checkpoint\":true}}");
            std::io::stdout().flush().unwrap();
            loop {
                std::thread::sleep(Duration::from_millis(5));
            }
        }
        _ => std::process::exit(66),
    }
}

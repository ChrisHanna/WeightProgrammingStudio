//! Coordinator-side cancellation bridge. Never transfers publication authority.
use super::*;
use std::sync::atomic::{AtomicBool, Ordering};

pub(super) struct Guard {
    pub cancel: CancellationToken,
    done: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<Result<()>>>,
}
fn poll(
    c: &Connection,
    store: &str,
    lease: &Lease,
    deadline: i64,
    external: &CancellationToken,
) -> Result<()> {
    external.check()?;
    let now = files::moment()?;
    if now.ms >= deadline {
        return Err(deny("promotion allocation deadline reached"));
    }
    let tx = c.unchecked_transaction().map_err(sql)?;
    check_lease(&tx, store, lease, &now)?;
    if supervision::stopped(&tx, &lease.contract)?
        || budgets::snapshot_in(&tx, &lease.contract)?.accounting_halted
    {
        return Err(deny("promotion durably stopped or accounting halted"));
    }
    tx.commit().map_err(sql)
}
struct CancelOnExit(CancellationToken);
impl Drop for CancelOnExit {
    fn drop(&mut self) {
        self.0.request();
    }
}
impl Guard {
    pub fn start(
        path: std::path::PathBuf,
        store: String,
        lease: Lease,
        deadline: i64,
        external: CancellationToken,
        cancel: CancellationToken,
    ) -> Result<Self> {
        let files = files::Files::open_read_only_file(&path, 1_073_741_824)?;
        let c = Connection::open_with_flags(
            files.sqlite_path(),
            OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NOFOLLOW,
        )
        .map_err(sql)?;
        // A locked/unreadable fence is NOT permission to keep running.
        c.busy_timeout(Duration::from_millis(1)).map_err(sql)?;
        poll(&c, &store, &lease, deadline, &external)?;
        let cancel_on_exit = CancelOnExit(cancel.clone());
        let done = Arc::new(AtomicBool::new(false));
        let finished = done.clone();
        let thread = std::thread::spawn(move || {
            let _cancel_on_exit = cancel_on_exit; // Also cancels on unwind.
            (|| loop {
                files.verify(1_073_741_824)?;
                poll(&c, &store, &lease, deadline, &external)?;
                if finished.load(Ordering::Acquire) {
                    return Ok(());
                }
                std::thread::sleep(Duration::from_millis(2));
            })()
        });
        Ok(Self {
            cancel,
            done,
            thread: Some(thread),
        })
    }
    pub fn finish(mut self) -> Result<()> {
        self.done.store(true, Ordering::Release);
        self.thread
            .take()
            .unwrap()
            .join()
            .map_err(|_| deny("promotion fence monitor panicked"))?
    }
}
impl Drop for Guard {
    fn drop(&mut self) {
        self.cancel.request();
        self.done.store(true, Ordering::Release);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

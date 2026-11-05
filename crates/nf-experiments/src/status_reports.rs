//! Internal T040 projections. No report is a scientific acceptance receipt.
//!
//! Coordinator capture uses the read-only
//! `pub(crate) Ledger::read_committed_status(path: &Path, id: &ContentDigest)`
//! returning `Result<CommittedRead>`, using ONE deferred read transaction on ONE
//! connection. `path` is the explicit existing database FILE, not a directory.
//! Open read-only, reject unsafe paths/side-file aliases and unsupported schema;
//! set bounded SQLite limits, query_only and trusted_schema=OFF. Never migrate.
//! Do not call Ledger::open (it initializes/migrates), or combine state/events/
//! budget_snapshot calls (each opens a separate transaction). Reuse load_contract,
//! state, decode_event, and budgets load/summarize/faults inside that transaction.
//! Retain all family request/outcome versions/faults, stop/worker observations,
//! final panels/openings and exposure events/gaps, with row/byte bounds and hashes.
//! Missing receipt links remain unknown. No lease acquisition, migration, worker,
//! evidence resolution or public export is authorized by capture.

use crate::coordinator::{budgets, Event, RunState};
use nf_core::canonical::{CanonicalDocument, ContentDigest, Limits};
use nf_core::error::{CoreError, ErrorCode, Result};
use nf_core::schemas::Validated;
use nf_core::study::{ExecutionLimits, StudyContract};
use serde_json::{json, Value};
use std::path::Path;

const MAX_REPORT_BYTES: usize = 32 * 1024 * 1024;

fn failure(code: ErrorCode, message: &str) -> CoreError {
    CoreError::new(code, "", message)
}
fn unavailable(reason: &str) -> Value {
    json!({"status":"unavailable", "value":null, "reason":reason})
}

/// Project a typed, hash-bound stop request, never infer that workers stopped.
fn stop_request(records: &Value, id: &ContentDigest) -> Result<Value> {
    for retained in records["stop_requests"].as_array().into_iter().flatten() {
        let Ok(record) = serde_json::from_value::<crate::coordinator::supervision::StopRecord>(
            retained["record"].clone(),
        ) else {
            continue; // Unresolved records remain visible in the retained section.
        };
        if record.schema_version != 1 || record.contract != *id {
            continue;
        }
        let bytes = serde_json::to_vec(&record)
            .map_err(|_| failure(ErrorCode::InvalidValue, "stop serialization failed"))?;
        let document = CanonicalDocument::parse(&bytes, Limits::default())?;
        if retained["digest"] != json!(document.digest())
            || retained["key"] != json!(record.operation)
        {
            return Err(failure(
                ErrorCode::IntegrityMismatch,
                "stop request binding differs",
            ));
        }
        return Ok(json!({"status":"recorded", "reason":record.reason,
            "record_digest":document.digest(), "record":record,
            "scope":"current contract stop request; not proof of worker termination or scientific outcome"}));
    }
    Ok(unavailable(
        "no resolved current-contract stop request; no success or stop code inferred",
    ))
}

/// A contract-bound frozen panel, NOT proof of execution, reload or acceptance.
/// No Deserialize implementation: construction belongs to the committed reader,
/// after checking the typed canonical digest and all frozen-contract bindings.
#[derive(serde::Serialize)]
pub(crate) struct FrozenPanelProjection {
    pub digest: ContentDigest,
    pub panel: crate::coordinator::splits::Panel,
}

/// The coordinator owner must construct this only within its validated read
/// transaction. It is intentionally not deserializable or exposed to Python.
#[allow(dead_code)]
pub(crate) struct CommittedRead {
    pub contract: Validated<StudyContract>,
    pub state: RunState,
    pub events: Vec<Event>,
    pub budgets: budgets::Snapshot,
    pub family_origin: ContentDigest,
    pub family_limits: ExecutionLimits,
    pub database_schema_version: u32,
    /// Hash/index/transition-validated journals for every family revision.
    pub family_journal: Value,
    /// Verified frozen identities only; missing execution links stay unresolved.
    pub frozen_panels: Vec<FrozenPanelProjection>,
    /// Report-only native witnesses, constructed inside the same transaction.
    pub final_evidence: Vec<crate::coordinator::final_fixture::FinalEvidenceProjection>,
    /// Retained records, not summaries: requests, outcomes (every version), faults.
    pub budget_records: Value,
    /// Retained stop requests, intents, updates, including failures and cleanup.
    pub supervision: Value,
    /// Retained final panels, holdout openings, exposures and exposure gaps.
    pub split_evidence: Value,
}

/// Detached bytes with no public constructor from untrusted authority DTOs.
#[derive(Debug, Clone)]
pub struct CommittedSnapshot {
    payload: String,
}
impl CommittedSnapshot {
    /// Read one committed coordinator transaction. Never initialize/migrate or
    /// fall back to Python, multiple transactions, receipt resolution or probes.
    pub fn capture(path: &Path, id: &ContentDigest) -> Result<Self> {
        Self::from_committed_read(crate::coordinator::Ledger::read_committed_status(path, id)?)
    }

    #[allow(dead_code)]
    pub(crate) fn from_committed_read(read: CommittedRead) -> Result<Self> {
        let contract = &*read.contract;
        let id = contract.digest()?;
        if read.state.contract != id
            || read.budgets.family != contract.study_family
            || read.events.len() as u64 != read.state.sequence
            || read.events.last().map(|event| &event.digest) != Some(&read.state.head)
            || !(5..=8).contains(&read.database_schema_version)
        {
            return Err(failure(
                ErrorCode::IntegrityMismatch,
                "committed read identity/head mismatch",
            ));
        }
        let mut previous = ContentDigest::of_bytes(b"nf.coordinator.event.genesis.v1");
        for (index, event) in read.events.iter().enumerate() {
            let raw = serde_json::to_vec(&event.body)
                .map_err(|_| failure(ErrorCode::InvalidValue, "event serialization failed"))?;
            let canonical = CanonicalDocument::parse(&raw, Limits::default())?;
            if event.body.contract != id
                || event.body.sequence != index as u64 + 1
                || event.body.previous != previous
                || canonical.digest() != event.digest
            {
                return Err(failure(
                    ErrorCode::IntegrityMismatch,
                    "committed event identity/hash mismatch",
                ));
            }
            previous = event.digest.clone();
        }
        let stop = stop_request(&read.supervision, &id)?;
        let final_evidence = serde_json::to_value(&read.final_evidence).map_err(|_| {
            failure(
                ErrorCode::InvalidValue,
                "final evidence serialization failed",
            )
        })?;
        let links: Vec<_> = final_evidence
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|e| e["receipt_links"].as_array().into_iter().flatten().cloned())
            .collect();
        let current_final = final_evidence
            .as_array()
            .unwrap()
            .iter()
            .find(|e| e["contract"] == json!(id) && e["execution_status"] == "complete");
        let baselines: Vec<_> = contract.baselines.iter().map(|arm| json!({
            "configuration":arm, "status":if arm.enabled {"unknown"} else {"not_run"},
            "reason":if arm.enabled {"enabled configuration is not execution evidence"} else {"disabled in frozen contract"},
            "costs":null, "scores":null, "candidate_verdict":null,
            "execution_disclosure":{"source_checkpoint":null,"examples":null,
                "teacher_access":null,"selection_attempts":null,"trainable_parameters":null,
                "edited_parameters":null,"training_steps":null,"wall_ms":null,
                "peak_host_bytes":null,"peak_device_bytes":null,
                "status":if arm.enabled {"unknown"} else {"not_run"},
                "reason":"frozen configuration does not establish executed method conditions or per-arm costs"}
        })).collect();
        let gym_gates: Vec<_> = contract
            .portfolio
            .bindings
            .iter()
            .map(|binding| {
                let results: Vec<_> = current_final.into_iter().flat_map(|e|
                    e["metric_results"].as_array().into_iter().flatten())
                    .filter(|r| r["binding"] == json!(binding.id)).cloned().collect();
                json!({
                    "binding":binding, "status":if results.is_empty() {"unknown"} else {"recorded_fixture_results"},
                    "result":if results.is_empty() {Value::Null} else {json!(results)},
                    "reason":"only linked native fixture aggregates are results; never qualified acceptance"
                })
            })
            .collect();
        let value = json!({
            "schema_version":1,
            "snapshot":{
                "contract":id, "revision":contract.revision, "family":contract.study_family,
                "parent":contract.parent_contract, "event_sequence":read.state.sequence,
                "event_head":read.state.head, "database_schema_version":read.database_schema_version,
                "consistency":"single committed coordinator read transaction",
                "integrity":"native validated contract, journal and family budget fold; not scientific acceptance"
            },
            "authority":{"scope":"internal diagnostic report; not a public/sanitized export",
                "accepted":false, "promotion_authorized":false, "scientific_verdict":"unknown",
                "independently_verified":false,
                "qualification":"recorded fixture outcomes are not fully qualified acceptance"},
            "known_limitations":[
                {"id":"t038_hard_address_space","status":"implemented_for_supported_native_fixture",
                    "scope":"finite kernel hard RLIMIT_AS admission; not per-arm memory",
                    "detail":"Supported native execution requires an irreversible finite hard address-space limit. RSS remains a whole-controller measurement, not an allocation enforcement mechanism."},
                {"id":"t038_completion_publication","status":"unresolved",
                    "scope":"hard wall/CPU enforcement through durable observation publication",
                    "detail":"Completion and settlement are atomic and fenced, with a linked postcommit observation. The observation's own commit remains unbounded; neither stored timing nor conservative ceiling charges qualify strict T038 wall/CPU enforcement."},
                {"id":"t038_pre_intent_reservation","status":"repaired_atomic_admission",
                    "scope":"reservation and intent share one transaction",
                    "detail":"Interrupted attempts retain process identity; reconciliation requires proof the old executor is gone and charges unknown costs conservatively. Capture never probes liveness or performs reconciliation."},
                {"id":"independent_verification","status":"not_performed_by_capture",
                    "scope":"scientific qualification and end-to-end hard resource enforcement",
                    "detail":"Capture verifies committed record bindings, not independent reproduction, strict resource enforcement or scientific generalization. Passing fixture tests do not waive these limitations."}
            ],
            "status":{"stage":read.state.stage, "fencing_token":read.state.fencing_token,
                "attempt_counts_by_candidate_verdict":null,
                "attempts_reason":"budget admission/settlement is not a candidate verdict",
                "terminal_transition":if read.state.stage.terminal() {read.events.last()} else {None},
                "study_stop_reason":stop},
            "frozen_contract":contract,
            "budgets":{"scope":"whole study family; native nested branch fold, not sum of allocations",
                "origin":read.family_origin, "family_limits":read.family_limits,
                "contract_limits":contract.limits, "aggregate":read.budgets,
                "records":read.budget_records,
                "cost_semantics":"unknown usage is charged at reservation ceiling, not zero; held is not measured spend; live reservations are not measured peaks; elapsed wall time is a capture-time observation"},
            "supervision":read.supervision, "split_evidence":read.split_evidence,
            "candidates":{"scope":"whole family frozen panels; not candidate attempt verdicts",
                "status":if !links.is_empty() {"native_fixture_record_links_verified"} else if read.frozen_panels.is_empty() {"unavailable"} else {"frozen_identities_only"},
                "panels":read.frozen_panels,
                "execution_receipt_links":if links.is_empty() {Value::Null} else {json!(links)},
                "reason":"native fixture links verify historical event anchors, panel, controls, plan and exact task/seed coverage; they do not qualify strict T038 publication or scientific acceptance"},
            "final_evidence":{"scope":"whole family; verified recorded native fixture facts only",
                "status":if read.final_evidence.is_empty() {"unknown"} else {"record_links_verified"},
                "records":final_evidence,
                "missing_semantics":"absent decision/completion/postcommit observation remains unknown; not success or a candidate verdict",
                "unknowns":["strict hard wall/CPU publication", "independent reproduction", "CAS availability at capture", "generalization", "per-arm costs", "method provenance beyond linked native records"]},
            "baselines":baselines,
            "controls":{"policy":contract.controls,
                "status":if current_final.is_some() {"recorded_fixture_links_verified"} else {"unknown"},
                "results":current_final.map(|e| &e["receipt_links"])},
            "gym_gates":gym_gates,
            "measurement_settings":{"runtime":contract.runtime,"seeds":contract.seeds,
                "statistics":contract.statistics,
                "actual_measurements":current_final.map(|e| e["measurements"].clone())
                    .unwrap_or_else(|| unavailable("no committed measurement receipt links"))},
            "evidence":{"source":contract.source,"best_known":contract.best_known,
                "origin":"unknown","support":"unresolved event references, not verified evidence",
                "proposer":"unknown","method_lineage":null,"gradient_inspection":null,
                "mechanism_verdict":"unknown",
                "artifact_reload_results":current_final.map(|e| &e["export_reload_links"]),
                "comparison_matching":if current_final.is_some() {"verified native fixture record links; not independent reproduction"}
                    else {"unknown; frozen settings are not matched execution evidence"}},
            "events":read.events,
            "family_journal":read.family_journal,
            "integration_gaps":[
                "Native fixture receipt links are verified; non-native per-candidate/arm evidence still needs an adapter.",
                "Controls, gates, measurements and reload results need those links; enablement and stage are not results.",
                "Evidence CAS access policy, origin/support, method lineage and exposure must be independently resolved.",
                "No public-export confidentiality/authorization check; internal records may be sensitive."
            ]
        });
        let payload = serde_json::to_string_pretty(&value)
            .map_err(|_| failure(ErrorCode::InvalidValue, "report serialization failed"))?;
        if payload.len() > MAX_REPORT_BYTES {
            return Err(failure(
                ErrorCode::LimitExceeded,
                "snapshot exceeds byte bounds; no partial report",
            ));
        }
        Ok(Self { payload })
    }

    pub fn render_json(&self) -> String {
        format!("{}\n", self.payload)
    }

    /// Every section contains the exact JSON value; no field is dropped or
    /// recomputed. Backticks in stored messages cannot escape the data fences.
    pub fn render_markdown(&self) -> Result<String> {
        let value: Value = serde_json::from_str(&self.payload)
            .map_err(|_| failure(ErrorCode::IntegrityMismatch, "invalid detached snapshot"))?;
        let mut output = String::from(
            "# Committed study status\n\nInternal diagnostic data; not acceptance authority.\n",
        );
        for (key, value) in value
            .as_object()
            .ok_or_else(|| failure(ErrorCode::IntegrityMismatch, "snapshot must be an object"))?
        {
            let body = serde_json::to_string_pretty(value)
                .map_err(|_| failure(ErrorCode::InvalidValue, "section serialization failed"))?;
            let longest = body.split(|c| c != '`').map(str::len).max().unwrap_or(0);
            let fence = "`".repeat(3.max(longest + 1));
            output.push_str(&format!("\n## {key}\n\n{fence}json\n{body}\n{fence}\n"));
        }
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coordinator::{EventBody, Stage, Transition};
    #[allow(dead_code)]
    mod fixture {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../nf-core/tests/common/study_fixture.rs"
        ));
    }

    // Projection-only fixture; deliberately NOT evidence of atomic DB capture.
    fn read() -> CommittedRead {
        let mut draft = fixture::standard();
        for kind in ["lora", "sft", "distillation", "rl"] {
            let mut arm = draft["baselines"][0].clone();
            arm["kind"] = json!(kind);
            arm["required"] = json!(false);
            arm["enabled"] = json!(false);
            draft["baselines"].as_array_mut().unwrap().push(arm);
        }
        let contract =
            Validated::<StudyContract>::parse(&serde_json::to_vec(&draft).unwrap()).unwrap();
        let id = contract.digest().unwrap();
        let body = EventBody {
            schema_version: 1, contract: id.clone(), sequence: 1,
            previous: ContentDigest::of_bytes(b"nf.coordinator.event.genesis.v1"),
            epoch: 0, owner: "fixture".into(), boot: "fixture".into(),
            monotonic_ms: 0, wall_ms: 0,
            transition: Transition {
                operation: id.clone(), expected: Stage::Frozen, next: Stage::Frozen,
                protocol: fixture::d("protocol"), evidence: fixture::d("unresolved"),
                message: "FAILED retention; gradient-assisted inspection; synthetic_known_mechanism\n``````\n</script><script>claim()</script>".into(),
            },
        };
        let head = CanonicalDocument::parse(&serde_json::to_vec(&body).unwrap(), Limits::default())
            .unwrap()
            .digest();
        let budgets = budgets::Snapshot {
            family: contract.study_family.clone(),
            spent: budgets::Resources {
                cpu_ms: 20,
                ..Default::default()
            },
            held: budgets::Resources {
                workers: 1,
                ..Default::default()
            },
            unknown_charged: budgets::Resources {
                cpu_ms: 20,
                ..Default::default()
            },
            elapsed_wall_ms: None,
            active_roots: 1,
            reservation_count: 2,
            accounting_halted: true,
            reported_overruns: vec![],
            measured_peak_host_bytes: None,
            measured_peak_device_bytes: None,
        };
        CommittedRead {
            family_origin: id.clone(),
            family_limits: contract.limits.clone(),
            state: RunState {
                contract: id,
                stage: Stage::Frozen,
                sequence: 1,
                head: head.clone(),
                fencing_token: 0,
            },
            contract,
            events: vec![Event { digest: head, body }],
            budgets,
            database_schema_version: 5,
            family_journal: json!({"scope":"whole_family","revisions":[]}),
            frozen_panels: Vec::new(),
            final_evidence: Vec::new(),
            budget_records: json!({"outcomes":[{"version":1,"outcome":"incomplete","usage":null},{"version":2,"outcome":"rejected","usage":0}]}),
            supervision: json!({"stop_requests":[{"reason":"owner_stop"}],"liveness":"unknown"}),
            split_evidence: json!({"exposure_gaps":["fixture gap"]}),
        }
    }

    #[test]
    fn native_projection_retains_unknowns_failures_cost_and_provenance() {
        let snapshot = CommittedSnapshot::from_committed_read(read()).unwrap();
        let value: Value = serde_json::from_str(&snapshot.render_json()).unwrap();
        assert_eq!(value["budgets"]["aggregate"]["spent"]["cpu_ms"], 20);
        assert_eq!(
            value["budgets"]["aggregate"]["unknown_charged"]["cpu_ms"],
            20
        );
        assert_eq!(value["budgets"]["aggregate"]["held"]["workers"], 1);
        assert!(value["budgets"]["aggregate"]["elapsed_wall_ms"].is_null());
        assert_eq!(
            value["budgets"]["records"]["outcomes"]
                .as_array()
                .unwrap()
                .len(),
            2
        );
        assert!(value["budgets"]["records"]["outcomes"][0]["usage"].is_null());
        assert_eq!(value["budgets"]["records"]["outcomes"][1]["usage"], 0);
        assert_eq!(value["authority"]["accepted"], false);
        assert_eq!(value["evidence"]["origin"], "unknown");
        assert_eq!(value["controls"]["status"], "unknown");
        assert_eq!(value["gym_gates"][0]["status"], "unknown");
        for arm in value["baselines"].as_array().unwrap() {
            assert_eq!(
                arm["status"],
                if arm["configuration"]["enabled"] == false {
                    "not_run"
                } else {
                    "unknown"
                }
            );
            assert!(arm["costs"].is_null() && arm["scores"].is_null());
        }
        assert!(snapshot
            .render_json()
            .contains("gradient-assisted inspection"));
        assert!(snapshot.render_json().contains("synthetic_known_mechanism"));
    }

    #[test]
    fn markdown_is_lossless_and_stored_text_cannot_break_fences() {
        let snapshot = CommittedSnapshot::from_committed_read(read()).unwrap();
        let value: Value = serde_json::from_str(&snapshot.render_json()).unwrap();
        let markdown = snapshot.render_markdown().unwrap();
        for (key, section) in value.as_object().unwrap() {
            let start = markdown.find(&format!("## {key}\n\n")).unwrap() + key.len() + 5;
            let tail = &markdown[start..];
            let (opening, rest) = tail.split_once('\n').unwrap();
            let fence = opening.strip_suffix("json").unwrap();
            let (body, _) = rest.split_once(&format!("\n{fence}\n")).unwrap();
            assert_eq!(serde_json::from_str::<Value>(body).unwrap(), *section);
        }
        assert!(markdown.contains("```````json"));
        let mut copy = value;
        copy["authority"]["accepted"] = json!(true);
        assert!(
            !serde_json::from_str::<Value>(&snapshot.render_json()).unwrap()["authority"]
                ["accepted"]
                .as_bool()
                .unwrap()
        );
    }

    #[test]
    fn mismatched_native_read_is_rejected() {
        let mut parts = read();
        parts.budgets.family = fixture::d("another-family");
        assert!(CommittedSnapshot::from_committed_read(parts).is_err());
        let mut parts = read();
        parts.events[0].body.transition.message = "tampered".into();
        assert!(CommittedSnapshot::from_committed_read(parts).is_err());
        let mut parts = read();
        parts.state.sequence = 2;
        assert!(CommittedSnapshot::from_committed_read(parts).is_err());
    }

    #[test]
    fn stored_authority_claims_never_promote_the_report() {
        let mut parts = read();
        parts.supervision = json!({"accepted":true,"scientific_verdict":"proven"});
        let snapshot = CommittedSnapshot::from_committed_read(parts).unwrap();
        let value: Value = serde_json::from_str(&snapshot.render_json()).unwrap();
        assert_eq!(value["supervision"]["accepted"], true);
        assert_eq!(value["authority"]["accepted"], false);
        assert_eq!(value["authority"]["scientific_verdict"], "unknown");
    }

    #[test]
    fn review_limitations_are_explicit_and_never_waived_by_stored_claims() {
        let mut parts = read();
        parts.supervision = json!({"accepted":true,"independently_verified":true,
            "hard_memory_enforced":true,"publication_within_deadline":true,
            "known_limitations":[]});
        let snapshot = CommittedSnapshot::from_committed_read(parts).unwrap();
        let value: Value = serde_json::from_str(&snapshot.render_json()).unwrap();
        assert_eq!(value["authority"]["accepted"], false);
        assert_eq!(value["authority"]["promotion_authorized"], false);
        assert_eq!(value["authority"]["independently_verified"], false);
        assert_eq!(value["candidates"]["status"], "unavailable");
        assert!(value["candidates"]["execution_receipt_links"].is_null());
        let limitations = value["known_limitations"].as_array().unwrap();
        for id in ["t038_completion_publication"] {
            let limitation = limitations.iter().find(|row| row["id"] == id).unwrap();
            assert_eq!(limitation["status"], "unresolved");
        }
        assert!(snapshot
            .render_markdown()
            .unwrap()
            .contains("Passing fixture tests do not waive these limitations"));
    }

    #[test]
    fn oversized_retained_data_fails_without_partial_report() {
        let mut parts = read();
        parts.supervision = json!({"message":"x".repeat(MAX_REPORT_BYTES)});
        let error = CommittedSnapshot::from_committed_read(parts).unwrap_err();
        assert_eq!(error.code, ErrorCode::LimitExceeded);
    }

    #[test]
    fn typed_stop_request_is_current_hash_bound_and_not_a_termination_claim() {
        let mut parts = read();
        let record = json!({"schema_version":1,"contract":parts.state.contract,
            "operation":fixture::d("stop"),"stage":"frozen","reason":"owner_stop",
            "boot":"fixture","monotonic_ms":1});
        let document =
            CanonicalDocument::parse(&serde_json::to_vec(&record).unwrap(), Limits::default())
                .unwrap();
        parts.supervision = json!({"stop_requests":[{"key":fixture::d("stop"),
            "digest":document.digest(),"record":record}]});
        let snapshot = CommittedSnapshot::from_committed_read(parts).unwrap();
        let value: Value = serde_json::from_str(&snapshot.render_json()).unwrap();
        assert_eq!(value["status"]["study_stop_reason"]["status"], "recorded");
        assert_eq!(value["status"]["study_stop_reason"]["reason"], "owner_stop");
        assert_eq!(value["authority"]["accepted"], false);
        assert!(value["status"]["study_stop_reason"]["scope"]
            .as_str()
            .unwrap()
            .contains("not proof of worker termination"));
        let mut parts = read();
        parts.supervision = value["supervision"].clone();
        parts.supervision["stop_requests"][0]["digest"] = json!(fixture::d("forged"));
        assert_eq!(
            CommittedSnapshot::from_committed_read(parts)
                .unwrap_err()
                .code,
            ErrorCode::IntegrityMismatch
        );
        let mut parts = read();
        parts.supervision = value["supervision"].clone();
        parts.supervision["stop_requests"][0]["record"]["contract"] = json!(fixture::d("old"));
        let snapshot = CommittedSnapshot::from_committed_read(parts).unwrap();
        let value: Value = serde_json::from_str(&snapshot.render_json()).unwrap();
        assert_eq!(
            value["status"]["study_stop_reason"]["status"],
            "unavailable"
        );
    }

    #[test]
    fn capture_missing_file_does_not_initialize_or_fall_back() {
        let error = CommittedSnapshot::capture(
            Path::new("/nonexistent/events.sqlite3"),
            &fixture::d("contract"),
        )
        .unwrap_err();
        assert_eq!(error.code, ErrorCode::Io);
        assert!(!Path::new("/nonexistent/events.sqlite3").exists());
    }
}

//! Scientific-loop revision handling and comparison invalidation (T057).
//!
//! Permits controlled return to `Profile`, `Baseline`, and `Diagnose` during
//! iterative experimentation (`Baseline..=Search`), blocks search restarts after
//! final holdout feedback, and computes exact comparison invalidations across
//! linked study revisions while preserving cumulative budgets and exposures.

use super::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RevisionComparisonDelta {
    pub schema_version: u32,
    pub family: ContentDigest,
    pub parent_contract: ContentDigest,
    pub child_contract: ContentDigest,
    pub parent_revision: u32,
    pub child_revision: u32,
    pub objective_changed: bool,
    pub search_space_changed: bool,
    pub source_changed: bool,
    pub best_known_changed: bool,
    pub all_comparisons_invalidated: bool,
    pub invalidated_bindings: BTreeSet<ContentDigest>,
    pub retained_bindings: BTreeSet<ContentDigest>,
    pub rebaseline_required: bool,
}

impl Ledger {
    /// Inspect two linked revisions in the same study family and determine which
    /// paired gym/baseline comparisons were invalidated by gym, scorer, threshold,
    /// objective, or search-space changes.
    pub fn evaluate_revision_invalidation(
        &self,
        parent_id: &ContentDigest,
        child_id: &ContentDigest,
    ) -> Result<RevisionComparisonDelta> {
        self.files.verify(self.limits.max_database_bytes)?;
        let parent = self.contract(parent_id)?;
        let child = self.contract(child_id)?;
        if parent.study_family != child.study_family
            || child.revision != parent.revision + 1
            || child.parent_contract.as_ref() != Some(parent_id)
        {
            return Err(failure(
                ErrorCode::RevisionConflict,
                "contracts are not consecutive linked revisions in the same study family",
            ));
        }
        let objective_changed = canonical(&parent.objective)? != canonical(&child.objective)?;
        let search_space_changed =
            parent.intervention.transformation_space != child.intervention.transformation_space;
        let source_changed = canonical(&parent.source)? != canonical(&child.source)?;
        let best_known_changed = canonical(&parent.best_known)? != canonical(&child.best_known)?;
        let runtime_changed = canonical(&parent.runtime)? != canonical(&child.runtime)?;

        let all_comparisons_invalidated =
            objective_changed || source_changed || best_known_changed || runtime_changed;

        let parent_bindings: BTreeMap<_, _> = parent
            .portfolio
            .bindings
            .iter()
            .map(|b| (&b.id, b))
            .collect();

        let mut invalidated_bindings = BTreeSet::new();
        let mut retained_bindings = BTreeSet::new();

        for child_b in &child.portfolio.bindings {
            if all_comparisons_invalidated {
                invalidated_bindings.insert(child_b.id.clone());
                continue;
            }
            let Some(parent_b) = parent_bindings.get(&child_b.id) else {
                invalidated_bindings.insert(child_b.id.clone());
                continue;
            };
            let parent_gym = &parent.portfolio.gyms[&parent_b.gym];
            let child_gym = &child.portfolio.gyms[&child_b.gym];
            let same_binding = parent_b.gym == child_b.gym
                && parent_b.validation == child_b.validation
                && canonical(&parent_b.roles)? == canonical(&child_b.roles)?
                && canonical(&parent_b.metrics)? == canonical(&child_b.metrics)?
                && canonical(&parent_b.splits)? == canonical(&child_b.splits)?
                && canonical(&parent_b.schedule)? == canonical(&child_b.schedule)?
                && canonical(parent_gym)? == canonical(child_gym)?;
            if same_binding {
                retained_bindings.insert(child_b.id.clone());
            } else {
                invalidated_bindings.insert(child_b.id.clone());
            }
        }

        for parent_id_key in parent_bindings.keys() {
            if !child
                .portfolio
                .bindings
                .iter()
                .any(|b| &b.id == *parent_id_key)
            {
                invalidated_bindings.insert((*parent_id_key).clone());
            }
        }

        let rebaseline_required =
            all_comparisons_invalidated || !invalidated_bindings.is_empty() || search_space_changed;

        Ok(RevisionComparisonDelta {
            schema_version: 1,
            family: child.study_family.clone(),
            parent_contract: parent_id.clone(),
            child_contract: child_id.clone(),
            parent_revision: parent.revision,
            child_revision: child.revision,
            objective_changed,
            search_space_changed,
            source_changed,
            best_known_changed,
            all_comparisons_invalidated,
            invalidated_bindings,
            retained_bindings,
            rebaseline_required,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coordinator::budgets::{
        OutcomeKind, Request, Resources, Settlement, Usage, WorkKind,
    };
    use crate::coordinator::splits::{publish_dataset, Dataset, Item, Recipient};
    use nf_checkpoint::store::{ArtifactStore, StoreLimits};
    use nf_core::study::SplitRole;
    use serde_json::json;
    use std::sync::atomic::{AtomicU64, Ordering};

    #[allow(dead_code)]
    mod fixture {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../nf-core/tests/common/study_fixture.rs"
        ));
    }

    static NEXT: AtomicU64 = AtomicU64::new(0);

    fn temp_root(tag: &str) -> std::path::PathBuf {
        let base = std::env::var_os("KIROCREW_SCRATCH")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(std::env::temp_dir);
        let dir = base.join(format!(
            "nf-scientific-loop-{tag}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn parse_contract(v: &serde_json::Value) -> Validated<StudyContract> {
        Validated::parse(&serde_json::to_vec(v).unwrap()).unwrap()
    }

    fn trans(op: &str, from: Stage, to: Stage) -> Transition {
        Transition {
            operation: fixture::d(op),
            expected: from,
            next: to,
            protocol: fixture::d("scientific-loop-protocol"),
            evidence: fixture::d(op),
            message: format!("transition {from:?} -> {to:?}"),
        }
    }

    #[test]
    fn scientific_loop_permits_return_to_profile_baseline_diagnose_and_blocks_post_verify_restart()
    {
        let root = temp_root("stages");
        let mut ledger = Ledger::open(&root.join("ledger"), LedgerLimits::default()).unwrap();
        let c1 = parse_contract(&fixture::standard());
        let id1 = ledger.freeze(&c1).unwrap();
        let lease = ledger.acquire(&id1, "scientist", 60_000).unwrap();

        // Advance Frozen -> Preflight -> Profile -> Baseline -> Diagnose -> Design -> Execute -> Evaluate -> Search
        for (i, (from, to)) in [
            (Stage::Frozen, Stage::Preflight),
            (Stage::Preflight, Stage::Profile),
            (Stage::Profile, Stage::Baseline),
            (Stage::Baseline, Stage::Diagnose),
            (Stage::Diagnose, Stage::Design),
            (Stage::Design, Stage::Execute),
            (Stage::Execute, Stage::Evaluate),
            (Stage::Evaluate, Stage::Search),
        ]
        .into_iter()
        .enumerate()
        {
            ledger
                .advance(&lease, trans(&format!("step-{i}"), from, to))
                .unwrap();
        }

        // Loop back from Search -> Baseline -> Diagnose -> Profile -> Baseline -> Design -> Execute -> Evaluate
        for (i, (from, to)) in [
            (Stage::Search, Stage::Baseline),
            (Stage::Baseline, Stage::Diagnose),
            (Stage::Diagnose, Stage::Profile),
            (Stage::Profile, Stage::Baseline),
            (Stage::Baseline, Stage::Design),
            (Stage::Design, Stage::Execute),
            (Stage::Execute, Stage::Evaluate),
            (Stage::Evaluate, Stage::Diagnose),
            (Stage::Diagnose, Stage::Design),
            (Stage::Design, Stage::Execute),
            (Stage::Execute, Stage::Evaluate),
            (Stage::Evaluate, Stage::Verify),
        ]
        .into_iter()
        .enumerate()
        {
            ledger
                .advance(&lease, trans(&format!("loop-{i}"), from, to))
                .unwrap();
        }

        assert_eq!(ledger.state(&id1).unwrap().stage, Stage::Verify);

        // Once in Verify (final feedback), returning to Search, Evaluate, Design, Diagnose, Baseline, or Profile is forbidden!
        for forbidden in [
            Stage::Search,
            Stage::Evaluate,
            Stage::Execute,
            Stage::Design,
            Stage::Diagnose,
            Stage::Baseline,
            Stage::Profile,
        ] {
            assert_eq!(
                ledger
                    .advance(&lease, trans("illegal-restart", Stage::Verify, forbidden))
                    .unwrap_err()
                    .code,
                ErrorCode::InvalidTransition
            );
        }

        drop(ledger);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn scientific_loop_revisions_invalidate_changed_gyms_thresholds_and_inherit_budgets_and_exposure(
    ) {
        let root = temp_root("revisions");
        let store = ArtifactStore::create(
            &root.join("store"),
            StoreLimits {
                total_bytes: 16 << 20,
                max_blob_bytes: 1 << 20,
                max_entries: 1024,
            },
        )
        .unwrap();
        let mut ledger = Ledger::open(&root.join("ledger"), LedgerLimits::default()).unwrap();

        let mut v1 = fixture::standard();
        for b in v1["portfolio"]["bindings"].as_array_mut().unwrap() {
            for role in ["development", "selection", "holdout"] {
                let split_role = match role {
                    "development" => SplitRole::Development,
                    "selection" => SplitRole::Selection,
                    _ => SplitRole::Holdout,
                };
                let task_id = b["splits"][role]["items"][0].clone();
                let inp = store
                    .put(
                        canonical(&json!({"x": role, "b": b["id"]}))
                            .unwrap()
                            .as_slice(),
                    )
                    .unwrap()
                    .digest;
                let lbl = store
                    .put(canonical(&json!(1.0)).unwrap().as_slice())
                    .unwrap()
                    .digest;
                let ds = publish_dataset(
                    &store,
                    &Dataset {
                        schema_version: 1,
                        role: split_role,
                        items: vec![Item {
                            identity: serde_json::from_value(task_id).unwrap(),
                            input: inp,
                            label: lbl,
                        }],
                    },
                )
                .unwrap();
                b["splits"][role]["dataset"] = json!(ds);
            }
        }
        fixture::refresh(&mut v1);
        let c1 = parse_contract(&v1);
        let id1 = ledger.freeze(&c1).unwrap();
        let lease1 = ledger.acquire(&id1, "scientist-r1", 60_000).unwrap();

        // Spend part of the family budget and record development exposure in revision 1
        let b0_id = c1.portfolio.bindings[0].id.clone();
        ledger.bind_split_catalog(&lease1, &store, &b0_id).unwrap();
        let res = ledger
            .reserve(
                &lease1,
                Request {
                    schema_version: 1,
                    operation: fixture::d("r1-attempt-1"),
                    attempt: 0,
                    parent: None,
                    kind: WorkKind::Evaluation,
                    resources: Resources {
                        cpu_ms: 500,
                        attempts: 2,
                        evaluations: 10,
                        host_bytes: 1024,
                        workers: 1,
                        ..Default::default()
                    },
                    wall_ms: 2000,
                    external: None,
                },
            )
            .unwrap();
        ledger
            .settle(
                &lease1,
                &res,
                Settlement {
                    operation: fixture::d("r1-attempt-1-settle"),
                    outcome: OutcomeKind::Completed,
                    usage: Usage {
                        cpu_ms: Some(400),
                        attempts: Some(2),
                        evaluations: Some(10),
                        ..Default::default()
                    },
                    resources_released: true,
                    evidence: fixture::d("r1-evidence"),
                },
            )
            .unwrap();
        let score_blob = store.put(&b"0.75"[..]).unwrap().digest;
        ledger
            .disclose_score(
                &lease1,
                &store,
                exposure::Disclosure {
                    operation: fixture::d("r1-dev-score"),
                    binding: b0_id.clone(),
                    item: c1.portfolio.bindings[0].splits.development.items[0]
                        .item
                        .clone(),
                    split: SplitRole::Development,
                    recipient: "scientist".into(),
                    role: Recipient::Proposer,
                    fields: [exposure::Field::Summary].into_iter().collect(),
                    artifact: score_blob,
                    context: exposure::Context::default(),
                },
            )
            .unwrap();
        ledger.release(&lease1).unwrap();

        // Create Revision 2 changing binding[0]'s threshold and binding[1]'s scorer
        let mut v2 = v1.clone();
        v2["revision"] = json!(2);
        v2["parent_contract"] = json!(id1);
        v2["approval_receipt"] = json!(fixture::d("owner-approval-r2"));
        v2["portfolio"]["bindings"][0]["metrics"][0]["threshold"]["min_gain"] = json!(0.15);
        let gym1_key = v2["portfolio"]["bindings"][1]["gym"]
            .as_str()
            .unwrap()
            .to_owned();
        v2["portfolio"]["gyms"][&gym1_key]["scorer"] = json!(fixture::d("new-scorer-v2"));
        fixture::rebind_gym(&mut v2, 1);
        fixture::refresh(&mut v2);

        let c2 = parse_contract(&v2);
        let id2 = ledger.freeze(&c2).unwrap();
        let delta = ledger.evaluate_revision_invalidation(&id1, &id2).unwrap();
        assert!(!delta.all_comparisons_invalidated);
        assert!(delta.rebaseline_required);
        assert!(delta
            .invalidated_bindings
            .contains(&c1.portfolio.bindings[0].id));
        assert!(delta
            .invalidated_bindings
            .contains(&c1.portfolio.bindings[1].id));
        assert!(delta
            .retained_bindings
            .contains(&c1.portfolio.bindings[2].id));
        assert_eq!(delta.retained_bindings.len(), 3);

        // Cumulative budget and exposure history are inherited by Revision 2!
        let snap2 = ledger.budget_snapshot(&id2).unwrap();
        assert_eq!(snap2.spent.attempts, 2);
        assert_eq!(snap2.spent.evaluations, 10);
        assert_eq!(snap2.spent.cpu_ms, 400);

        let lease2 = ledger.acquire(&id2, "scientist-r2", 60_000).unwrap();
        let exposures = ledger.exposure_events(&lease2, 0, 100).unwrap();
        assert!(!exposures.is_empty());
        ledger.release(&lease2).unwrap();

        // Revision 3 changing the objective invalidates ALL comparisons
        let mut v3 = v2;
        v3["revision"] = json!(3);
        v3["parent_contract"] = json!(id2);
        v3["approval_receipt"] = json!(fixture::d("owner-approval-r3"));
        v3["objective"]["description"] = json!("Revised scientific objective");
        fixture::refresh(&mut v3);
        let c3 = parse_contract(&v3);
        let id3 = ledger.freeze(&c3).unwrap();
        let delta3 = ledger.evaluate_revision_invalidation(&id2, &id3).unwrap();
        assert!(delta3.objective_changed);
        assert!(delta3.all_comparisons_invalidated);
        assert!(delta3.retained_bindings.is_empty());
        assert_eq!(delta3.invalidated_bindings.len(), 5);

        drop(ledger);
        std::fs::remove_dir_all(root).unwrap();
    }
}

//! Storage-only cleanup tests: these values never mint native reload witnesses.
use super::*;

#[test]
fn global_cleanup_inventory_tracks_promotion_admission_and_bound_reaping() {
    for mode in [
        "unknown",
        "reload_only",
        "foreign_supervisor",
        "reaped",
        "prelaunch",
        "fake_reboot",
    ] {
        let path =
            std::env::temp_dir().join(format!("promotion-inventory-{}-{mode}", std::process::id()));
        let mut ledger = Ledger::open(&path, LedgerLimits::default()).unwrap();
        let c =
            Validated::<StudyContract>::parse(&serde_json::to_vec(&fixture::standard()).unwrap())
                .unwrap();
        let id = ledger.freeze(&c).unwrap();
        assert!(supervision::all_cleanup_confirmed(&ledger.connection, &id).unwrap());
        let tx = ledger.connection.transaction().unwrap();
        let boot = files::moment().unwrap().boot;
        let intent = json!({"native_process":process::Owner::current().unwrap(),"boot":boot});
        let intent_id = record(&tx, &id, "intent", &intent).unwrap();
        record(&tx, &id, "child_intent", &json!({"intent":intent_id})).unwrap();
        assert!(!supervision::all_cleanup_confirmed(&tx, &id).unwrap());
        assert!(child_inventory::observe_reboot(&tx, &id, &intent_id, &intent).is_err());
        if ["reload_only", "foreign_supervisor", "reaped"].contains(&mode) {
            record(&tx, &id, "reload", &json!({"cleanup_confirmed":true})).unwrap();
        }
        if ["foreign_supervisor", "reaped"].contains(&mode) {
            let binding = if mode == "reaped" {
                intent_id.clone()
            } else {
                fixture::d("foreign-intent")
            };
            record(
                &tx,
                &id,
                "supervision",
                &json!({"intent":binding,"cleanup_confirmed":true}),
            )
            .unwrap();
        }
        if mode == "prelaunch" {
            record(
                &tx,
                &id,
                "attempt_end",
                &json!({"intent":intent_id,"cleanup_confirmed":true,
                "child_entered":false,"status":"incomplete_no_rerun"}),
            )
            .unwrap();
        }
        if mode == "fake_reboot" {
            // Even canonical, correctly linked JSON cannot assert a different
            // kernel boot while the native launch boot is still current.
            record(
                &tx,
                &id,
                "child_recovery",
                &json!({"intent":intent_id,"launch_boot":boot,
                "observed_boot":"invented-boot","status":"previous_boot_processes_gone"}),
            )
            .unwrap();
        }
        tx.commit().unwrap();
        drop(ledger);
        let ledger = Ledger::open(&path, LedgerLimits::default()).unwrap();
        let clean = supervision::all_cleanup_confirmed(&ledger.connection, &id);
        if mode == "fake_reboot" {
            assert!(clean.is_err());
        } else {
            assert_eq!(
                clean.unwrap(),
                mode == "reaped" || mode == "prelaunch",
                "{mode}"
            );
        }
        drop(ledger);
        std::fs::remove_dir_all(path).unwrap();
    }
}

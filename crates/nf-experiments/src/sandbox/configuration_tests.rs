//! Capability representation compatibility never broadens the frozen profile.
use super::*;

const NAME: &str = "nf-sandbox-00000000000000000000000000000000";
fn root() -> &'static Path {
    Path::new("/private/controller/nf-sandbox-00000000000000000000000000000000")
}
fn inspection() -> Value {
    json!({
        "HostConfig": {
            "NetworkMode":"none", "ReadonlyRootfs":true, "Privileged":false,
            "PidMode":"", "IpcMode":"private", "Memory":MEMORY_BYTES,
            "MemorySwap":MEMORY_BYTES, "PidsLimit":16, "NanoCpus":1000000000,
            "ShmSize":65536, "CgroupnsMode":"private", "CapDrop":["ALL"],
            "CapAdd":["CAP_KILL","CAP_SETGID","CAP_SETUID"],
            "SecurityOpt":["no-new-privileges"]
        },
        "Config":{"Image":IMAGE,"Labels":{"org.neural-foundry.owner":NAME}},
        "Mounts":[
            {"Destination":"/guard","Source":root().join("guard"),"Type":"bind","RW":false},
            {"Destination":"/job","Source":root().join("job"),"Type":"bind","RW":false}
        ]
    })
}

#[test]
fn exact_capability_set_accepts_every_order_and_optional_prefix_combination() {
    for order in [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ] {
        for mask in 0..8 {
            let mut value = inspection();
            let names = ["KILL", "SETGID", "SETUID"];
            let caps: Vec<_> = order
                .iter()
                .map(|&index| {
                    if mask & (1 << index) != 0 {
                        format!("CAP_{}", names[index])
                    } else {
                        names[index].to_owned()
                    }
                })
                .collect();
            value["HostConfig"]["CapAdd"] = json!(caps);
            QualifiedDocker::verify_configuration(&value, NAME, root()).unwrap();
        }
    }
}

#[test]
fn exact_capability_set_rejects_missing_extra_duplicate_alias_and_malformed_values() {
    for caps in [
        json!(["KILL", "SETGID"]),
        json!(["KILL", "SETGID", "SETUID", "SYS_ADMIN"]),
        json!(["KILL", "KILL", "SETUID"]),
        json!(["KILL", "CAP_KILL", "SETUID"]),
        json!(["KILL", "SETGID", "ALL"]),
        json!(["KILL", "SETGID", "SYS_ADMIN"]),
        json!(["kill", "SETGID", "SETUID"]),
        json!(["CAP_CAP_KILL", "SETGID", "SETUID"]),
        json!([" KILL", "SETGID", "SETUID"]),
        json!(["KILL", "SETGID", null]),
        json!(["KILL", "SETGID", 1]),
        json!({"KILL":true,"SETGID":true,"SETUID":true}),
        json!("KILL,SETGID,SETUID"),
        json!(null),
        json!([]),
    ] {
        let mut value = inspection();
        value["HostConfig"]["CapAdd"] = caps.clone();
        assert!(
            QualifiedDocker::verify_configuration(&value, NAME, root()).is_err(),
            "{caps}"
        );
    }
}

#[test]
fn capability_normalization_preserves_every_other_exact_host_security_field() {
    for (field, bad) in [
        ("NetworkMode", json!("host")),
        ("ReadonlyRootfs", json!(false)),
        ("Privileged", json!(true)),
        ("PidMode", json!("host")),
        ("IpcMode", json!("host")),
        ("Memory", json!(MEMORY_BYTES + 1)),
        ("MemorySwap", json!(-1)),
        ("PidsLimit", json!(-1)),
        ("NanoCpus", json!(0)),
        ("ShmSize", json!(65537)),
        ("CgroupnsMode", json!("host")),
        ("CapDrop", json!([])),
        ("SecurityOpt", json!([])),
    ] {
        let mut value = inspection();
        value["HostConfig"]["CapAdd"] = json!(["SETUID", "KILL", "SETGID"]);
        value["HostConfig"][field] = bad;
        assert!(
            QualifiedDocker::verify_configuration(&value, NAME, root()).is_err(),
            "{field}"
        );
    }
    for pointer in [
        "/Config/Image",
        "/Config/Labels/org.neural-foundry.owner",
        "/Mounts/0/Source",
        "/Mounts/0/Type",
        "/Mounts/0/Destination",
    ] {
        let mut value = inspection();
        *value.pointer_mut(pointer).unwrap() = json!("foreign");
        assert!(
            QualifiedDocker::verify_configuration(&value, NAME, root()).is_err(),
            "{pointer}"
        );
    }
    let mut value = inspection();
    value["Mounts"][0]["RW"] = json!(true);
    assert!(QualifiedDocker::verify_configuration(&value, NAME, root()).is_err());
}

//! Deterministic mutation smoke, not a coverage-guided fuzz qualification.
use nf_checkpoint::serialization::{serialize_tensors, SerializationLimits, TensorPayload};
use nf_checkpoint::tensor_file::{inspect_safetensors, TensorFileLimits};
use nf_core::canonical::{CanonicalDocument, ContentDigest, Limits};
use nf_core::schemas::DType;
use nf_weight_ir::program::parse_program;
use nf_weight_ir::search_space::parse_search_space;
use std::collections::BTreeMap;
#[derive(Debug, Default)]
pub struct Counts {
    pub cases: usize,
    pub canonical: usize,
    pub programs: usize,
    pub spaces: usize,
    pub checkpoints: usize,
}
pub fn run() -> Counts {
    const MAX_INPUT: usize = 8192;
    const CASES: usize = 4096;
    let fixture = crate::compiler_fixture::Fixture::new(false);
    let bytes = 1.0f32.to_le_bytes();
    let checkpoint = serialize_tensors(
        &BTreeMap::from([(
            "w".to_owned(),
            TensorPayload {
                dtype: DType::F32,
                shape: &[1],
                bytes: &bytes,
            },
        )]),
        SerializationLimits::default(),
    )
    .unwrap()
    .bytes;
    let seeds = [
        b"{}".to_vec(),
        serde_json::to_vec(&fixture.program).unwrap(),
        serde_json::to_vec(&fixture.space).unwrap(),
        checkpoint,
        b"{\"x\":1,\"x\":2}".to_vec(),
        vec![255],
        b"1e400".to_vec(),
    ];
    assert!(seeds.iter().all(|seed| seed.len() <= MAX_INPUT));
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    let mut random = 0x9e3779b97f4a7c15u64;
    let mut counts = Counts::default();
    for case in 0..CASES {
        assert!(
            std::time::Instant::now() < deadline,
            "bounded fuzz smoke deadline exceeded"
        );
        let mut input = seeds[case % seeds.len()].clone();
        random ^= random << 13;
        random ^= random >> 7;
        random ^= random << 17;
        if case >= seeds.len() {
            match case % 4 {
                0 if !input.is_empty() => {
                    let index = random as usize % input.len();
                    input[index] ^= (random >> 32) as u8;
                }
                1 => {
                    input.truncate(random as usize % (input.len() + 1));
                }
                2 if input.len() < MAX_INPUT => {
                    let index = random as usize % (input.len() + 1);
                    input.insert(index, (random >> 24) as u8);
                }
                _ => {
                    input = (0..random as usize % 128)
                        .map(|i| (random.rotate_left(i as u32 % 64) >> 8) as u8)
                        .collect();
                }
            }
        }
        assert!(input.len() <= MAX_INPUT);
        if let Ok(document) = CanonicalDocument::parse(
            &input,
            Limits {
                max_bytes: MAX_INPUT,
                max_nodes: 4096,
                max_depth: 32,
            },
        ) {
            counts.canonical += 1;
            assert_eq!(
                CanonicalDocument::parse(document.bytes(), Limits::default())
                    .unwrap()
                    .digest(),
                document.digest()
            );
        }
        if let Ok(program) = parse_program(&input) {
            counts.programs += 1;
            let bytes = serde_json::to_vec(&*program).unwrap();
            assert_eq!(
                parse_program(&bytes).unwrap().digest().unwrap(),
                program.digest().unwrap()
            );
        }
        if let Ok(space) = parse_search_space(&input) {
            counts.spaces += 1;
            let bytes = serde_json::to_vec(&*space).unwrap();
            assert_eq!(
                parse_search_space(&bytes).unwrap().digest().unwrap(),
                space.digest().unwrap()
            );
        }
        if let Ok(file) = inspect_safetensors(
            &input,
            TensorFileLimits {
                max_file_bytes: MAX_INPUT,
                max_header_bytes: 2048,
                max_tensors: 16,
                max_rank: 8,
            },
        ) {
            counts.checkpoints += 1;
            assert_eq!(file.file_digest, ContentDigest::of_bytes(&input));
        }
        counts.cases += 1;
    }
    assert_eq!(counts.cases, CASES);
    assert!(
        counts.canonical > 0 && counts.programs > 0 && counts.spaces > 0 && counts.checkpoints > 0,
        "all parser families need successful seed coverage"
    );
    counts
}

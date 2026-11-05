//! Frozen study eligibility schemas. Validation never dispatches or accepts a model.
mod validation;
use crate::canonical::{CanonicalDocument, ContentDigest, Limits};
use crate::error::{CoreError, ErrorCode, Result};
use crate::schemas::{DType, DataClass, FactoryRef, Permissions, ToolKind, Validate};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
fn invalid(path: &str, message: &str) -> CoreError {
    CoreError::new(ErrorCode::InvalidValue, path, message)
}
fn version(value: u32) -> Result<()> {
    if value != 1 {
        Err(CoreError::new(
            ErrorCode::UnsupportedVersion,
            "/schema_version",
            "only study schemas v1 supported",
        ))
    } else {
        Ok(())
    }
}
fn name(value: &str, path: &str) -> Result<()> {
    if value.trim().is_empty() || value.len() > 512 {
        Err(invalid(path, "nonempty bounded name required"))
    } else {
        Ok(())
    }
}
fn factory(value: &FactoryRef, path: &str) -> Result<()> {
    name(&value.id, path)?;
    name(&value.version, path)
}
fn hash<T: Serialize>(value: &T) -> Result<ContentDigest> {
    let raw = serde_json::to_vec(value).map_err(|e| invalid("", &e.to_string()))?;
    Ok(CanonicalDocument::parse(&raw, Limits::default())?.digest())
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelIdentity {
    pub checkpoint: ContentDigest,
    pub manifest: ContentDigest,
    pub parameters: ContentDigest,
    pub buffers: ContentDigest,
    pub graph: ContentDigest,
    pub config: ContentDigest,
    pub tokenizer: Option<ContentDigest>,
    pub system_state: Option<ContentDigest>,
    pub runtime_artifact: Option<ContentDigest>,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SplitRole {
    Development,
    Selection,
    Holdout,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExposureState {
    KnownUnexposed,
    KnownExposed,
    Unknown,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskIdentity {
    pub item: ContentDigest,
    pub group: ContentDigest,
    pub family: ContentDigest,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SplitManifest {
    pub schema_version: u32,
    pub role: SplitRole,
    pub dataset: ContentDigest,
    pub items: Vec<TaskIdentity>,
    pub exposure: ExposureState,
    pub exposure_ledger: ContentDigest,
    pub access_policy: ContentDigest,
}
impl SplitManifest {
    pub fn digest(&self) -> Result<ContentDigest> {
        hash(self)
    }
    fn validate_role(&self, role: SplitRole) -> Result<()> {
        version(self.schema_version)?;
        if self.role != role || self.items.is_empty() || self.items.len() > 4096 {
            return Err(invalid(
                "/splits",
                "exact nonempty canonical split roles/items required",
            ));
        }
        let mut items = BTreeSet::new();
        for item in &self.items {
            if !items.insert(&item.item) {
                return Err(invalid("/splits/items", "duplicate task in split"));
            }
        }
        if role == SplitRole::Holdout && self.exposure != ExposureState::KnownUnexposed {
            return Err(invalid(
                "/splits/holdout/exposure",
                "new locked holdout requires known unexposed history",
            ));
        }
        Ok(())
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GymSplits {
    pub development: SplitManifest,
    pub selection: SplitManifest,
    pub holdout: SplitManifest,
}
impl GymSplits {
    fn all(&self) -> [(SplitRole, &SplitManifest); 3] {
        [
            (SplitRole::Development, &self.development),
            (SplitRole::Selection, &self.selection),
            (SplitRole::Holdout, &self.holdout),
        ]
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GymOrigin {
    Foundry,
    User,
    Generated,
    Community,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionTrust {
    TrustedFixture,
    UntrustedSubprocess,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GymRole {
    Target,
    Retention,
    Diagnostic,
    Efficiency,
    Safety,
    IndependentVerification,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetricDomain {
    Quality,
    Latency,
    Memory,
    Cost,
    Mechanism,
    Safety,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Direction {
    HigherIsBetter,
    LowerIsBetter,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetricUnit {
    Fraction,
    Score,
    Count,
    Bytes,
    Nanoseconds,
    Milliseconds,
    Seconds,
    TokensPerSecond,
    UsdMicros,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MetricSchema {
    pub domain: MetricDomain,
    pub direction: Direction,
    pub unit: MetricUnit,
    pub lower: Option<f64>,
    pub upper: Option<f64>,
}
impl MetricSchema {
    pub fn validate(&self) -> Result<()> {
        if self.lower.is_some_and(|x| !x.is_finite())
            || self.upper.is_some_and(|x| !x.is_finite())
            || self.lower.zip(self.upper).is_some_and(|(a, b)| a > b)
        {
            return Err(invalid("/metrics", "invalid finite metric range"));
        }
        if self.unit == MetricUnit::Fraction && (self.lower != Some(0.0) || self.upper != Some(1.0))
        {
            return Err(invalid("/metrics", "fraction metrics require [0,1] range"));
        }
        if (self.domain == MetricDomain::Memory && self.unit != MetricUnit::Bytes)
            || (self.domain == MetricDomain::Latency
                && !matches!(
                    self.unit,
                    MetricUnit::Nanoseconds | MetricUnit::Milliseconds | MetricUnit::Seconds
                ))
        {
            return Err(invalid("/metrics/unit", "measurement domain/unit mismatch"));
        }
        Ok(())
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GymManifest {
    pub schema_version: u32,
    pub package: String,
    pub name: String,
    pub version: String,
    pub origin: GymOrigin,
    pub adapter: FactoryRef,
    pub code: ContentDigest,
    pub dependencies: ContentDigest,
    pub image: Option<ContentDigest>,
    pub dataset: ContentDigest,
    pub input_schema: ContentDigest,
    pub output_schema: ContentDigest,
    pub scorer: ContentDigest,
    pub oracle: ContentDigest,
    pub lifecycle_protocol: ContentDigest,
    pub model_interface: FactoryRef,
    pub data_policy: ContentDigest,
    pub scoring_protocol: ContentDigest,
    pub privacy_policy: ContentDigest,
    pub metrics: BTreeMap<String, MetricSchema>,
    pub execution_trust: ExecutionTrust,
    pub isolation_policy: ContentDigest,
    pub validation: ContentDigest,
    pub approval: Option<ContentDigest>,
    pub usage_rights: String,
    pub confidentiality: crate::schemas::AccessClass,
    pub max_steps: u32,
    pub max_episode_ms: u64,
    pub max_output_bytes: u64,
}
impl GymManifest {
    pub fn digest(&self) -> Result<ContentDigest> {
        hash(self)
    }
}
impl Validate for GymManifest {
    fn validate(&self) -> Result<()> {
        version(self.schema_version)?;
        for text in [&self.package, &self.name, &self.version, &self.usage_rights] {
            name(text, "/gym")?;
        }
        factory(&self.adapter, "/gym/adapter")?;
        if self.metrics.is_empty()
            || self.metrics.len() > 32
            || self.max_steps == 0
            || self.max_episode_ms == 0
            || self.max_output_bytes == 0
        {
            return Err(invalid("/gym", "finite gym limits and metrics required"));
        }
        factory(&self.model_interface, "/gym/model_interface")?;
        for (key, metric) in &self.metrics {
            name(key, "/gym/metrics")?;
            metric.validate()?;
        }
        if matches!(self.origin, GymOrigin::Generated | GymOrigin::Community)
            && (self.execution_trust != ExecutionTrust::UntrustedSubprocess
                || self.approval.is_none()
                || self.image.is_none())
        {
            return Err(invalid(
                "/gym",
                "generated/community code needs explicit approved isolated image",
            ));
        }
        Ok(())
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Threshold {
    pub floor: Option<f64>,
    pub ceiling: Option<f64>,
    pub max_regression: Option<f64>,
    pub min_gain: Option<f64>,
}
impl Threshold {
    fn any(&self) -> bool {
        self.floor.is_some()
            || self.ceiling.is_some()
            || self.max_regression.is_some()
            || self.min_gain.is_some()
    }
    pub fn validate(&self, metric: &MetricSchema) -> Result<()> {
        if [self.floor, self.ceiling, self.max_regression, self.min_gain]
            .into_iter()
            .flatten()
            .any(|x| !x.is_finite())
            || self.floor.zip(self.ceiling).is_some_and(|(a, b)| a > b)
            || self.max_regression.is_some_and(|x| x < 0.0)
            || self.min_gain.is_some_and(|x| x <= 0.0)
        {
            return Err(invalid(
                "/threshold",
                "finite coherent thresholds/positive primary gain required",
            ));
        }
        for x in [self.floor, self.ceiling].into_iter().flatten() {
            if metric.lower.is_some_and(|v| x < v) || metric.upper.is_some_and(|v| x > v) {
                return Err(invalid("/threshold", "threshold outside metric range"));
            }
        }
        Ok(())
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Statistic {
    Exact,
    Mean,
    Median,
    P95,
    P99,
    Maximum,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Comparator {
    Original,
    NoOp,
    BestKnown,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MeasurementScope {
    ForwardLatency,
    EndToEndLatency,
    ProcessRss,
    DeviceAllocatorPeak,
    DevicePeak,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Device {
    Cpu,
    Cuda,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Synchronization {
    CpuMonotonic,
    CudaSynchronize,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheMode {
    Cold,
    Warm,
    Disabled,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MeasurementOrder {
    PairedInterleaved,
    RandomizedInterleaved,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeasurementProtocol {
    pub scope: MeasurementScope,
    pub statistic: Statistic,
    pub warmup: u32,
    pub repetitions: u32,
    pub order: MeasurementOrder,
    pub synchronization: Synchronization,
    pub timer: FactoryRef,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MetricBinding {
    pub metric: String,
    pub required: bool,
    pub priority: f64,
    pub comparator: Comparator,
    pub threshold: Threshold,
    pub statistic: Statistic,
    pub measurement: Option<MeasurementProtocol>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Schedule {
    pub development_episodes: u32,
    pub selection_episodes: u32,
    pub holdout_episodes: u32,
    pub every_n_candidates: u32,
    pub final_required: bool,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PortfolioBinding {
    pub id: ContentDigest,
    pub objective: ContentDigest,
    pub gym: ContentDigest,
    pub validation: ContentDigest,
    pub roles: BTreeSet<GymRole>,
    pub required: bool,
    pub metrics: Vec<MetricBinding>,
    pub splits: GymSplits,
    pub schedule: Schedule,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AssuranceProfile {
    StandardFiveGym,
    Exploratory,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Isolation {
    TrustedFixtureBroker,
    SeparateProcess,
    SeparateAccount,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClaimScope {
    FixtureOnly,
    LimitedTaskScope,
    HeldOutGeneralization,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IndependentPolicy {
    pub scope: ClaimScope,
    pub evaluator: FactoryRef,
    pub administration: ContentDigest,
    pub isolation: Isolation,
    pub frozen_panel_protocol: ContentDigest,
    pub close_revision_after_feedback: bool,
    pub require_group_disjoint: bool,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GymPortfolio {
    pub schema_version: u32,
    pub profile: AssuranceProfile,
    pub label: String,
    pub gyms: BTreeMap<ContentDigest, GymManifest>,
    pub bindings: Vec<PortfolioBinding>,
    pub coverage: ContentDigest,
    pub curriculum: ContentDigest,
    pub deduplicate_shared_tasks: bool,
    pub independent: IndependentPolicy,
}
impl GymPortfolio {
    pub fn digest(&self) -> Result<ContentDigest> {
        self.validate()?;
        hash(self)
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Estimator {
    DeterministicExact,
    PairedGroupBootstrap,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StatisticsPolicy {
    pub estimator: Estimator,
    pub minimum_groups: u32,
    pub repeats: u32,
    pub confidence: f64,
    pub multiplicity_family: ContentDigest,
    pub maximum_comparisons: u32,
    pub maximum_finalists: u32,
    pub stopping_rule: ContentDigest,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObjectiveKind {
    CapabilityGain,
    EfficiencyGain,
    CausalClaim,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Objective {
    pub id: ContentDigest,
    pub kind: ObjectiveKind,
    pub description: String,
    pub primary_binding: ContentDigest,
    pub primary_metric: String,
    pub primary_comparator: Comparator,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Intervention {
    pub scope: ContentDigest,
    pub transformation_space: ContentDigest,
    pub method_protocol: ContentDigest,
    pub allowed_engines: Vec<FactoryRef>,
    pub allow_graph_change: bool,
    pub allow_buffer_write: bool,
    pub policy_checks: ContentDigest,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BaselineKind {
    Original,
    NoOp,
    BestKnown,
    #[serde(rename = "lora")]
    LoRa,
    Sft,
    Distillation,
    #[serde(rename = "rl")]
    Reinforce,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BaselineArm {
    pub kind: BaselineKind,
    pub required: bool,
    pub enabled: bool,
    pub input: ModelIdentity,
    pub backend: FactoryRef,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ControlPolicy {
    pub original_required: bool,
    pub noop_required: bool,
    pub best_known_required: bool,
    pub matched_tasks_and_seeds: bool,
    pub noop_apply_export_reload: bool,
    pub noop_exact_parameters_and_parity: bool,
    pub positive_gain_required: bool,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LengthUnit {
    Tokens,
    TensorElements,
    Bytes,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeProfile {
    pub schema_version: u32,
    pub device: Device,
    pub physical_device: ContentDigest,
    pub backend: FactoryRef,
    pub precision: DType,
    pub driver: ContentDigest,
    pub dependencies: ContentDigest,
    pub code_revision: ContentDigest,
    pub deterministic: bool,
    pub batch_size: u32,
    pub input_length: u32,
    pub max_output_length: u32,
    pub length_unit: LengthUnit,
    pub concurrency: u32,
    pub cache: CacheMode,
    pub synchronization: Synchronization,
    pub state_reset_protocol: ContentDigest,
    pub environment: ContentDigest,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionLimits {
    pub wall_ms: u64,
    pub cpu_ms: u64,
    pub gpu_ms: u64,
    pub candidate_attempts: u32,
    pub model_evaluations: u64,
    pub reasoning_tokens: u64,
    pub teacher_tokens: u64,
    pub external_spend_microusd: u64,
    pub host_memory_bytes: u64,
    pub device_memory_bytes: u64,
    pub storage_bytes: u64,
    pub concurrent_workers: u32,
    pub retries_per_operation: u32,
}
impl Validate for ExecutionLimits {
    fn validate(&self) -> Result<()> {
        if [
            self.wall_ms,
            self.cpu_ms,
            self.model_evaluations,
            self.host_memory_bytes,
            self.storage_bytes,
        ]
        .contains(&0)
            || [
                self.wall_ms,
                self.cpu_ms,
                self.gpu_ms,
                self.model_evaluations,
                self.reasoning_tokens,
                self.teacher_tokens,
                self.external_spend_microusd,
                self.host_memory_bytes,
                self.device_memory_bytes,
                self.storage_bytes,
            ]
            .contains(&u64::MAX)
            || self.wall_ms > 604_800_000
            || self.candidate_attempts == 0
            || self.concurrent_workers == 0
            || self.concurrent_workers > 1024
            || self.retries_per_operation > 100
        {
            return Err(invalid(
                "/limits",
                "finite positive core limits and bounded retries/workers required",
            ));
        }
        Ok(())
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceKind {
    Knowledge,
    Teacher,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuthorizedSource {
    pub kind: SourceKind,
    pub artifact: ContentDigest,
    pub destination: Option<String>,
    pub payloads: BTreeSet<DataClass>,
    pub access: crate::schemas::AccessClass,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StudyContract {
    pub schema_version: u32,
    pub contract_key: ContentDigest,
    pub study_family: ContentDigest,
    pub revision: u32,
    pub parent_contract: Option<ContentDigest>,
    pub approval_receipt: ContentDigest,
    pub source: ModelIdentity,
    pub best_known: ModelIdentity,
    pub objective: Objective,
    pub intervention: Intervention,
    pub permissions: Permissions,
    pub sources: Vec<AuthorizedSource>,
    pub portfolio: GymPortfolio,
    pub portfolio_digest: ContentDigest,
    pub baselines: Vec<BaselineArm>,
    pub controls: ControlPolicy,
    pub seeds: Vec<u64>,
    pub statistics: StatisticsPolicy,
    pub runtime: RuntimeProfile,
    pub limits: ExecutionLimits,
    pub promotion_protocol: ContentDigest,
    pub export_parity_protocol: ContentDigest,
}
impl StudyContract {
    pub fn digest(&self) -> Result<ContentDigest> {
        self.validate()?;
        hash(self)
    }
}

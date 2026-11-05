//! Untrusted wire records become usable only through `Validated::parse`.
//! Validation is not authorization: runtime execution also requires a scoped lease.

use crate::canonical::{CanonicalDocument, ContentDigest, Limits};
use crate::error::{child_path, CoreError, ErrorCode, Result};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::ops::Deref;

pub trait Validate {
    fn validate(&self) -> Result<()>;
}

#[derive(Debug, Clone)]
pub struct Validated<T>(T);

impl<T: Validate + DeserializeOwned> Validated<T> {
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        let canonical = CanonicalDocument::parse(bytes, Limits::default())?;
        let value: T = serde_json::from_slice(canonical.bytes()).map_err(|error| {
            let code = if error.to_string().starts_with("unknown field") {
                ErrorCode::UnknownField
            } else {
                ErrorCode::TypeMismatch
            };
            CoreError::new(code, "", error.to_string())
        })?;
        value.validate()?;
        Ok(Self(value))
    }
}

impl<T: Validate> Validated<T> {
    pub fn new(value: T) -> Result<Self> {
        value.validate()?;
        Ok(Self(value))
    }
}

impl<T> Deref for Validated<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.0
    }
}

fn invalid(path: &str, message: &str) -> CoreError {
    CoreError::new(ErrorCode::InvalidValue, path, message)
}

fn version(version: u32) -> Result<()> {
    if version != 1 {
        return Err(CoreError::new(
            ErrorCode::UnsupportedVersion,
            "/schema_version",
            "only schema version 1 is supported",
        ));
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AccessClass {
    Public,
    Development,
    Selection,
    Holdout,
    Confidential,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DataClass {
    Weights,
    Activations,
    Tasks,
    Labels,
    Feedback,
    ToolTraces,
    Summaries,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolKind {
    Inspect,
    Capture,
    Attribution,
    DirectEdit,
    Training,
    ExternalRequest,
    GeneratedGym,
    Download,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Permissions {
    pub schema_version: u32,
    pub tools: BTreeSet<ToolKind>,
    pub external_destinations: BTreeMap<String, BTreeSet<DataClass>>,
    pub allow_gradient_inspection: bool,
    pub allow_parameter_training: bool,
    pub require_fully_gradient_free: bool,
    pub allow_generated_gym: bool,
    pub allow_downloads: bool,
}

impl Validate for Permissions {
    fn validate(&self) -> Result<()> {
        version(self.schema_version)?;
        if self.require_fully_gradient_free
            && (self.allow_gradient_inspection || self.allow_parameter_training)
        {
            return Err(invalid(
                "/require_fully_gradient_free",
                "gradient-free policy contradicts gradient permissions",
            ));
        }
        for (tool, allowed, path) in [
            (
                ToolKind::Attribution,
                self.allow_gradient_inspection,
                "/allow_gradient_inspection",
            ),
            (
                ToolKind::Training,
                self.allow_parameter_training,
                "/allow_parameter_training",
            ),
            (
                ToolKind::GeneratedGym,
                self.allow_generated_gym,
                "/allow_generated_gym",
            ),
            (ToolKind::Download, self.allow_downloads, "/allow_downloads"),
        ] {
            if self.tools.contains(&tool) && !allowed {
                return Err(invalid(path, "tool grant contradicts permission flag"));
            }
        }
        if !self.external_destinations.is_empty()
            && !self.tools.contains(&ToolKind::ExternalRequest)
        {
            return Err(invalid(
                "/external_destinations",
                "destinations require an external-request tool grant",
            ));
        }
        for (destination, data) in &self.external_destinations {
            if destination.trim().is_empty() || data.is_empty() {
                return Err(invalid(
                    "/external_destinations",
                    "destination and allowed data classes must be explicit",
                ));
            }
        }
        Ok(())
    }
}

impl Permissions {
    pub fn check_tool(&self, tool: ToolKind) -> Result<()> {
        self.validate()?;
        if !self.tools.contains(&tool) {
            return Err(CoreError::new(
                ErrorCode::PermissionDenied,
                "/tools",
                "tool was not granted",
            ));
        }
        Ok(())
    }

    pub fn check_external(&self, destination: &str, classes: &BTreeSet<DataClass>) -> Result<()> {
        self.check_tool(ToolKind::ExternalRequest)?;
        if classes.is_empty()
            || !self
                .external_destinations
                .get(destination)
                .is_some_and(|allowed| classes.is_subset(allowed))
        {
            return Err(CoreError::new(
                ErrorCode::PermissionDenied,
                "/external_destinations",
                "destination or payload class was not granted",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DType {
    F32,
    F64,
    F16,
    Bf16,
    I8,
    U8,
    I32,
    I64,
}

impl DType {
    pub fn byte_width(self) -> u64 {
        match self {
            Self::F64 | Self::I64 => 8,
            Self::F32 | Self::I32 => 4,
            Self::F16 | Self::Bf16 => 2,
            Self::I8 | Self::U8 => 1,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FactoryRef {
    pub id: String,
    pub version: String,
    pub config: ContentDigest,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TensorSpec {
    pub shape: Vec<u64>,
    pub dtype: DType,
    pub digest: ContentDigest,
    pub storage_group: String,
}

impl TensorSpec {
    pub fn byte_len(&self) -> Result<u64> {
        if self.shape.len() > 32 || self.storage_group.trim().is_empty() {
            return Err(invalid("", "tensor rank or storage group is invalid"));
        }
        self.shape
            .iter()
            .try_fold(self.dtype.byte_width(), |bytes, dimension| {
                bytes
                    .checked_mul(*dimension)
                    .ok_or_else(|| invalid("/shape", "tensor byte count overflow"))
            })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TensorFormat {
    DenseSafetensors,
    QuantizedSafetensors,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelManifest {
    pub schema_version: u32,
    pub architecture: FactoryRef,
    pub parameters: BTreeMap<String, TensorSpec>,
    pub buffers: BTreeMap<String, TensorSpec>,
    pub parameter_digest: ContentDigest,
    pub format: TensorFormat,
    pub packing_policy: Option<FactoryRef>,
    pub assets: BTreeMap<String, ContentDigest>,
    pub usage_rights: String,
}

impl Validate for ModelManifest {
    fn validate(&self) -> Result<()> {
        version(self.schema_version)?;
        if self.architecture.id.trim().is_empty() || self.architecture.version.trim().is_empty() {
            return Err(invalid(
                "/architecture",
                "factory identity/version is required",
            ));
        }
        if self.parameters.is_empty() || self.usage_rights.trim().is_empty() {
            return Err(invalid(
                "",
                "parameters and supplied usage-rights statement are required",
            ));
        }
        if (self.format == TensorFormat::QuantizedSafetensors) != self.packing_policy.is_some() {
            return Err(invalid(
                "/packing_policy",
                "packing policy must match tensor format",
            ));
        }
        let mut groups: BTreeMap<&str, &TensorSpec> = BTreeMap::new();
        for (kind, tensors) in [("parameters", &self.parameters), ("buffers", &self.buffers)] {
            for (name, tensor) in tensors {
                let path = child_path(&format!("/{kind}"), name);
                if name.is_empty() {
                    return Err(invalid(&path, "tensor name cannot be empty"));
                }
                tensor.byte_len().map_err(|mut error| {
                    error.path = format!("{path}{}", error.path);
                    error
                })?;
                if kind == "buffers" && self.parameters.contains_key(name) {
                    return Err(invalid(&path, "buffer and parameter names collide"));
                }
                if let Some(other) = groups.insert(&tensor.storage_group, tensor) {
                    if other.shape != tensor.shape
                        || other.dtype != tensor.dtype
                        || other.digest != tensor.digest
                    {
                        return Err(invalid(
                            &path,
                            "declared shared storage must have identical shape, dtype, and content",
                        ));
                    }
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TargetUpdate {
    DirectTensorEdit,
    GradientTraining,
    NoChange,
    ArchitectureTransformation,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProposalOrigin {
    Scripted,
    ReasoningModel,
    LearnedProposer,
    ExternalUnknown,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Adaptation {
    None,
    ExternalMemory,
    AdaptiveMemory,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProductionHistory {
    DirectOnly,
    Trained,
    Hybrid,
    ExternalUnverified,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MethodRecord {
    pub schema_version: u32,
    pub target_update: TargetUpdate,
    pub proposal_origin: ProposalOrigin,
    pub adaptation: Adaptation,
    pub production_history: ProductionHistory,
    pub inspection_uses_backprop: bool,
    pub parameter_update_uses_backprop: bool,
    pub gradient_derived_update_values: bool,
    pub alignment_fitted: bool,
    pub optimizer_steps: u64,
    pub changed_parameters: u64,
}

impl Validate for MethodRecord {
    fn validate(&self) -> Result<()> {
        version(self.schema_version)?;
        let fitted = self.parameter_update_uses_backprop
            || self.gradient_derived_update_values
            || self.alignment_fitted
            || self.optimizer_steps > 0
            || self.target_update == TargetUpdate::GradientTraining;
        if self.production_history == ProductionHistory::DirectOnly && fitted {
            return Err(invalid(
                "/production_history",
                "candidate fitting cannot be labeled direct-only",
            ));
        }
        if self.target_update == TargetUpdate::NoChange && self.changed_parameters != 0 {
            return Err(invalid(
                "/changed_parameters",
                "no-change control must preserve parameters",
            ));
        }
        if self.proposal_origin == ProposalOrigin::ExternalUnknown
            && self.production_history != ProductionHistory::ExternalUnverified
        {
            return Err(invalid(
                "/production_history",
                "unknown external origin cannot establish direct/trained history",
            ));
        }
        Ok(())
    }
}

impl MethodRecord {
    pub fn require_fully_gradient_free(&self) -> Result<()> {
        self.validate()?;
        if self.inspection_uses_backprop
            || self.production_history != ProductionHistory::DirectOnly
            || self.adaptation == Adaptation::AdaptiveMemory
        {
            return Err(CoreError::new(
                ErrorCode::PermissionDenied,
                "/inspection_uses_backprop",
                "pipeline does not qualify as fully gradient-free",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceOrigin {
    SuppliedFact,
    ModelClaim,
    MeasuredResult,
    Assumption,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SupportGrade {
    Untested,
    Suggestive,
    InterventionSupported,
    IndependentlyConfirmedUnderProtocol,
    Contradicted,
    Inconclusive,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceLabel {
    pub schema_version: u32,
    pub origin: EvidenceOrigin,
    pub support: SupportGrade,
    pub provenance: String,
    pub evidence: Vec<ContentDigest>,
    pub access: AccessClass,
}

impl Validate for EvidenceLabel {
    fn validate(&self) -> Result<()> {
        version(self.schema_version)?;
        if self.provenance.trim().is_empty() {
            return Err(invalid("/provenance", "source identity is required"));
        }
        let causal = matches!(
            self.support,
            SupportGrade::InterventionSupported | SupportGrade::IndependentlyConfirmedUnderProtocol
        );
        if (self.origin == EvidenceOrigin::MeasuredResult || causal) && self.evidence.is_empty() {
            return Err(invalid(
                "/evidence",
                "measured or causal support requires evidence references",
            ));
        }
        if causal && self.origin != EvidenceOrigin::MeasuredResult {
            return Err(invalid(
                "/support",
                "a claim cannot become causal evidence without a measured record",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IsolationRequirement {
    TrustedBuiltin,
    RestrictedSandbox,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CapabilityDescriptor {
    pub schema_version: u32,
    pub id: String,
    pub version: String,
    pub model_factories: BTreeSet<String>,
    pub operations: BTreeSet<String>,
    pub required_tools: BTreeSet<ToolKind>,
    pub isolation: IsolationRequirement,
    pub uses_backprop: bool,
    pub fits_parameters: bool,
    pub max_input_bytes: u64,
    pub max_output_bytes: u64,
    pub max_wall_millis: u64,
}

impl Validate for CapabilityDescriptor {
    fn validate(&self) -> Result<()> {
        version(self.schema_version)?;
        if self.id.trim().is_empty()
            || self.version.trim().is_empty()
            || self.operations.is_empty()
            || self
                .operations
                .iter()
                .any(|operation| operation.trim().is_empty())
            || self.model_factories.is_empty()
            || self
                .model_factories
                .iter()
                .any(|model| model.trim().is_empty())
        {
            return Err(invalid(
                "",
                "explicit engine, model, and operation identities are required",
            ));
        }
        if self.max_input_bytes == 0 || self.max_output_bytes == 0 || self.max_wall_millis == 0 {
            return Err(invalid("", "finite positive execution bounds are required"));
        }
        if self.fits_parameters && !self.required_tools.contains(&ToolKind::Training) {
            return Err(invalid(
                "/required_tools",
                "fitting requires a training permission",
            ));
        }
        if self.uses_backprop
            && !self.required_tools.contains(&ToolKind::Training)
            && !self.required_tools.contains(&ToolKind::Attribution)
        {
            return Err(invalid(
                "/required_tools",
                "backprop requires attribution or training permission",
            ));
        }
        if self.required_tools.contains(&ToolKind::GeneratedGym)
            && self.isolation != IsolationRequirement::RestrictedSandbox
        {
            return Err(invalid(
                "/isolation",
                "generated gyms require restricted sandbox execution",
            ));
        }
        Ok(())
    }
}

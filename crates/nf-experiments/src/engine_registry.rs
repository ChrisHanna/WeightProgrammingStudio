//! Versioned research-engine protocol and operator-controlled registry (T046).
//!
//! Engines implement `describe`, `validate`, `estimate`, `execute`, and the
//! registry provides `collect(operation_id)`. Agent text, model metadata, and
//! untrusted payloads cannot self-register an engine or escalate permissions.

use nf_core::canonical::{CanonicalDocument, ContentDigest, Limits};
use nf_core::error::{CoreError, ErrorCode, Result};
use nf_core::schemas::{
    Adaptation, CapabilityDescriptor, FactoryRef, IsolationRequirement, MethodRecord, Permissions,
    ProductionHistory, ProposalOrigin, TargetUpdate, ToolKind, Validate, Validated,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

fn err(code: ErrorCode, path: &str, message: impl Into<String>) -> CoreError {
    CoreError::new(code, path, message)
}

fn canonical_bytes<T: Serialize>(value: &T) -> Result<Vec<u8>> {
    let raw = serde_json::to_vec(value)
        .map_err(|e| err(ErrorCode::InvalidValue, "/engine_registry", e.to_string()))?;
    Ok(CanonicalDocument::parse(&raw, Limits::default())?
        .bytes()
        .to_vec())
}

pub fn descriptor_digest(description: &EngineDescription) -> Result<ContentDigest> {
    Ok(ContentDigest::of_bytes(&canonical_bytes(description)?))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EngineDescription {
    pub schema_version: u32,
    pub capability: CapabilityDescriptor,
    pub input_schema: ContentDigest,
    pub output_schema: ContentDigest,
    pub deterministic: bool,
    pub method_template: MethodRecord,
    pub runtime_dependencies: BTreeSet<String>,
}

impl Validate for EngineDescription {
    fn validate(&self) -> Result<()> {
        if self.schema_version != 1 {
            return Err(err(
                ErrorCode::UnsupportedVersion,
                "/schema_version",
                "only engine description schema v1 is supported",
            ));
        }
        self.capability.validate()?;
        self.method_template.validate()?;
        if self
            .runtime_dependencies
            .iter()
            .any(|dep| dep.trim().is_empty())
        {
            return Err(err(
                ErrorCode::InvalidValue,
                "/runtime_dependencies",
                "runtime dependency identifiers cannot be empty",
            ));
        }
        let m = &self.method_template;
        let fits = m.parameter_update_uses_backprop
            || m.gradient_derived_update_values
            || m.alignment_fitted
            || m.optimizer_steps > 0
            || m.target_update == TargetUpdate::GradientTraining
            || matches!(
                m.production_history,
                ProductionHistory::Trained | ProductionHistory::Hybrid
            );
        if self.capability.fits_parameters != fits {
            return Err(err(
                ErrorCode::InvalidValue,
                "/method_template",
                "capability fits_parameters must match method_template training/fitting flags",
            ));
        }
        let uses_grad = m.inspection_uses_backprop
            || m.parameter_update_uses_backprop
            || m.gradient_derived_update_values;
        if self.capability.uses_backprop != uses_grad {
            return Err(err(
                ErrorCode::InvalidValue,
                "/method_template",
                "capability uses_backprop must match method_template gradient flags",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EngineEstimate {
    pub cpu_ms: u64,
    pub wall_ms: u64,
    pub host_memory_bytes: u64,
    pub storage_bytes: u64,
    pub evaluations: u64,
    pub backward_calls: u64,
    pub optimizer_steps: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EngineRequest {
    pub schema_version: u32,
    pub operation_id: ContentDigest,
    pub engine_id: String,
    pub engine_version: String,
    pub operation: String,
    pub model_factory: String,
    pub payload: Value,
}

#[derive(Debug, Clone)]
pub struct ExecutionCapabilities {
    pub contract_permissions: Validated<Permissions>,
    pub allowed_engines: Vec<FactoryRef>,
    pub sandbox_qualified: bool,
    pub max_wall_ms: u64,
    pub max_host_memory_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EngineReceipt {
    pub schema_version: u32,
    pub operation_id: ContentDigest,
    pub engine_id: String,
    pub engine_version: String,
    pub config_digest: ContentDigest,
    pub operation: String,
    pub model_factory: String,
    pub request_digest: ContentDigest,
    pub method: MethodRecord,
    pub estimate: EngineEstimate,
    pub output: Value,
    pub deterministic: bool,
}

pub trait ResearchEngine: Send + Sync {
    fn describe(&self) -> &Validated<EngineDescription>;
    fn validate(&self, request: &EngineRequest) -> Result<()>;
    fn estimate(&self, request: &EngineRequest) -> Result<EngineEstimate>;
    fn execute(
        &self,
        request: &EngineRequest,
        capabilities: &ExecutionCapabilities,
    ) -> Result<Value>;
}

/// Proof of explicit operator installation; never deserializable from JSON or agent text.
pub struct OperatorRegistrationToken(());

impl OperatorRegistrationToken {
    pub fn new_operator() -> Self {
        Self(())
    }
}

#[derive(Default)]
pub struct EngineRegistry {
    engines: BTreeMap<(String, String), (ContentDigest, Arc<dyn ResearchEngine>)>,
    receipts: BTreeMap<ContentDigest, EngineReceipt>,
}

impl EngineRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Reject any attempt by agent text, prompts, or untrusted JSON to self-register an engine.
    pub fn reject_agent_text_registration(untrusted_input: &str) -> Result<()> {
        let _ = untrusted_input;
        Err(err(
            ErrorCode::PermissionDenied,
            "/engine_registry/self_register",
            "runtime self-registration from agent text or untrusted payload is forbidden",
        ))
    }

    pub fn register(
        &mut self,
        _operator: &OperatorRegistrationToken,
        engine: Arc<dyn ResearchEngine>,
    ) -> Result<ContentDigest> {
        let desc = engine.describe();
        desc.validate()?;
        let digest = descriptor_digest(desc)?;
        let key = (desc.capability.id.clone(), desc.capability.version.clone());
        if self.engines.contains_key(&key) {
            return Err(err(
                ErrorCode::DuplicateKey,
                "/engine_registry",
                "engine id and version already registered",
            ));
        }
        self.engines.insert(key, (digest.clone(), engine));
        Ok(digest)
    }

    pub fn describe(&self, id: &str, version: &str) -> Result<&Validated<EngineDescription>> {
        if let Some((_, engine)) = self.engines.get(&(id.to_owned(), version.to_owned())) {
            return Ok(engine.describe());
        }
        if self.engines.keys().any(|(eid, _)| eid == id) {
            return Err(err(
                ErrorCode::UnsupportedVersion,
                "/engine_version",
                "requested engine version is not registered",
            ));
        }
        Err(err(
            ErrorCode::UnsupportedCapability,
            "/engine_id",
            "requested research engine is not registered",
        ))
    }

    fn resolve(
        &self,
        request: &EngineRequest,
        capabilities: &ExecutionCapabilities,
    ) -> Result<(ContentDigest, &Arc<dyn ResearchEngine>)> {
        if request.schema_version != 1 {
            return Err(err(
                ErrorCode::UnsupportedVersion,
                "/schema_version",
                "only engine request schema v1 is supported",
            ));
        }
        if let Some(obj) = request.payload.as_object() {
            for forbidden in [
                "register_engine",
                "self_register",
                "override_permissions",
                "grant_tools",
            ] {
                if obj.contains_key(forbidden) {
                    return Err(err(
                        ErrorCode::PermissionDenied,
                        "/payload",
                        "agent payload cannot self-register engines or alter permissions",
                    ));
                }
            }
        }
        let (config_digest, engine) = self
            .engines
            .get(&(request.engine_id.clone(), request.engine_version.clone()))
            .ok_or_else(|| {
                if self.engines.keys().any(|(id, _)| id == &request.engine_id) {
                    err(
                        ErrorCode::UnsupportedVersion,
                        "/engine_version",
                        "incompatible engine version rejected before dispatch",
                    )
                } else {
                    err(
                        ErrorCode::UnsupportedCapability,
                        "/engine_id",
                        "unregistered engine rejected before dispatch",
                    )
                }
            })?;
        let desc = engine.describe();
        let payload_bytes = canonical_bytes(&request.payload)?;
        if payload_bytes.len() as u64 > desc.capability.max_input_bytes {
            return Err(err(
                ErrorCode::LimitExceeded,
                "/payload",
                "engine request payload exceeds max_input_bytes",
            ));
        }
        if !desc
            .capability
            .model_factories
            .contains(&request.model_factory)
        {
            return Err(err(
                ErrorCode::UnsupportedCapability,
                "/model_factory",
                "model architecture is not supported by this engine",
            ));
        }
        if !desc.capability.operations.contains(&request.operation) {
            return Err(err(
                ErrorCode::UnsupportedOperation,
                "/operation",
                "operation is not supported by this engine",
            ));
        }
        let allowed = capabilities
            .allowed_engines
            .iter()
            .find(|entry| entry.id == request.engine_id && entry.version == request.engine_version);
        match allowed {
            None => {
                if capabilities
                    .allowed_engines
                    .iter()
                    .any(|entry| entry.id == request.engine_id)
                {
                    return Err(err(
                        ErrorCode::UnsupportedVersion,
                        "/engine_version",
                        "engine version is not authorized by contract allowed_engines",
                    ));
                }
                return Err(err(
                    ErrorCode::PermissionDenied,
                    "/engine_id",
                    "engine is not authorized by contract allowed_engines",
                ));
            }
            Some(entry) if entry.config != *config_digest => {
                return Err(err(
                    ErrorCode::SourceMismatch,
                    "/engine_config",
                    "engine descriptor digest differs from frozen contract binding",
                ));
            }
            Some(_) => {}
        }
        for tool in &desc.capability.required_tools {
            capabilities.contract_permissions.check_tool(*tool)?;
        }
        if capabilities
            .contract_permissions
            .require_fully_gradient_free
        {
            desc.method_template.require_fully_gradient_free()?;
        }
        if !capabilities.contract_permissions.allow_parameter_training
            && desc.capability.fits_parameters
        {
            return Err(err(
                ErrorCode::PermissionDenied,
                "/allow_parameter_training",
                "parameter-fitting engine is forbidden when training is disabled",
            ));
        }
        if !capabilities.contract_permissions.allow_gradient_inspection
            && desc.method_template.inspection_uses_backprop
        {
            return Err(err(
                ErrorCode::PermissionDenied,
                "/allow_gradient_inspection",
                "gradient-assisted inspection engine is forbidden by contract",
            ));
        }
        if desc.capability.isolation == IsolationRequirement::RestrictedSandbox
            && !capabilities.sandbox_qualified
        {
            return Err(err(
                ErrorCode::PermissionDenied,
                "/isolation",
                "sandbox-isolated engine requires a qualified sandbox runtime",
            ));
        }
        engine.validate(request)?;
        Ok((config_digest.clone(), engine))
    }

    pub fn validate(
        &self,
        request: &EngineRequest,
        capabilities: &ExecutionCapabilities,
    ) -> Result<()> {
        self.resolve(request, capabilities).map(|_| ())
    }

    pub fn estimate(
        &self,
        request: &EngineRequest,
        capabilities: &ExecutionCapabilities,
    ) -> Result<EngineEstimate> {
        let (_, engine) = self.resolve(request, capabilities)?;
        let desc = engine.describe();
        let estimate = engine.estimate(request)?;
        if estimate.wall_ms == 0
            || estimate.wall_ms > desc.capability.max_wall_millis
            || estimate.wall_ms > capabilities.max_wall_ms
            || estimate.host_memory_bytes == 0
            || estimate.host_memory_bytes > capabilities.max_host_memory_bytes
        {
            return Err(err(
                ErrorCode::LimitExceeded,
                "/estimate",
                "engine estimate exceeds engine or contract resource ceiling",
            ));
        }
        if !desc.capability.uses_backprop && estimate.backward_calls > 0 {
            return Err(err(
                ErrorCode::PermissionDenied,
                "/estimate/backward_calls",
                "gradient-free engine cannot estimate nonzero backward calls",
            ));
        }
        if !desc.capability.fits_parameters && estimate.optimizer_steps > 0 {
            return Err(err(
                ErrorCode::PermissionDenied,
                "/estimate/optimizer_steps",
                "non-fitting engine cannot estimate optimizer steps",
            ));
        }
        Ok(estimate)
    }

    pub fn execute(
        &mut self,
        request: &EngineRequest,
        capabilities: &ExecutionCapabilities,
    ) -> Result<EngineReceipt> {
        let request_digest = ContentDigest::of_bytes(&canonical_bytes(request)?);
        if let Some(existing) = self.receipts.get(&request.operation_id) {
            if existing.request_digest != request_digest {
                return Err(err(
                    ErrorCode::IdempotencyConflict,
                    "/operation_id",
                    "operation_id reused with different engine request",
                ));
            }
            return Ok(existing.clone());
        }
        let estimate = self.estimate(request, capabilities)?;
        let (config_digest, engine) = self.resolve(request, capabilities)?;
        let desc = engine.describe().clone();
        let output = engine.execute(request, capabilities)?;
        let output_bytes = canonical_bytes(&output)?;
        if output_bytes.len() as u64 > desc.capability.max_output_bytes {
            return Err(err(
                ErrorCode::LimitExceeded,
                "/output",
                "engine output exceeds max_output_bytes",
            ));
        }
        let receipt = EngineReceipt {
            schema_version: 1,
            operation_id: request.operation_id.clone(),
            engine_id: request.engine_id.clone(),
            engine_version: request.engine_version.clone(),
            config_digest,
            operation: request.operation.clone(),
            model_factory: request.model_factory.clone(),
            request_digest,
            method: desc.method_template.clone(),
            estimate,
            output,
            deterministic: desc.deterministic,
        };
        self.receipts
            .insert(request.operation_id.clone(), receipt.clone());
        Ok(receipt)
    }

    pub fn collect(&self, operation_id: &ContentDigest) -> Result<&EngineReceipt> {
        self.receipts.get(operation_id).ok_or_else(|| {
            err(
                ErrorCode::InvalidValue,
                "/operation_id",
                "unknown engine operation_id",
            )
        })
    }
}

/// Built-in trusted engine implementations for multi-method weight programming.
pub struct StandardEngine {
    description: Validated<EngineDescription>,
    default_estimate: EngineEstimate,
}

impl StandardEngine {
    pub fn direct_subspace_repair() -> Result<Self> {
        let description = Validated::new(EngineDescription {
            schema_version: 1,
            capability: CapabilityDescriptor {
                schema_version: 1,
                id: "nf.engine.direct_subspace_repair".into(),
                version: "1.0.0".into(),
                model_factories: BTreeSet::from([
                    "fixture.cpu".into(),
                    "nf.arithmetic.relu".into(),
                    "nf.transformer.small_causal.v1".into(),
                    "hf.qwen2.dense".into(),
                ]),
                operations: BTreeSet::from([
                    "propose_subspace_projection".into(),
                    "propose_low_rank_update".into(),
                    "propose_scale".into(),
                ]),
                required_tools: BTreeSet::from([ToolKind::Inspect, ToolKind::DirectEdit]),
                isolation: IsolationRequirement::TrustedBuiltin,
                uses_backprop: false,
                fits_parameters: false,
                max_input_bytes: 65536,
                max_output_bytes: 65536,
                max_wall_millis: 10_000,
            },
            input_schema: ContentDigest::of_bytes(b"nf.schema.direct_subspace_repair.input.v1"),
            output_schema: ContentDigest::of_bytes(b"nf.schema.direct_subspace_repair.output.v1"),
            deterministic: true,
            method_template: MethodRecord {
                schema_version: 1,
                target_update: TargetUpdate::DirectTensorEdit,
                proposal_origin: ProposalOrigin::Scripted,
                adaptation: Adaptation::None,
                production_history: ProductionHistory::DirectOnly,
                inspection_uses_backprop: false,
                parameter_update_uses_backprop: false,
                gradient_derived_update_values: false,
                alignment_fitted: false,
                optimizer_steps: 0,
                changed_parameters: 1,
            },
            runtime_dependencies: BTreeSet::from(["nf-weight-ir".into(), "nf-surgeon".into()]),
        })?;
        Ok(Self {
            description,
            default_estimate: EngineEstimate {
                cpu_ms: 50,
                wall_ms: 100,
                host_memory_bytes: 16 << 20,
                storage_bytes: 65536,
                evaluations: 1,
                backward_calls: 0,
                optimizer_steps: 0,
            },
        })
    }

    pub fn gradient_assisted_saliency() -> Result<Self> {
        let description = Validated::new(EngineDescription {
            schema_version: 1,
            capability: CapabilityDescriptor {
                schema_version: 1,
                id: "nf.engine.saliency_locator".into(),
                version: "1.0.0".into(),
                model_factories: BTreeSet::from([
                    "nf.arithmetic.relu".into(),
                    "nf.transformer.small_causal.v1".into(),
                    "hf.qwen2.dense".into(),
                ]),
                operations: BTreeSet::from(["locate_saliency_sites".into()]),
                required_tools: BTreeSet::from([
                    ToolKind::Inspect,
                    ToolKind::Attribution,
                    ToolKind::DirectEdit,
                ]),
                isolation: IsolationRequirement::TrustedBuiltin,
                uses_backprop: true,
                fits_parameters: false,
                max_input_bytes: 65536,
                max_output_bytes: 65536,
                max_wall_millis: 15_000,
            },
            input_schema: ContentDigest::of_bytes(b"nf.schema.saliency_locator.input.v1"),
            output_schema: ContentDigest::of_bytes(b"nf.schema.saliency_locator.output.v1"),
            deterministic: true,
            method_template: MethodRecord {
                schema_version: 1,
                target_update: TargetUpdate::DirectTensorEdit,
                proposal_origin: ProposalOrigin::Scripted,
                adaptation: Adaptation::None,
                production_history: ProductionHistory::DirectOnly,
                inspection_uses_backprop: true,
                parameter_update_uses_backprop: false,
                gradient_derived_update_values: false,
                alignment_fitted: false,
                optimizer_steps: 0,
                changed_parameters: 1,
            },
            runtime_dependencies: BTreeSet::from(["torch==2.9.1+cpu".into()]),
        })?;
        Ok(Self {
            description,
            default_estimate: EngineEstimate {
                cpu_ms: 200,
                wall_ms: 500,
                host_memory_bytes: 64 << 20,
                storage_bytes: 65536,
                evaluations: 2,
                backward_calls: 1,
                optimizer_steps: 0,
            },
        })
    }

    pub fn fitted_alignment_transfer() -> Result<Self> {
        let description = Validated::new(EngineDescription {
            schema_version: 1,
            capability: CapabilityDescriptor {
                schema_version: 1,
                id: "nf.engine.fitted_alignment_transfer".into(),
                version: "1.0.0".into(),
                model_factories: BTreeSet::from([
                    "nf.transformer.small_causal.v1".into(),
                    "hf.qwen2.dense".into(),
                ]),
                operations: BTreeSet::from(["fit_linear_alignment".into()]),
                required_tools: BTreeSet::from([ToolKind::Inspect, ToolKind::Training]),
                isolation: IsolationRequirement::TrustedBuiltin,
                uses_backprop: false,
                fits_parameters: true,
                max_input_bytes: 65536,
                max_output_bytes: 65536,
                max_wall_millis: 20_000,
            },
            input_schema: ContentDigest::of_bytes(b"nf.schema.fitted_alignment.input.v1"),
            output_schema: ContentDigest::of_bytes(b"nf.schema.fitted_alignment.output.v1"),
            deterministic: true,
            method_template: MethodRecord {
                schema_version: 1,
                target_update: TargetUpdate::DirectTensorEdit,
                proposal_origin: ProposalOrigin::Scripted,
                adaptation: Adaptation::None,
                production_history: ProductionHistory::Hybrid,
                inspection_uses_backprop: false,
                parameter_update_uses_backprop: false,
                gradient_derived_update_values: false,
                alignment_fitted: true,
                optimizer_steps: 1,
                changed_parameters: 4,
            },
            runtime_dependencies: BTreeSet::from(["nf-surgeon".into()]),
        })?;
        Ok(Self {
            description,
            default_estimate: EngineEstimate {
                cpu_ms: 300,
                wall_ms: 600,
                host_memory_bytes: 64 << 20,
                storage_bytes: 131072,
                evaluations: 4,
                backward_calls: 0,
                optimizer_steps: 1,
            },
        })
    }
}

impl ResearchEngine for StandardEngine {
    fn describe(&self) -> &Validated<EngineDescription> {
        &self.description
    }

    fn validate(&self, request: &EngineRequest) -> Result<()> {
        let obj = request.payload.as_object().ok_or_else(|| {
            err(
                ErrorCode::TypeMismatch,
                "/payload",
                "engine payload must be a JSON object",
            )
        })?;
        let target_site = obj
            .get("target_site")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                err(
                    ErrorCode::InvalidValue,
                    "/payload/target_site",
                    "target_site string is required",
                )
            })?;
        if target_site.trim().is_empty() {
            return Err(err(
                ErrorCode::InvalidValue,
                "/payload/target_site",
                "target_site cannot be empty",
            ));
        }
        Ok(())
    }

    fn estimate(&self, request: &EngineRequest) -> Result<EngineEstimate> {
        self.validate(request)?;
        Ok(self.default_estimate.clone())
    }

    fn execute(
        &self,
        request: &EngineRequest,
        _capabilities: &ExecutionCapabilities,
    ) -> Result<Value> {
        self.validate(request)?;
        Ok(json!({
            "status": "completed",
            "engine_id": self.description.capability.id,
            "engine_version": self.description.capability.version,
            "operation": request.operation,
            "model_factory": request.model_factory,
            "target_site": request.payload["target_site"],
            "method_classification": self.description.method_template,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn op_id(tag: &str) -> ContentDigest {
        ContentDigest::of_bytes(tag.as_bytes())
    }

    fn base_permissions(
        allow_grad_inspect: bool,
        allow_training: bool,
        fully_grad_free: bool,
    ) -> Validated<Permissions> {
        let mut tools = BTreeSet::from([ToolKind::Inspect, ToolKind::DirectEdit]);
        if allow_grad_inspect {
            tools.insert(ToolKind::Attribution);
        }
        if allow_training {
            tools.insert(ToolKind::Training);
        }
        Validated::new(Permissions {
            schema_version: 1,
            tools,
            external_destinations: BTreeMap::new(),
            allow_gradient_inspection: allow_grad_inspect,
            allow_parameter_training: allow_training,
            require_fully_gradient_free: fully_grad_free,
            allow_generated_gym: false,
            allow_downloads: false,
        })
        .unwrap()
    }

    #[test]
    fn engine_registry_registers_executes_collects_and_enforces_idempotency() {
        let token = OperatorRegistrationToken::new_operator();
        let mut registry = EngineRegistry::new();
        let direct = Arc::new(StandardEngine::direct_subspace_repair().unwrap());
        let config_digest = registry.register(&token, direct.clone()).unwrap();
        assert_eq!(
            registry.register(&token, direct).unwrap_err().code,
            ErrorCode::DuplicateKey
        );

        let capabilities = ExecutionCapabilities {
            contract_permissions: base_permissions(false, false, true),
            allowed_engines: vec![FactoryRef {
                id: "nf.engine.direct_subspace_repair".into(),
                version: "1.0.0".into(),
                config: config_digest.clone(),
            }],
            sandbox_qualified: false,
            max_wall_ms: 30_000,
            max_host_memory_bytes: 512 << 20,
        };

        let request = EngineRequest {
            schema_version: 1,
            operation_id: op_id("op-qwen-direct-1"),
            engine_id: "nf.engine.direct_subspace_repair".into(),
            engine_version: "1.0.0".into(),
            operation: "propose_low_rank_update".into(),
            model_factory: "hf.qwen2.dense".into(),
            payload: json!({"target_site": "model.layers.0.mlp.down_proj.weight"}),
        };

        registry.validate(&request, &capabilities).unwrap();
        let estimate = registry.estimate(&request, &capabilities).unwrap();
        assert_eq!(estimate.backward_calls, 0);
        assert_eq!(estimate.optimizer_steps, 0);

        let receipt = registry.execute(&request, &capabilities).unwrap();
        assert_eq!(receipt.config_digest, config_digest);
        assert_eq!(
            receipt.method.production_history,
            ProductionHistory::DirectOnly
        );
        assert!(!receipt.method.inspection_uses_backprop);
        assert_eq!(registry.collect(&request.operation_id).unwrap(), &receipt);

        // Idempotent repeat returns identical receipt; altered payload with same operation_id fails.
        assert_eq!(registry.execute(&request, &capabilities).unwrap(), receipt);
        let mut conflicting = request.clone();
        conflicting.payload = json!({"target_site": "model.layers.1.mlp.down_proj.weight"});
        assert_eq!(
            registry
                .execute(&conflicting, &capabilities)
                .unwrap_err()
                .code,
            ErrorCode::IdempotencyConflict
        );
    }

    #[test]
    fn engine_registry_rejects_version_model_capability_and_agent_self_registration_mismatches() {
        assert_eq!(
            EngineRegistry::reject_agent_text_registration(
                r#"{"register_engine":"evil.agent.engine","version":"1.0.0"}"#
            )
            .unwrap_err()
            .code,
            ErrorCode::PermissionDenied
        );

        let token = OperatorRegistrationToken::new_operator();
        let mut registry = EngineRegistry::new();
        let direct = Arc::new(StandardEngine::direct_subspace_repair().unwrap());
        let saliency = Arc::new(StandardEngine::gradient_assisted_saliency().unwrap());
        let fitted = Arc::new(StandardEngine::fitted_alignment_transfer().unwrap());

        let direct_cfg = registry.register(&token, direct).unwrap();
        let saliency_cfg = registry.register(&token, saliency).unwrap();
        let fitted_cfg = registry.register(&token, fitted).unwrap();

        let strict_caps = ExecutionCapabilities {
            contract_permissions: base_permissions(false, false, true),
            allowed_engines: vec![
                FactoryRef {
                    id: "nf.engine.direct_subspace_repair".into(),
                    version: "1.0.0".into(),
                    config: direct_cfg.clone(),
                },
                FactoryRef {
                    id: "nf.engine.saliency_locator".into(),
                    version: "1.0.0".into(),
                    config: saliency_cfg.clone(),
                },
                FactoryRef {
                    id: "nf.engine.fitted_alignment_transfer".into(),
                    version: "1.0.0".into(),
                    config: fitted_cfg.clone(),
                },
            ],
            sandbox_qualified: false,
            max_wall_ms: 30_000,
            max_host_memory_bytes: 512 << 20,
        };

        // 1. Version mismatch
        let bad_version = EngineRequest {
            schema_version: 1,
            operation_id: op_id("bad-version"),
            engine_id: "nf.engine.direct_subspace_repair".into(),
            engine_version: "2.0.0".into(),
            operation: "propose_scale".into(),
            model_factory: "hf.qwen2.dense".into(),
            payload: json!({"target_site": "model.layers.0.mlp.down_proj.weight"}),
        };
        assert_eq!(
            registry
                .validate(&bad_version, &strict_caps)
                .unwrap_err()
                .code,
            ErrorCode::UnsupportedVersion
        );

        // 2. Incompatible model factory
        let bad_model = EngineRequest {
            schema_version: 1,
            operation_id: op_id("bad-model"),
            engine_id: "nf.engine.direct_subspace_repair".into(),
            engine_version: "1.0.0".into(),
            operation: "propose_scale".into(),
            model_factory: "unsupported.mamba.v1".into(),
            payload: json!({"target_site": "layers.0.weight"}),
        };
        assert_eq!(
            registry
                .validate(&bad_model, &strict_caps)
                .unwrap_err()
                .code,
            ErrorCode::UnsupportedCapability
        );

        // 3. Agent self-registration key inside payload
        let injected = EngineRequest {
            schema_version: 1,
            operation_id: op_id("injected"),
            engine_id: "nf.engine.direct_subspace_repair".into(),
            engine_version: "1.0.0".into(),
            operation: "propose_scale".into(),
            model_factory: "hf.qwen2.dense".into(),
            payload: json!({"target_site": "w", "register_engine": "agent.custom"}),
        };
        assert_eq!(
            registry.validate(&injected, &strict_caps).unwrap_err().code,
            ErrorCode::PermissionDenied
        );

        // 4. Saliency engine rejected under strict fully-gradient-free permissions,
        //    but accepted when gradient-assisted inspection is granted.
        let saliency_req = EngineRequest {
            schema_version: 1,
            operation_id: op_id("saliency-1"),
            engine_id: "nf.engine.saliency_locator".into(),
            engine_version: "1.0.0".into(),
            operation: "locate_saliency_sites".into(),
            model_factory: "hf.qwen2.dense".into(),
            payload: json!({"target_site": "model.layers.0.self_attn.q_proj.weight"}),
        };
        assert_eq!(
            registry
                .validate(&saliency_req, &strict_caps)
                .unwrap_err()
                .code,
            ErrorCode::PermissionDenied
        );

        let grad_inspect_caps = ExecutionCapabilities {
            contract_permissions: base_permissions(true, false, false),
            ..strict_caps.clone()
        };
        let saliency_receipt = registry.execute(&saliency_req, &grad_inspect_caps).unwrap();
        assert!(saliency_receipt.method.inspection_uses_backprop);
        assert_eq!(
            saliency_receipt.method.production_history,
            ProductionHistory::DirectOnly
        );

        // 5. Fitted alignment engine rejected when training is disabled, accepted when training is enabled.
        let fitted_req = EngineRequest {
            schema_version: 1,
            operation_id: op_id("fitted-1"),
            engine_id: "nf.engine.fitted_alignment_transfer".into(),
            engine_version: "1.0.0".into(),
            operation: "fit_linear_alignment".into(),
            model_factory: "hf.qwen2.dense".into(),
            payload: json!({"target_site": "model.layers.0.mlp.up_proj.weight"}),
        };
        assert_eq!(
            registry
                .validate(&fitted_req, &grad_inspect_caps)
                .unwrap_err()
                .code,
            ErrorCode::PermissionDenied
        );

        let training_caps = ExecutionCapabilities {
            contract_permissions: base_permissions(true, true, false),
            ..strict_caps
        };
        let fitted_receipt = registry.execute(&fitted_req, &training_caps).unwrap();
        assert_eq!(
            fitted_receipt.method.production_history,
            ProductionHistory::Hybrid
        );
        assert!(fitted_receipt.method.alignment_fitted);
    }
}

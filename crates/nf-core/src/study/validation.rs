use super::*;

impl Validate for GymPortfolio {
    fn validate(&self) -> Result<()> {
        version(self.schema_version)?;
        name(&self.label, "/portfolio/label")?;
        factory(&self.independent.evaluator, "/independent/evaluator")?;
        if self.gyms.is_empty()
            || self.gyms.len() > 64
            || self.bindings.is_empty()
            || self.bindings.len() > 64
            || !self.deduplicate_shared_tasks
            || !self.independent.close_revision_after_feedback
            || !self.independent.require_group_disjoint
        {
            return Err(invalid("/portfolio","bounded exact gyms/bindings, deduplication and closed final feedback policy required"));
        }
        if self.independent.scope == ClaimScope::HeldOutGeneralization
            && self.independent.isolation == Isolation::TrustedFixtureBroker
        {
            return Err(invalid("/independent/isolation","trusted in-process fixture discipline cannot certify adversarial holdout isolation"));
        }
        if self.independent.isolation == Isolation::TrustedFixtureBroker
            && self.independent.scope != ClaimScope::FixtureOnly
        {
            return Err(invalid(
                "/independent/scope",
                "trusted fixture broker requires fixture-only claim scope",
            ));
        }
        for (id, gym) in &self.gyms {
            gym.validate()?;
            if *id != gym.digest()? {
                return Err(CoreError::new(
                    ErrorCode::IntegrityMismatch,
                    "/portfolio/gyms",
                    "gym manifest ID differs from canonical contents",
                ));
            }
        }
        let mut ids = BTreeSet::new();
        let mut used = BTreeSet::new();
        let mut required_names = BTreeSet::new();
        let mut required_roles = BTreeSet::new();
        let mut item_roles = BTreeMap::new();
        let mut group_roles = BTreeMap::new();
        let mut item_groups = BTreeMap::new();
        let mut total_items = 0usize;
        for binding in &self.bindings {
            if !ids.insert(&binding.id) || !used.insert(&binding.gym) {
                return Err(invalid(
                    "/portfolio/bindings",
                    "duplicate binding or gym counting",
                ));
            }
            let gym = self
                .gyms
                .get(&binding.gym)
                .ok_or_else(|| invalid("/portfolio/bindings/gym", "missing exact gym manifest"))?;
            if binding.validation != gym.validation
                || binding.roles.is_empty()
                || binding.metrics.is_empty()
                || binding.metrics.len() > 32
            {
                return Err(invalid(
                    "/portfolio/bindings",
                    "validation receipt, metric and role bindings required",
                ));
            }
            if binding.required {
                required_names.insert((&gym.package, &gym.name));
                required_roles.extend(binding.roles.iter().copied());
                if !binding.schedule.final_required
                    || binding.schedule.selection_episodes == 0
                    || binding.schedule.holdout_episodes == 0
                    || binding.schedule.every_n_candidates == 0
                {
                    return Err(invalid(
                        "/schedule",
                        "required gyms need nonzero selection/final schedules",
                    ));
                }
            }
            let mut metric_names = BTreeSet::new();
            let mut hard = 0;
            let mut protected_quality = false;
            let mut safety = false;
            for metric in &binding.metrics {
                if !metric_names.insert((&metric.metric, metric.comparator)) {
                    return Err(invalid("/metrics", "duplicate metric binding"));
                }
                let schema = gym
                    .metrics
                    .get(&metric.metric)
                    .ok_or_else(|| invalid("/metrics", "undeclared metric binding"))?;
                if !metric.priority.is_finite() || metric.priority < 0.0 {
                    return Err(invalid(
                        "/metrics/priority",
                        "priority must be finite and nonnegative; it cannot bypass gates",
                    ));
                }
                metric.threshold.validate(schema)?;
                if metric.required {
                    hard += 1;
                    if !metric.threshold.any() {
                        return Err(invalid(
                            "/metrics/threshold",
                            "required metric needs an explicit hard threshold",
                        ));
                    }
                }
                if metric.required
                    && schema.domain == MetricDomain::Quality
                    && metric.threshold.max_regression.is_some()
                {
                    protected_quality = true;
                }
                if metric.required && schema.domain == MetricDomain::Safety {
                    safety = true;
                }
                if binding.required
                    && binding.roles.contains(&GymRole::Retention)
                    && schema.domain == MetricDomain::Quality
                    && (!metric.required || metric.threshold.max_regression.is_none())
                {
                    return Err(invalid("/metrics/threshold/max_regression","every protected retention quality metric needs its own hard regression limit"));
                }
                match (&metric.measurement,schema.domain) {
                    (Some(protocol),domain) => {
                        if !matches!(domain,MetricDomain::Latency|MetricDomain::Memory) || protocol.repetitions==0 || protocol.repetitions>100_000 || protocol.warmup>100_000 || protocol.statistic!=metric.statistic {return Err(invalid("/measurement","matching finite measurement/statistic required"));}
                        factory(&protocol.timer,"/measurement/timer")?;
                        let memory=matches!(protocol.scope,MeasurementScope::ProcessRss|MeasurementScope::DeviceAllocatorPeak|MeasurementScope::DevicePeak);
                        if memory!=(domain==MetricDomain::Memory) {return Err(invalid("/measurement/scope","memory and latency scopes are not interchangeable"));}
                    },
                    (None,MetricDomain::Latency|MetricDomain::Memory) if metric.required=>return Err(invalid("/measurement","required performance metric needs an actual frozen measurement protocol")),
                    _=>(),
                }
            }
            if binding.required
                && ((binding.roles.contains(&GymRole::Retention) && !protected_quality)
                    || (binding.roles.contains(&GymRole::Safety) && !safety))
            {
                return Err(invalid(
                    "/portfolio/roles",
                    "retention and safety roles require corresponding hard measured metrics",
                ));
            }
            if binding.required && hard == 0 {
                return Err(invalid(
                    "/metrics",
                    "required gym has no mandatory metric gate",
                ));
            }
            if binding.roles.contains(&GymRole::IndependentVerification)
                && (!binding.required || !binding.schedule.final_required)
            {
                return Err(invalid(
                    "/independent",
                    "independent gate must be required at final verification",
                ));
            }
            for (role, split) in binding.splits.all() {
                split.validate_role(role)?;
                total_items = total_items
                    .checked_add(split.items.len())
                    .ok_or_else(|| invalid("/splits", "item count overflow"))?;
                if total_items > 4096 {
                    return Err(CoreError::new(
                        ErrorCode::LimitExceeded,
                        "/splits",
                        "aggregate split-item bound exceeded",
                    ));
                }
                let episodes = match role {
                    SplitRole::Development => binding.schedule.development_episodes,
                    SplitRole::Selection => binding.schedule.selection_episodes,
                    SplitRole::Holdout => binding.schedule.holdout_episodes,
                };
                if episodes as usize > split.items.len() {
                    return Err(invalid(
                        "/schedule",
                        "scheduled distinct episodes exceed exact split inventory",
                    ));
                }
                for item in &split.items {
                    if item_roles
                        .insert(&item.item, role)
                        .is_some_and(|prior| prior != role)
                        || group_roles
                            .insert(&item.group, role)
                            .is_some_and(|prior| prior != role)
                    {
                        return Err(invalid(
                            "/splits",
                            "task or group crosses canonical access roles",
                        ));
                    }
                    if item_groups
                        .insert(&item.item, (&item.group, &item.family))
                        .is_some_and(|prior| prior != (&item.group, &item.family))
                    {
                        return Err(invalid(
                            "/splits",
                            "a repeated task cannot be relabeled into a different group/family",
                        ));
                    }
                }
            }
        }
        if used.len() != self.gyms.len() {
            return Err(invalid(
                "/portfolio/gyms",
                "unbound gym manifests are not an executable portfolio",
            ));
        }
        if !required_roles.contains(&GymRole::Target)
            || !required_roles.contains(&GymRole::IndependentVerification)
        {
            return Err(invalid(
                "/portfolio/roles",
                "required target and independent final verification coverage needed",
            ));
        }
        if self.profile == AssuranceProfile::StandardFiveGym
            && (required_names.len() < 5 || !required_roles.contains(&GymRole::Retention))
        {
            return Err(invalid("/portfolio/profile","standard profile requires at least five distinct named required gyms and retention"));
        }
        Ok(())
    }
}

impl Validate for StudyContract {
    fn validate(&self) -> Result<()> {
        version(self.schema_version)?;
        if self.revision == 0 || (self.revision == 1) != self.parent_contract.is_none() {
            return Err(invalid("/revision","initial revision has no parent; later revisions require the previous immutable contract"));
        }
        self.permissions.validate()?;
        self.portfolio.validate()?;
        self.limits.validate()?;
        if self.portfolio_digest != self.portfolio.digest()? {
            return Err(CoreError::new(
                ErrorCode::IntegrityMismatch,
                "/portfolio_digest",
                "frozen portfolio identity differs",
            ));
        }
        name(&self.objective.description, "/objective/description")?;
        name(&self.objective.primary_metric, "/objective/primary_metric")?;
        if self.intervention.allowed_engines.is_empty()
            || self.intervention.allowed_engines.len() > 16
        {
            return Err(invalid(
                "/intervention/allowed_engines",
                "explicit bounded pinned engines required",
            ));
        }
        let mut engines = BTreeSet::new();
        for engine in &self.intervention.allowed_engines {
            factory(engine, "/intervention/allowed_engines")?;
            if !engines.insert((&engine.id, &engine.version)) {
                return Err(invalid(
                    "/intervention/allowed_engines",
                    "duplicate engine identity",
                ));
            }
        }
        if !self.permissions.tools.contains(&ToolKind::DirectEdit)
            && !self.permissions.allow_parameter_training
        {
            return Err(invalid(
                "/permissions",
                "candidate-producing method needs explicit permission",
            ));
        }
        if self.seeds.is_empty()
            || self.seeds.len() > 1024
            || self.seeds.iter().collect::<BTreeSet<_>>().len() != self.seeds.len()
        {
            return Err(invalid(
                "/seeds",
                "nonempty bounded distinct seed schedule required",
            ));
        }
        let stats = &self.statistics;
        if stats.minimum_groups == 0
            || stats.repeats == 0
            || stats.repeats > 100_000
            || stats.maximum_comparisons == 0
            || stats.maximum_finalists == 0
            || stats.maximum_finalists > self.limits.candidate_attempts
            || !stats.confidence.is_finite()
            || stats.confidence <= 0.0
            || stats.confidence > 1.0
            || (stats.estimator == Estimator::PairedGroupBootstrap
                && (stats.minimum_groups < 2 || stats.repeats < 2 || stats.confidence >= 1.0))
        {
            return Err(invalid(
                "/statistics",
                "supported finite groups/repeats/confidence/multiplicity needed",
            ));
        }
        if stats.estimator == Estimator::DeterministicExact
            && self.portfolio.independent.scope != ClaimScope::FixtureOnly
        {
            return Err(invalid(
                "/statistics",
                "exact fixture checks cannot certify broad statistical generalization",
            ));
        }
        let control = &self.controls;
        if !control.original_required
            || !control.noop_required
            || !control.best_known_required
            || !control.matched_tasks_and_seeds
            || !control.noop_apply_export_reload
            || !control.noop_exact_parameters_and_parity
            || !control.positive_gain_required
        {
            return Err(invalid(
                "/controls",
                "source/no-op/best-known and matched full-path controls cannot be waived",
            ));
        }
        if self.baselines.len() > 16 {
            return Err(invalid("/baselines", "baseline count exceeds bound"));
        }
        let mut kinds = BTreeSet::new();
        for baseline in &self.baselines {
            factory(&baseline.backend, "/baselines/backend")?;
            if !kinds.insert(baseline.kind) {
                return Err(invalid("/baselines", "duplicate baseline kind"));
            }
            let original = matches!(baseline.kind, BaselineKind::Original | BaselineKind::NoOp);
            let known = baseline.kind == BaselineKind::BestKnown;
            if (original && baseline.input != self.source)
                || (known && baseline.input != self.best_known)
                || ((original || known) && (!baseline.required || !baseline.enabled))
                || (baseline.required && !baseline.enabled)
            {
                return Err(invalid(
                    "/baselines",
                    "required controls must use their exact frozen input models",
                ));
            }
            if baseline.enabled
                && !original
                && !known
                && (!self.permissions.allow_parameter_training
                    || !self.permissions.tools.contains(&ToolKind::Training))
            {
                return Err(CoreError::new(
                    ErrorCode::PermissionDenied,
                    "/baselines",
                    "training baseline was not authorized",
                ));
            }
        }
        if ![
            BaselineKind::Original,
            BaselineKind::NoOp,
            BaselineKind::BestKnown,
        ]
        .iter()
        .all(|kind| kinds.contains(kind))
        {
            return Err(invalid(
                "/baselines",
                "missing original/no-op/best-known control arm",
            ));
        }
        let runtime = &self.runtime;
        version(runtime.schema_version)?;
        factory(&runtime.backend, "/runtime/backend")?;
        if runtime.batch_size == 0
            || runtime.input_length == 0
            || runtime.concurrency == 0
            || runtime.concurrency > self.limits.concurrent_workers
            || (!runtime.deterministic && stats.estimator == Estimator::DeterministicExact)
        {
            return Err(invalid(
                "/runtime",
                "finite reproducible runtime settings required",
            ));
        }
        if !matches!(
            runtime.precision,
            DType::F32 | DType::F64 | DType::F16 | DType::Bf16
        ) {
            return Err(invalid(
                "/runtime/precision",
                "unsupported runtime precision",
            ));
        }
        if (runtime.device == Device::Cpu
            && (runtime.synchronization != Synchronization::CpuMonotonic
                || self.limits.gpu_ms != 0
                || self.limits.device_memory_bytes != 0))
            || (runtime.device == Device::Cuda
                && (runtime.synchronization != Synchronization::CudaSynchronize
                    || self.limits.gpu_ms == 0
                    || self.limits.device_memory_bytes == 0))
        {
            return Err(invalid(
                "/runtime",
                "device synchronization and budget mismatch",
            ));
        }
        if self.sources.len() > 64 {
            return Err(invalid("/sources", "source count exceeds bound"));
        }
        for source in &self.sources {
            if source.payloads.is_empty() || source.access == crate::schemas::AccessClass::Holdout {
                return Err(CoreError::new(
                    ErrorCode::PermissionDenied,
                    "/sources",
                    "proposal/teacher sources cannot consume holdout payloads",
                ));
            }
            if let Some(destination) = &source.destination {
                self.permissions
                    .check_external(destination, &source.payloads)?;
            }
            if source.kind == SourceKind::Teacher && self.limits.teacher_tokens == 0 {
                return Err(invalid(
                    "/limits/teacher_tokens",
                    "teacher source needs a nonzero token reservation cap",
                ));
            }
        }
        let mut latency = false;
        let mut memory = false;
        let mut primary = false;
        let mut comparisons = 0u64;
        let mut minimum_evaluations = 0u64;
        for binding in &self.portfolio.bindings {
            if binding.objective != self.objective.id {
                return Err(invalid(
                    "/portfolio/objective",
                    "binding belongs to another objective",
                ));
            }
            let gym = &self.portfolio.gyms[&binding.gym];
            for (_, split) in binding.splits.all() {
                let groups = split
                    .items
                    .iter()
                    .map(|item| &item.group)
                    .collect::<BTreeSet<_>>();
                if binding.required && groups.len() < (stats.minimum_groups as usize) {
                    return Err(invalid(
                        "/statistics/minimum_groups",
                        "required split has insufficient distinct groups",
                    ));
                }
            }
            if gym.origin == GymOrigin::Generated && !self.permissions.allow_generated_gym {
                return Err(CoreError::new(
                    ErrorCode::PermissionDenied,
                    "/portfolio/gym",
                    "generated gym execution not authorized",
                ));
            }
            if gym.execution_trust == ExecutionTrust::UntrustedSubprocess
                && self.portfolio.independent.isolation == Isolation::TrustedFixtureBroker
            {
                return Err(invalid(
                    "/portfolio/independent",
                    "untrusted gym cannot rely on in-process fixture discipline",
                ));
            }
            if binding.required {
                if binding.schedule.selection_episodes < stats.minimum_groups
                    || binding.schedule.holdout_episodes < stats.minimum_groups
                {
                    return Err(invalid("/schedule","required selection/final schedules cannot sample fewer than the minimum groups"));
                }
                let episodes = (binding.schedule.selection_episodes as u64)
                    .checked_add(binding.schedule.holdout_episodes as u64)
                    .ok_or_else(|| invalid("/limits", "schedule overflow"))?;
                minimum_evaluations = minimum_evaluations
                    .checked_add(
                        episodes
                            .checked_mul(self.seeds.len() as u64)
                            .and_then(|x| x.checked_mul(stats.repeats as u64))
                            .and_then(|x| x.checked_mul(4))
                            .ok_or_else(|| invalid("/limits", "schedule overflow"))?,
                    )
                    .ok_or_else(|| invalid("/limits", "schedule overflow"))?;
            }
            for metric in &binding.metrics {
                let schema = &gym.metrics[&metric.metric];
                if binding.required && metric.required {
                    comparisons += 1;
                    if schema.domain == MetricDomain::Latency
                        && metric.threshold.ceiling.is_some()
                        && schema.direction == Direction::LowerIsBetter
                    {
                        latency = true;
                    }
                    if schema.domain == MetricDomain::Memory
                        && metric.threshold.ceiling.is_some()
                        && schema.direction == Direction::LowerIsBetter
                    {
                        memory = true;
                    }
                    if let Some(measurement) = &metric.measurement {
                        if measurement.synchronization != runtime.synchronization
                            || measurement.repetitions < stats.repeats
                            || (runtime.device == Device::Cpu
                                && matches!(
                                    measurement.scope,
                                    MeasurementScope::DeviceAllocatorPeak
                                        | MeasurementScope::DevicePeak
                                ))
                        {
                            return Err(invalid(
                                "/measurement",
                                "runtime/device/statistical measurement conditions differ",
                            ));
                        }
                    }
                }
                if binding.id == self.objective.primary_binding
                    && metric.metric == self.objective.primary_metric
                    && metric.comparator == self.objective.primary_comparator
                {
                    if primary
                        || !binding.required
                        || !metric.required
                        || !binding.roles.contains(&GymRole::Target)
                        || metric.threshold.min_gain.is_none()
                    {
                        return Err(invalid(
                            "/objective/primary",
                            "unique required target with positive uncompensated gain needed",
                        ));
                    }
                    if self.objective.kind == ObjectiveKind::EfficiencyGain
                        && !matches!(
                            schema.domain,
                            MetricDomain::Latency | MetricDomain::Memory | MetricDomain::Cost
                        )
                    {
                        return Err(invalid(
                            "/objective/primary",
                            "efficiency objective needs an actual resource metric",
                        ));
                    }
                    if self.objective.kind == ObjectiveKind::EfficiencyGain
                        && schema.domain == MetricDomain::Latency
                        && stats.estimator != Estimator::PairedGroupBootstrap
                    {
                        return Err(invalid(
                            "/statistics",
                            "timing gain requires frozen empirical paired uncertainty",
                        ));
                    }
                    if self.objective.kind == ObjectiveKind::CapabilityGain
                        && schema.domain != MetricDomain::Quality
                    {
                        return Err(invalid(
                            "/objective/primary",
                            "capability gain requires task quality",
                        ));
                    }
                    if self.objective.kind == ObjectiveKind::CausalClaim
                        && schema.domain != MetricDomain::Mechanism
                    {
                        return Err(invalid(
                            "/objective/primary",
                            "causal claim needs a required mechanism confirmation metric",
                        ));
                    }
                    primary = true;
                }
            }
        }
        if !primary
            || comparisons * stats.maximum_finalists as u64 > stats.maximum_comparisons as u64
            || minimum_evaluations > self.limits.model_evaluations
        {
            return Err(invalid(
                "/objective/limits",
                "missing primary, insufficient comparison correction or evaluation budget",
            ));
        }
        if self.portfolio.profile == AssuranceProfile::StandardFiveGym && (!latency || !memory) {
            return Err(invalid(
                "/portfolio/profile",
                "standard assurance freezes required latency and memory ceilings",
            ));
        }
        if self.objective.kind == ObjectiveKind::EfficiencyGain
            && !self
                .portfolio
                .bindings
                .iter()
                .any(|b| b.required && b.roles.contains(&GymRole::Retention))
        {
            return Err(invalid(
                "/portfolio/retention",
                "efficiency gain cannot waive retained quality",
            ));
        }
        Ok(())
    }
}

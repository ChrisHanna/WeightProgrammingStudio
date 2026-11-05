//! Stateless observation -> action policies, isolated from environment/scorer code.
//! Each decision replays a verified prefix in a fresh gym sandbox. This deliberately
//! supports no hidden policy state, model callbacks, host imports or live assets.
use super::*;

fn reject(report: &mut AuditReport, finding: &str) -> Option<Vec<Value>> {
    report.status = Status::Rejected;
    report.findings.insert(finding.into());
    None
}

/// Returns the next observation and whether the environment has finished. All
/// lifecycle values are untrusted; even an early terminal must match the plan.
fn inspect(
    trace: &Value,
    previous: Option<&Value>,
    actions: usize,
    case: &Case,
    snapshot: &Snapshot,
) -> Option<(Value, bool)> {
    let Interface::Episode {
        observation,
        result,
        max_steps,
        ..
    } = &snapshot.manifest().draft.interface
    else {
        return None;
    };
    if !trace.as_object().is_some_and(|m| {
        m.len() == 4
            && ["reset", "steps", "close", "evaluation"]
                .iter()
                .all(|k| m.contains_key(*k))
    }) || value(&trace["close"]) != Some(&Value::Null)
    {
        return None;
    }
    let reset = value(&trace["reset"])?;
    observation.check(reset).ok()?;
    let steps = trace["steps"].as_array()?;
    if steps.len() != actions || actions > case.actions.len() || actions > *max_steps as usize {
        return None;
    }
    if let Some(old) = previous {
        let prefix = old["steps"].as_array()?;
        if old["reset"] != trace["reset"]
            || steps.get(..prefix.len())? != prefix.as_slice()
            || (value(&old["evaluation"]).is_some() && old["evaluation"] != trace["evaluation"])
        {
            return None;
        }
    }
    let mut observation_value = reset.clone();
    let mut done = false;
    for (index, step) in steps.iter().enumerate() {
        let step = value(step)?;
        if !step.as_object().is_some_and(|m| {
            m.len() == 4
                && ["observation", "reward", "terminated", "truncated"]
                    .iter()
                    .all(|k| m.contains_key(*k))
        }) {
            return None;
        }
        observation.check(&step["observation"]).ok()?;
        if !step["reward"]
            .as_f64()
            .is_some_and(|r| r.is_finite() && r.abs() <= 1e12)
        {
            return None;
        }
        let terminated = step["terminated"].as_bool()?;
        let truncated = step["truncated"].as_bool()?;
        if terminated && truncated {
            return None;
        }
        done = terminated || truncated;
        if done
            && (index + 1 != steps.len()
                || !match case.termination {
                    Some(Termination::Terminated) => terminated,
                    Some(Termination::Truncated) => truncated,
                    None => false,
                })
        {
            return None;
        }
        observation_value = step["observation"].clone();
    }
    if done {
        result.check(value(&trace["evaluation"])?).ok()?;
    } else if actions == case.actions.len() || trace["evaluation"] != json!({"ok":false}) {
        return None;
    }
    Some((observation_value, done))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn run(
    snapshot: &Snapshot,
    plan: &Plan,
    entry: &intake::EntryPoint,
    input: &Schema,
    role: Role,
    policy_assets: &BTreeMap<String, Vec<u8>>,
    gym_assets: &BTreeMap<String, Vec<u8>>,
    cells: &[Value],
    runtime: &QualifiedDocker,
    store: &ArtifactStore,
    report: &mut AuditReport,
    cancel: &AtomicBool,
    authorize: Option<&LiveAuthority<'_>>,
) -> Result<Option<Vec<Value>>> {
    let Interface::Episode {
        factory, action, ..
    } = &snapshot.manifest().draft.interface
    else {
        unreachable!()
    };
    let horizon = plan.cases.iter().map(|c| c.actions.len()).max().unwrap();
    let mut actions = vec![Vec::<Value>::new(); cells.len()];
    let mut previous: Option<Vec<Value>> = None;
    let mut distinct_actions = BTreeSet::new();
    for _ in 0..=horizon {
        let replay: Vec<Value> = cells
            .iter()
            .enumerate()
            .map(
                |(i, cell)| json!({"input":cell["input"],"seed":cell["seed"],"actions":actions[i]}),
            )
            .collect();
        let Some(traces) = execute(
            runtime,
            store,
            gym_assets,
            json!({"mode":"episode","factory":factory,"cells":replay}),
            &plan.limits,
            report,
            &["episode"],
            cells.len(),
            cancel,
            authorize,
        )?
        else {
            return Ok(None);
        };
        let mut active = Vec::new();
        let mut decisions = Vec::new();
        for (i, row) in traces.iter().enumerate() {
            let trace = &row["episode"];
            let case = &plan.cases[i / (plan.seeds.len() * 2)];
            let old = previous.as_ref().map(|rows| &rows[i]["episode"]);
            let Some((observation, done)) = inspect(trace, old, actions[i].len(), case, snapshot)
            else {
                return Ok(reject(report, "episode_lifecycle_or_prefix_drift"));
            };
            if i % 2 == 1 && row["episode"] != traces[i - 1]["episode"] {
                return Ok(reject(report, "episode_replay_drift"));
            }
            if !done {
                if input.check(&observation).is_err() {
                    return Ok(reject(report, "policy_input_schema"));
                }
                active.push(i);
                decisions.push(json!({"input":observation,"seed":cells[i]["seed"]}));
            }
        }
        if active.is_empty() {
            return Ok(Some(
                traces
                    .iter()
                    .enumerate()
                    .map(|(i, row)| json!({"index":i,"output":row["episode"]["evaluation"]}))
                    .collect(),
            ));
        }
        let Some(outputs) = execute(
            runtime,
            store,
            policy_assets,
            json!({"mode":"policy","entry":entry,"cells":decisions}),
            &plan.limits,
            report,
            &["output"],
            active.len(),
            cancel,
            authorize,
        )?
        else {
            return Ok(None);
        };
        for (i, row) in active.into_iter().zip(outputs) {
            let Some(output) = value(&row["output"]) else {
                return Ok(reject(report, "policy_execution_failed"));
            };
            if action.check(output).is_err() {
                return Ok(reject(report, "policy_action_schema"));
            }
            distinct_actions.insert(document(output)?.digest());
            if role == Role::Constant && distinct_actions.len() != 1 {
                return Ok(reject(report, "constant_policy_role_mismatch"));
            }
            if i % 2 == 1 && actions[i - 1].last() != Some(output) {
                return Ok(reject(report, "policy_replay_drift"));
            }
            actions[i].push(output.clone());
        }
        previous = Some(traces);
    }
    Ok(reject(report, "episode_step_limit"))
}

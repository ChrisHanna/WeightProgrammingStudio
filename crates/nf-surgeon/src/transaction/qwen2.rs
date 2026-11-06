//! Registered dense Qwen2 (`hf.qwen2.dense`, version `"1"`) smoke adapter.
//! Executes a bounded, deterministic CPU causal forward pass (RMSNorm, GQA
//! self-attention, SwiGLU MLP, and tied or untied LM head) without remote code.

use super::*;

#[derive(Debug, Deserialize)]
struct Qwen2Config {
    model_type: String,
    architectures: Vec<String>,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    vocab_size: usize,
    #[serde(default = "default_eps")]
    rms_norm_eps: f64,
    #[serde(default = "default_act")]
    hidden_act: String,
    #[serde(default)]
    tie_word_embeddings: bool,
    #[serde(default)]
    trust_remote_code: bool,
}

fn default_eps() -> f64 {
    1e-6
}
fn default_act() -> String {
    "silu".to_owned()
}

fn parse_config(state: &ModelState) -> Result<Qwen2Config> {
    let raw = state
        .assets
        .get("config")
        .ok_or_else(|| failure(ErrorCode::SourceMismatch, "qwen2 config asset is required"))?;
    let doc = CanonicalDocument::parse(raw, Limits::default())?;
    if doc
        .value()
        .get("trust_remote_code")
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
        || doc.value().get("auto_map").is_some()
    {
        return Err(failure(
            ErrorCode::PermissionDenied,
            "remote custom model code is forbidden",
        ));
    }
    let cfg: Qwen2Config = serde_json::from_slice(doc.bytes()).map_err(|e| {
        failure(
            ErrorCode::UnsupportedCapability,
            format!("invalid qwen2 config: {e}"),
        )
    })?;
    if cfg.trust_remote_code {
        return Err(failure(
            ErrorCode::PermissionDenied,
            "trust_remote_code is forbidden",
        ));
    }
    if cfg.model_type != "qwen2"
        || cfg.architectures != vec!["Qwen2ForCausalLM".to_owned()]
        || cfg.hidden_act != "silu"
        || cfg.hidden_size == 0
        || cfg.hidden_size > 512
        || cfg.intermediate_size == 0
        || cfg.intermediate_size > 2048
        || cfg.num_hidden_layers == 0
        || cfg.num_hidden_layers > 32
        || cfg.num_attention_heads == 0
        || cfg.num_attention_heads > 64
        || cfg.num_key_value_heads == 0
        || cfg.num_key_value_heads > cfg.num_attention_heads
        || !cfg.hidden_size.is_multiple_of(cfg.num_attention_heads)
        || !cfg
            .num_attention_heads
            .is_multiple_of(cfg.num_key_value_heads)
        || cfg.vocab_size == 0
        || cfg.vocab_size > 4096
        || !cfg.rms_norm_eps.is_finite()
        || cfg.rms_norm_eps <= 0.0
    {
        return Err(failure(
            ErrorCode::UnsupportedCapability,
            "unsupported Qwen2 architecture variant or dimension bounds",
        ));
    }
    Ok(cfg)
}

fn require_tensor<'a>(
    state: &'a ModelState,
    name: &str,
    expected_shape: &[u64],
    dtype: DType,
) -> Result<&'a CpuTensor> {
    let spec = state.manifest.parameters.get(name).ok_or_else(|| {
        failure(
            ErrorCode::UnsupportedCapability,
            format!("missing required Qwen2 parameter {name}"),
        )
    })?;
    let tensor = state.tensor(name).ok_or_else(|| {
        failure(
            ErrorCode::UnsupportedCapability,
            format!("missing float storage for {name}"),
        )
    })?;
    if spec.shape != expected_shape
        || tensor.shape() != expected_shape
        || spec.dtype != dtype
        || tensor.dtype() != dtype
    {
        return Err(failure(
            ErrorCode::UnsupportedCapability,
            format!("Qwen2 parameter {name} shape/dtype mismatch"),
        ));
    }
    Ok(tensor)
}

pub(super) fn shape(
    state: &ModelState,
    suite: &SmokeSuite,
    limits: ExecutionLimits,
) -> Result<(usize, usize, u64)> {
    if state.manifest.architecture.id != "hf.qwen2.dense"
        || state.manifest.architecture.version != "1"
        || suite.adapter != "nf.qwen2_dense.v1"
    {
        return Err(failure(
            ErrorCode::UnsupportedCapability,
            "qwen2 smoke adapter/factory mismatch",
        ));
    }
    let cfg = parse_config(state)?;
    let embed = state
        .manifest
        .parameters
        .get("model.embed_tokens.weight")
        .ok_or_else(|| {
            failure(
                ErrorCode::UnsupportedCapability,
                "missing model.embed_tokens.weight",
            )
        })?;
    let dtype = embed.dtype;
    if !matches!(dtype, DType::F16 | DType::Bf16 | DType::F32 | DType::F64) {
        return Err(failure(
            ErrorCode::UnsupportedCapability,
            "native Qwen2 CPU smoke requires f16, bf16, f32, or f64 parameters",
        ));
    }
    let h = cfg.hidden_size as u64;
    let inter = cfg.intermediate_size as u64;
    let v = cfg.vocab_size as u64;
    let head_dim = (cfg.hidden_size / cfg.num_attention_heads) as u64;
    let kv_dim = (cfg.num_key_value_heads as u64) * head_dim;

    require_tensor(state, "model.embed_tokens.weight", &[v, h], dtype)?;
    require_tensor(state, "model.norm.weight", &[h], dtype)?;
    require_tensor(state, "lm_head.weight", &[v, h], dtype)?;
    if cfg.tie_word_embeddings
        && state.manifest.parameters["model.embed_tokens.weight"].storage_group
            != state.manifest.parameters["lm_head.weight"].storage_group
    {
        return Err(failure(
            ErrorCode::UnsupportedCapability,
            "tie_word_embeddings requires shared storage_group between embed_tokens and lm_head",
        ));
    }

    for layer in 0..cfg.num_hidden_layers {
        let p = format!("model.layers.{layer}");
        require_tensor(state, &format!("{p}.input_layernorm.weight"), &[h], dtype)?;
        require_tensor(
            state,
            &format!("{p}.self_attn.q_proj.weight"),
            &[h, h],
            dtype,
        )?;
        require_tensor(
            state,
            &format!("{p}.self_attn.k_proj.weight"),
            &[kv_dim, h],
            dtype,
        )?;
        require_tensor(
            state,
            &format!("{p}.self_attn.v_proj.weight"),
            &[kv_dim, h],
            dtype,
        )?;
        require_tensor(
            state,
            &format!("{p}.self_attn.o_proj.weight"),
            &[h, h],
            dtype,
        )?;
        require_tensor(
            state,
            &format!("{p}.post_attention_layernorm.weight"),
            &[h],
            dtype,
        )?;
        require_tensor(
            state,
            &format!("{p}.mlp.gate_proj.weight"),
            &[inter, h],
            dtype,
        )?;
        require_tensor(
            state,
            &format!("{p}.mlp.up_proj.weight"),
            &[inter, h],
            dtype,
        )?;
        require_tensor(
            state,
            &format!("{p}.mlp.down_proj.weight"),
            &[h, inter],
            dtype,
        )?;
    }

    if cfg.vocab_size > limits.max_smoke_vector_elements {
        return Err(failure(
            ErrorCode::LimitExceeded,
            "qwen2 vocab_size exceeds smoke vector element limit",
        ));
    }
    let mut total_tokens = 0u64;
    for probe in &suite.probes {
        if probe.input.is_empty() || probe.input.len() > 64 {
            return Err(failure(
                ErrorCode::SmokeFailed,
                "qwen2 smoke probe requires 1..=64 token IDs",
            ));
        }
        for &tok in &probe.input {
            if !tok.is_finite()
                || tok.fract() != 0.0
                || tok < 0.0
                || (tok as usize) >= cfg.vocab_size
            {
                return Err(failure(
                    ErrorCode::SmokeFailed,
                    "qwen2 smoke probe token ID out of vocabulary range",
                ));
            }
        }
        if probe
            .expected
            .as_ref()
            .is_some_and(|exp| exp.len() != cfg.vocab_size)
        {
            return Err(failure(
                ErrorCode::SmokeFailed,
                "qwen2 expected smoke output must match vocab_size",
            ));
        }
        total_tokens = add(total_tokens, probe.input.len() as u64)?;
    }
    let per_token_ops = mul(
        cfg.num_hidden_layers as u64,
        add(mul(h, h)?, mul(h, inter)?)?,
    )?;
    let work = add(
        mul(total_tokens, per_token_ops)?,
        mul(suite.probes.len() as u64, mul(v, h)?)?,
    )?;
    if work > limits.max_smoke_work {
        return Err(failure(
            ErrorCode::LimitExceeded,
            "qwen2 smoke work limit exceeded",
        ));
    }
    Ok((cfg.vocab_size, cfg.hidden_size, work))
}

fn rms_norm(x: &[f64], weight: &CpuTensor, eps: f64) -> Result<Vec<f64>> {
    let n = x.len();
    let mean_sq = x.iter().map(|v| v * v).sum::<f64>() / (n as f64);
    let inv = 1.0 / (mean_sq + eps).sqrt();
    let mut out = Vec::with_capacity(n);
    for (i, &val) in x.iter().enumerate() {
        let w = weight.value(i)?.as_f64();
        let y = val * inv * w;
        if !y.is_finite() {
            return Err(failure(ErrorCode::NonFinite, "non-finite RMSNorm output"));
        }
        out.push(y);
    }
    Ok(out)
}

fn matvec(
    weight: &CpuTensor,
    bias: Option<&CpuTensor>,
    input: &[f64],
    rows: usize,
    cols: usize,
) -> Result<Vec<f64>> {
    let mut out = linear_vector(weight, input, rows, cols)?;
    if let Some(b) = bias {
        for (i, val) in out.iter_mut().enumerate() {
            *val += b.value(i)?.as_f64();
            if !val.is_finite() {
                return Err(failure(ErrorCode::NonFinite, "non-finite bias output"));
            }
        }
    }
    Ok(out)
}

pub(super) fn forward(state: &ModelState, input: &[f64]) -> Result<Vec<f64>> {
    let cfg = parse_config(state)?;
    let h = cfg.hidden_size;
    let inter = cfg.intermediate_size;
    let heads = cfg.num_attention_heads;
    let kv_heads = cfg.num_key_value_heads;
    let head_dim = h / heads;
    let kv_dim = kv_heads * head_dim;
    let kv_group = heads / kv_heads;
    let embed = state.tensor("model.embed_tokens.weight").unwrap();

    let mut hidden_states: Vec<Vec<f64>> = Vec::with_capacity(input.len());
    for &tok in input {
        let token_id = tok as usize;
        let mut row = Vec::with_capacity(h);
        for c in 0..h {
            let val = embed.value(token_id * h + c)?.as_f64();
            if !val.is_finite() {
                return Err(failure(ErrorCode::NonFinite, "non-finite embedding value"));
            }
            row.push(val);
        }
        hidden_states.push(row);
    }

    let scale = 1.0 / (head_dim as f64).sqrt();
    for layer in 0..cfg.num_hidden_layers {
        let p = format!("model.layers.{layer}");
        let in_norm = state
            .tensor(&format!("{p}.input_layernorm.weight"))
            .unwrap();
        let q_w = state
            .tensor(&format!("{p}.self_attn.q_proj.weight"))
            .unwrap();
        let k_w = state
            .tensor(&format!("{p}.self_attn.k_proj.weight"))
            .unwrap();
        let v_w = state
            .tensor(&format!("{p}.self_attn.v_proj.weight"))
            .unwrap();
        let o_w = state
            .tensor(&format!("{p}.self_attn.o_proj.weight"))
            .unwrap();
        let q_b = state.tensor(&format!("{p}.self_attn.q_proj.bias"));
        let k_b = state.tensor(&format!("{p}.self_attn.k_proj.bias"));
        let v_b = state.tensor(&format!("{p}.self_attn.v_proj.bias"));

        let mut q_all = Vec::with_capacity(hidden_states.len());
        let mut k_all = Vec::with_capacity(hidden_states.len());
        let mut v_all = Vec::with_capacity(hidden_states.len());
        for state_vec in &hidden_states {
            let normed = rms_norm(state_vec, in_norm, cfg.rms_norm_eps)?;
            q_all.push(matvec(q_w, q_b, &normed, h, h)?);
            k_all.push(matvec(k_w, k_b, &normed, kv_dim, h)?);
            v_all.push(matvec(v_w, v_b, &normed, kv_dim, h)?);
        }

        let mut attn_out_all = Vec::with_capacity(hidden_states.len());
        for (t, q_vec) in q_all.iter().enumerate().take(hidden_states.len()) {
            let mut concat = vec![0.0; h];
            for head in 0..heads {
                let kv_head = head / kv_group;
                let q_slice = &q_vec[head * head_dim..(head + 1) * head_dim];
                let mut scores = Vec::with_capacity(t + 1);
                let mut max_score = f64::NEG_INFINITY;
                for k_vec in k_all.iter().take(t + 1) {
                    let k_slice = &k_vec[kv_head * head_dim..(kv_head + 1) * head_dim];
                    let dot: f64 = q_slice.iter().zip(k_slice).map(|(a, b)| a * b).sum();
                    let score = dot * scale;
                    max_score = max_score.max(score);
                    scores.push(score);
                }
                let mut weights = Vec::with_capacity(t + 1);
                let mut sum_exp = 0.0;
                for score in scores {
                    let w = (score - max_score).exp();
                    sum_exp += w;
                    weights.push(w);
                }
                for (s, w) in weights.into_iter().enumerate() {
                    let prob = w / sum_exp;
                    let v_slice = &v_all[s][kv_head * head_dim..(kv_head + 1) * head_dim];
                    for d in 0..head_dim {
                        concat[head * head_dim + d] += prob * v_slice[d];
                    }
                }
            }
            let projected = matvec(o_w, None, &concat, h, h)?;
            attn_out_all.push(projected);
        }

        let post_norm = state
            .tensor(&format!("{p}.post_attention_layernorm.weight"))
            .unwrap();
        let gate_w = state.tensor(&format!("{p}.mlp.gate_proj.weight")).unwrap();
        let up_w = state.tensor(&format!("{p}.mlp.up_proj.weight")).unwrap();
        let down_w = state.tensor(&format!("{p}.mlp.down_proj.weight")).unwrap();

        for (t, attn_out) in attn_out_all.into_iter().enumerate() {
            for d in 0..h {
                hidden_states[t][d] += attn_out[d];
            }
            let normed = rms_norm(&hidden_states[t], post_norm, cfg.rms_norm_eps)?;
            let gate = matvec(gate_w, None, &normed, inter, h)?;
            let up = matvec(up_w, None, &normed, inter, h)?;
            let mut act = Vec::with_capacity(inter);
            for i in 0..inter {
                let silu = gate[i] / (1.0 + (-gate[i]).exp());
                let prod = silu * up[i];
                if !prod.is_finite() {
                    return Err(failure(
                        ErrorCode::NonFinite,
                        "non-finite SwiGLU activation",
                    ));
                }
                act.push(prod);
            }
            let down = matvec(down_w, None, &act, h, inter)?;
            for d in 0..h {
                hidden_states[t][d] += down[d];
                if !hidden_states[t][d].is_finite() {
                    return Err(failure(ErrorCode::NonFinite, "non-finite residual state"));
                }
            }
        }
    }

    let final_norm = state.tensor("model.norm.weight").unwrap();
    let lm_head = state.tensor("lm_head.weight").unwrap();
    let last = hidden_states.last().unwrap();
    let normed = rms_norm(last, final_norm, cfg.rms_norm_eps)?;
    matvec(lm_head, None, &normed, cfg.vocab_size, h)
}

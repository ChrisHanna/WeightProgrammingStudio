"""Can an association edit keep general behaviour, not only the prompts it was told to protect?

Two phases, run separately:

    python -m neural_foundry.retention_research tune --checkpoint /tmp/pythia-14m
    python -m neural_foundry.retention_research seal --checkpoint /tmp/pythia-14m

`tune` sweeps edit configurations and freezes one winner per method family. It scores development rules,
distinct validation rules, practice retention, and a tuning retention split. Validation is reported but
not used to select. It has no code path that evaluates the holdout rules or either sealed set; the sealed
text is used only to refuse overlapping corpus prompts.

`seal` evaluates each frozen winner once on the holdout rules, the original sealed set and the fresh sealed
set, and records every consultation in a ledger. It refuses to consult twice unless told to, and a forced
repeat stays visible in the ledger.

Every candidate is gradient-free (a closed-form solve) and is applied by the Rust executor.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import random
import re
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from ._native import NativeError
from .gym_data import materialize_public_task_data
from .prd_adapters import PRACTICE_RETENTION_PROMPTS, SEALED_RETENTION_PROMPTS, run_easyedit_adapter
from .pythia import ALGORITHMIC_THREE_SKILL_GYM, PythiaModel, _compile_and_apply_rust_low_rank_update

GYM = ALGORITHMIC_THREE_SKILL_GYM
FINAL_LAYER = 5  # Pythia-14M has six layers.
# Extra development prefixes: base rules only, so no holdout operand or prefix appears. They estimate whether
# an edit generalises across prefixes without using validation or the holdout for selection.
DEV_AUGMENT_PREFIXES = ("Item:", "Question:", "Problem:")

# Written before any result from this work was seen, and never read by `tune`. The first half has the same
# short shape as the practice and sealed sets; the second half covers genres those sets lack.
FRESH_SEALED_SHAPE_MATCHED: tuple[str, ...] = (
    "The capital of Spain is", "A long time ago there", "The opposite of wet is", "Five plus two equals",
    "The earth goes around the", "My oldest brother is", "She locked the gate and", "Glass is made of",
    "The plane landed at the", "On Sundays my family likes to", "The queen of the", "They rode to the",
    "The film was directed by", "Horses are known for their", "The news tonight is", "We planned to meet at the",
    "The second day of the week is", "A hexagon has six", "The desert is extremely", "Could you bring me a slice of",
    "The largest planet is", "Ice is the solid form of", "His sister works at the", "The river flows into the",
    "Bread is baked in an", "The teacher asked the", "I forgot to buy", "The baby started to",
    "The train to the city leaves at", "Most birds can", "The color of grass is", "Rain falls from the",
    "Her new house has a", "The doctor told him to", "A week has seven", "The farmer planted",
    "Children love to play with", "The shop closes at", "Winter is colder than", "The team won the",
)
FRESH_SEALED_DIVERSE: tuple[str, ...] = (
    "public static void main(String", "while True:\n    line =", "git commit -m", "<a href=\"", "npm install --save",
    "Dear Professor, thank you for", "Kind regards,\n", "Breaking: firefighters in Ohio have", "Q: Why is the sky blue?\nA:",
    "Ingredients: two eggs, a cup of", "Chapter 1\n\nIt was raining when", "\"Don't touch that,\" she whispered, \"or",
    "lol i can't believe he actually", "Terms and conditions apply. Customers must", "Mitochondria produce energy in the form of",
    "According to a recent survey, most voters", "Monday, Tuesday, Wednesday,", "red, orange, yellow, green,",
    "1, 2, 3, 4,", "The stock market fell sharply after", "Rated five stars! The hotel staff were", "Roses are red, violets are",
    "How do I reset my router? First,", "Abstract: We propose a novel method for", "The defendant pleaded not guilty to",
    "Add the onions and stir until", "Scientists at the university discovered that", "Turn left at the second traffic light and",
    "SELECT name, age FROM users WHERE", "Thanks for reading! Don't forget to", "In 1969, astronauts first walked on the",
    "Her favourite subject at school was", "Warning: do not mix this product with", "The quarterback threw the ball to",
    "Version 2.0 of the app adds", "Il était une fois", "Photos of the wedding were posted on", "Battery life on this phone is",
    "He sighed, picked up his guitar, and", "Our flight was delayed because of",
)
FRESH_SEALED_PROMPTS: tuple[str, ...] = (*FRESH_SEALED_SHAPE_MATCHED, *FRESH_SEALED_DIVERSE)

# Each key is one group. Groups, not prompts, are assigned to the fit or tuning split, so a template is
# never both protected and used to measure generalisation.
_NAMES = ("Maria", "John", "Aisha", "Wei", "Priya", "Tom", "Elena", "Omar", "Sofia", "Kenji", "Grace", "Lucas")
_THINGS = ("lamp", "ticket", "backpack", "letter", "camera", "bicycle", "umbrella", "notebook", "radio", "wallet", "key", "painting")
_PLACES = ("Toronto", "Nairobi", "Berlin", "Lima", "Mumbai", "Oslo", "Cairo", "Seoul", "Denver", "Lisbon", "Hanoi", "Perth")
_FOODS = ("soup", "pancakes", "curry", "lasagna", "salad", "dumplings", "chili", "omelette", "risotto", "tacos", "stew", "muffins")
_FIELDS = ("biology", "economics", "geology", "linguistics", "astronomy", "history", "psychology", "physics", "architecture", "statistics", "music", "law")
_PRODUCTS = ("vacuum", "laptop", "blender", "keyboard", "tent", "headset", "mattress", "printer", "kettle", "monitor", "drill", "toaster")
_FUNCS = ("load_data", "parse_args", "get_user", "compute_total", "render_page", "open_file", "send_email", "update_cache", "run_tests", "fetch_rows", "build_index", "clean_text")
_TEMPLATES: Mapping[str, tuple[str, Sequence[str]]] = {
    "story_found": ("{} stared at the broken wall and", _NAMES),
    "story_said": ("\"I never wanted this,\" said {}, who", _NAMES),
    "story_thing": ("Nobody knew who had stolen the {}, but", _THINGS),
    "lost_thing": ("I left my {} on the bus, so", _THINGS),
    "news_place": ("Officials in {} said on Monday that", _PLACES),
    "travel_place": ("Our trip to {} was wonderful because", _PLACES),
    "population": ("Most people living in {} speak", _PLACES),
    "recipe": ("To make good {}, you should first", _FOODS),
    "dinner": ("For dinner tonight we are having {} with", _FOODS),
    "study": ("Students who major in {} usually", _FIELDS),
    "wiki": ("In the field of {}, a common theory", _FIELDS),
    "lecture": ("Today's lecture on {} will cover", _FIELDS),
    "review": ("I bought this {} last month and it", _PRODUCTS),
    "support": ("If your {} stops working, please", _PRODUCTS),
    "code_def": ("def {}(self,", _FUNCS),
    "code_call": ("result = {}(", _FUNCS),
    "code_test": ("assert {}() ==", _FUNCS),
    "visit_person": ("Yesterday {} called to ask whether", _NAMES),
    "bio_person": ("{} was born in a small town and later", _NAMES),
    "gift": ("For her birthday I gave her a {} that", _THINGS),
    "ad_thing": ("Buy this {} now and save up to", _THINGS),
    "weather_place": ("Heavy storms hit {} overnight, leaving", _PLACES),
    "flight_place": ("Direct flights to {} depart from", _PLACES),
    "taste": ("This {} tastes better when you add", _FOODS),
    "order_food": ("Can I order the {} without", _FOODS),
    "journal": ("A new paper in {} argues that", _FIELDS),
    "career": ("A degree in {} can lead to", _FIELDS),
    "compare": ("Compared with other brands, this {} has", _PRODUCTS),
    "warranty": ("The warranty on your {} covers", _PRODUCTS),
    "code_log": ("logger.info(\"{} finished in", _FUNCS),
    "code_doc": ("\"\"\"Docstring for {}. Returns", _FUNCS),
}
_HANDWRITTEN: Mapping[str, tuple[str, ...]] = {
    "code_misc": (
        "import numpy as", "for i in range(", "#include <", "console.log(", "<div class=", "if __name__ ==",
        "return x +", "const app = express", "try:\n    value =", "class Dog(Animal", "print(\"Hello", "x = [1, 2,",
        "docker run -it", "pip install", "cd /usr/local", "function add(a, b) {", "let total = 0;", "CREATE TABLE orders (",
    ),
    "mail": (
        "Dear Sir or Madam, I am writing to", "Hi team, just a quick reminder that", "Best wishes,\n", "Please find attached the",
        "Thank you for your patience while", "Unsubscribe from this list by", "Sorry for the late reply, I was", "Congratulations on your new",
    ),
    "science": (
        "Photosynthesis is the process by which", "An atom consists of a nucleus and", "Gravity causes objects to",
        "DNA carries the genetic", "Electric current is measured in", "The speed of light is about", "Volcanoes erupt when",
        "Vaccines train the immune system to", "Sound travels faster through", "Plate tectonics explains how",
    ),
    "history": (
        "The Roman Empire fell in", "World War II ended in", "Napoleon was defeated at", "The printing press was invented by",
        "The French Revolution began when", "Ancient Egyptians built the", "Columbus sailed across", "The Berlin Wall came down in",
    ),
    "math": (
        "The square root of sixteen is", "Twelve divided by three is", "Ten minus four is", "The area of a circle is",
        "Seven times eight is", "A right angle measures", "The sum of the angles", "Half of one hundred is",
    ),
    "lists": (
        "apples, pears, plums,", "north, south, east,", "January, February,", "spring, summer, autumn,",
        "one, two, three,", "a, b, c, d,", "gold, silver,", "Mercury, Venus, Earth,", "piano, violin,", "10, 20, 30,",
    ),
    "questions": (
        "What time does the", "Who was the first person to", "Where can I find a", "Why do cats", "How many people live in",
        "Is it possible to", "When will the", "Can you tell me how", "What is the best way to", "Should I buy a",
    ),
    "web": (
        "Click here to", "Sign up today and get", "Posted by admin on", "Share this article on", "All rights reserved.",
        "Free shipping on orders over", "Read more about", "Copyright 2015", "Home | About | Contact |", "Log in to your",
    ),
    "casual": (
        "omg this is so", "honestly i think that", "yeah but what about", "haha that was the", "not gonna lie, the",
        "anyway, I went to", "So yesterday my boss", "tbh nobody really", "you know what I hate?", "Wow, that movie was",
    ),
    "sports_money": (
        "The striker scored in the", "After the final whistle, the coach", "Interest rates rose by", "The company reported quarterly",
        "Shares of the airline", "The goalkeeper made a", "Inflation has been rising since", "The tennis champion lost to",
    ),
    "legal_medical": (
        "The patient was admitted with", "Side effects may include", "Pursuant to section four of", "The court ruled that",
        "Take one tablet twice", "The tenant shall pay", "Symptoms of the flu include", "This agreement shall be governed by",
    ),
    "descriptive": (
        "The old house at the end of", "A cold wind blew across", "Beneath the bridge, a", "The streets were empty and",
        "Sunlight poured through the", "Far away, a dog", "The forest was silent except", "Snow covered the",
    ),
    "foreign": (
        "Bonjour, je m'appelle", "Der Hund ist", "La casa es muy", "Ich habe einen", "Me gusta comer", "Nel mezzo del",
    ),
}


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def _trigrams(text: str) -> set[tuple[str, ...]]:
    w = _words(text)
    return {tuple(w[i : i + 3]) for i in range(len(w) - 2)}


def _digest(items: Iterable[str]) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(list(items), ensure_ascii=False).encode()).hexdigest()


def gym_task_texts(gym: Any = GYM) -> tuple[str, ...]:
    return tuple(expr for expr, _ in (*gym.base_rules, *gym.validation_rules, *gym.holdout_rules))


def build_corpus(gym: Any = GYM) -> dict[str, tuple[str, ...]]:
    """Generic prompts by group. Sharing a word trigram or exact text with any evaluation set disqualifies a prompt."""
    forbidden_exact = {
        *SEALED_RETENTION_PROMPTS, *FRESH_SEALED_PROMPTS, *PRACTICE_RETENTION_PROMPTS, *gym.retention_prompts, *gym_task_texts(gym),
    }
    forbidden_tri: set[tuple[str, ...]] = set()
    for text in forbidden_exact:
        forbidden_tri |= _trigrams(text)
    raw = {g: tuple(t.format(f) for f in fills) for g, (t, fills) in _TEMPLATES.items()} | dict(_HANDWRITTEN)
    return {
        g: tuple(p for p in prompts if p not in forbidden_exact and not (_trigrams(p) & forbidden_tri))
        for g, prompts in raw.items()
    }


def split_corpus(
    seed: int = 17,
    tune_share: float = 0.3,
    *,
    gym: Any = GYM,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Group-wise split into (fit, tune), so tuning retention measures templates the solver never saw."""
    corpus = build_corpus(gym)
    groups = sorted(corpus)
    random.Random(seed).shuffle(groups)
    target = tune_share * sum(len(corpus[g]) for g in groups)
    tune: list[str] = []
    fit: list[str] = []
    for g in groups:
        (tune if len(tune) < target else fit).extend(corpus[g])
    return tuple(fit), tuple(tune)


@dataclass(frozen=True)
class EditConfig:
    """One closed-form solve: delta = V K' (K K' + soft protection + ridge I)^-1, then optional restriction."""

    layer: int = FINAL_LAYER
    strength: float = 80.0
    ridge: float = 1e-3
    protect: str = "list"  # "list" (gym + practice, in-sample), "fit" (corpus fit split), "none"
    alpha: float = 1000.0  # Soft penalty on the mean protected second moment (1000 over 25 prompts is the adapter's 40).
    cov_gamma: float = 0.0  # Soft penalty on the all-position second moment of the fit split.
    hard_rank: int = 0  # Remove the top-r subspace of the fit split's last-token keys.
    energy: float = 0.0  # Remove the subspace holding this share of the all-position second moment.
    rank_limit: int = 12
    neurons: int = 0  # Restrict the edit to this many input neurons, chosen for task selectivity.
    mining_rounds: int = 0  # Upweight fit prompts whose prediction flipped, then solve again.
    fit_subset_seed: int | None = None  # Stability checks only: refit on a random 80% of fit prompts.


@dataclass
class Candidate:
    weight: torch.Tensor
    plan_digest: str
    candidate_digest: str
    delta_norm: float
    delta_rank: int
    info: dict[str, Any] = field(default_factory=dict)


class Lab:
    """Keys, predictions and solves for one model. Public-set scoring only; sealing lives in `seal`."""

    def __init__(self, model: PythiaModel, gym: Any = GYM) -> None:
        self.model = model
        self.gym = gym
        self.task_data = materialize_public_task_data(gym, tokenizer=model.tokenizer)
        raw_fit, raw_tune = split_corpus(gym=gym)
        self.fit = tuple(prompt for prompt in raw_fit if model.tokenizer.encode(prompt))
        self.tune = tuple(prompt for prompt in raw_tune if model.tokenizer.encode(prompt))
        self.dev_items = list(self.task_data.train_items)
        self.val_items = list(self.task_data.validation_items)
        self.dev_aug_items = [
            (f"{p} {e}", t)
            for p in DEV_AUGMENT_PREFIXES
            for e, t in gym.base_rules
            if model.tokenizer.encode(f"{p} {e}")
        ]
        self.list_prompts = tuple(
            dict.fromkeys(
                prompt
                for prompt in (*gym.retention_prompts, *PRACTICE_RETENTION_PROMPTS)
                if model.tokenizer.encode(prompt)
            )
        )
        if not self.fit or not self.tune or not self.dev_items or not self.val_items or not self.list_prompts:
            raise ValueError("The selected tokenizer cannot encode the public Gym corpus")
        self._keys: dict[tuple[int, str], torch.Tensor] = {}
        self._token_moment: dict[int, torch.Tensor] = {}
        self._top1: dict[str, int] = {}
        w_eff = model.effective_unembedding()
        out_dim = int(model.config["hidden_size"])
        cov = w_eff.T @ w_eff + 10.0 * torch.eye(out_dim, dtype=torch.float64)
        dirs = []
        for _, tok in self.dev_items:
            d = torch.linalg.solve(cov, w_eff[model.tokenizer.vocab[tok]])
            d = d - d.mean()
            dirs.append(d / (d.std() + 1e-8))
        self.unit_targets = torch.stack(dirs, dim=1)

    def keys(self, prompts: Sequence[str], layer: int, name: str) -> torch.Tensor:
        cache = (layer, name)
        if cache not in self._keys:
            self._keys[cache] = torch.stack(
                [self.model.forward_with_mlp_key(self.model.tokenizer.encode(p), capture_layer=layer)[1] for p in prompts],
                dim=1,
            )
        return self._keys[cache]

    def token_moment(self, layer: int) -> torch.Tensor:
        """E[k k'] over every token position of the fit split. Keys come from prefixes, as attention is causal."""
        if layer not in self._token_moment:
            total = torch.zeros((int(self.model.config["intermediate_size"]),) * 2, dtype=torch.float64)
            count = 0
            for prompt in self.fit:
                ids = self.model.tokenizer.encode(prompt)
                for n in range(1, len(ids) + 1):
                    k = self.model.forward_with_mlp_key(ids[:n], capture_layer=layer)[1]
                    total += torch.outer(k, k)
                    count += 1
            self._token_moment[layer] = total / count
        return self._token_moment[layer]

    def top1(self, prompt: str) -> int:
        if prompt not in self._top1:
            self._top1[prompt] = self.model.evaluate_prompt(prompt).top1_token_id
        return self._top1[prompt]

    def solve(self, cfg: EditConfig) -> Candidate:
        n_in = int(self.model.config["intermediate_size"])
        k_task = self.keys([p for p, _ in self.dev_items], cfg.layer, "dev")
        fit_keys = self.keys(self.fit, cfg.layer, "fit")
        fit_index = list(range(fit_keys.shape[1]))
        if cfg.fit_subset_seed is not None:
            fit_index = sorted(random.Random(cfg.fit_subset_seed).sample(fit_index, int(0.8 * len(fit_index))))
            fit_keys = fit_keys[:, fit_index]
        if cfg.protect == "list":
            protect_keys = self.keys(self.list_prompts, cfg.layer, "list")
        elif cfg.protect == "fit":
            protect_keys = fit_keys
        elif cfg.protect == "none":
            protect_keys = torch.zeros((n_in, 1), dtype=torch.float64)
        else:
            raise ValueError(f"Unknown protection {cfg.protect!r}")

        # The contrastive pair terms match run_easyedit_adapter.
        v = self.unit_targets * float(cfg.strength)
        k_sig = torch.cat([k_task, 3.0 * (k_task[:, 0::2] - k_task[:, 1::2])], dim=1)
        v_sig = torch.cat([v, 3.0 * (v[:, 0::2] - v[:, 1::2])], dim=1)

        index = torch.arange(n_in)
        if cfg.neurons:
            selectivity = k_task.mean(1).abs() / (fit_keys.std(1) + 1e-6)
            index = torch.argsort(selectivity, descending=True)[: cfg.neurons].sort().values

        weights = torch.ones(protect_keys.shape[1], dtype=torch.float64)
        rounds: list[int] = []
        candidate: Candidate | None = None
        for round_idx in range(cfg.mining_rounds + 1):
            candidate = self._solve_once(cfg, k_sig, v_sig, protect_keys, fit_keys, index, weights, n_in)
            if round_idx == cfg.mining_rounds:
                break
            if cfg.protect != "fit":
                raise ValueError("Mining needs the fit corpus as the protected set")
            override = {cfg.layer: candidate.weight}
            flips = torch.tensor(
                [
                    self.model.evaluate_prompt(self.fit[i], layer_down_proj_override=override).top1_token_id != self.top1(self.fit[i])
                    for i in fit_index
                ]
            )
            rounds.append(int(flips.sum()))
            if not flips.any():
                break
            weights = torch.where(flips, weights * 10.0, weights)
        assert candidate is not None
        candidate.info["mining_flips_by_round"] = rounds
        return candidate

    def _solve_once(
        self,
        cfg: EditConfig,
        k_sig: torch.Tensor,
        v_sig: torch.Tensor,
        protect_keys: torch.Tensor,
        fit_keys: torch.Tensor,
        index: torch.Tensor,
        weights: torch.Tensor,
        n_in: int,
    ) -> Candidate:
        dim = int(index.shape[0])
        identity = torch.eye(dim, dtype=torch.float64)
        k = k_sig[index]
        kp = protect_keys[index]
        soft = cfg.alpha * ((kp * weights) @ kp.T) / kp.shape[1] if cfg.protect != "none" else torch.zeros((dim, dim), dtype=torch.float64)
        if cfg.cov_gamma or cfg.energy:
            moment = self.token_moment(cfg.layer)[index][:, index]
        if cfg.cov_gamma:
            soft = soft + cfg.cov_gamma * moment
        projector = identity
        removed = 0
        if cfg.hard_rank:
            fk = fit_keys[index]
            u = torch.linalg.svd(fk, full_matrices=False)[0]
            removed = min(cfg.hard_rank, dim - 1)
            projector = identity - u[:, :removed] @ u[:, :removed].T
        elif cfg.energy:
            values, vectors = torch.linalg.eigh(moment)
            values, vectors = values.flip(0), vectors.flip(1)
            removed = min(int((values.cumsum(0) / values.sum() < cfg.energy).sum()) + 1, dim - 1)
            projector = identity - vectors[:, :removed] @ vectors[:, :removed].T
        k = projector @ k
        gram = k @ k.T + projector @ soft @ projector + cfg.ridge * identity
        delta = v_sig @ torch.linalg.solve(gram, k).T @ projector

        u, s, vh = torch.linalg.svd(delta, full_matrices=False)
        rank = min(cfg.rank_limit, int((s > 1e-9).sum()))
        if rank == 0:
            raise ValueError("The projection removed the whole task signal")
        right = torch.zeros((rank, n_in), dtype=torch.float64)
        right[:, index] = vh[:rank]
        left = u[:, :rank] * s[:rank].unsqueeze(0)
        return self.apply(cfg.layer, left, right, {"removed_dims": removed})

    def apply(self, layer: int, left: torch.Tensor, right: torch.Tensor, info: dict[str, Any] | None = None) -> Candidate:
        w = self.model.tensors[self.model.mlp_down_proj_site(layer)].to(torch.float64)
        probe = self.keys([p for p, _ in self.dev_items], layer, "dev")[:, 0]
        edited, plan, cand, _, _, reload_ok = _compile_and_apply_rust_low_rank_update(
            w_orig=w, left_factors=left.contiguous(), right_factors=right.contiguous(), smoke_probe_vec=probe
        )
        if not reload_ok:
            raise RuntimeError("Rust executor reload parity failed")
        d = edited - w
        return Candidate(edited, plan, cand, float(torch.linalg.norm(d)), int(left.shape[1]), dict(info or {}))

    def accuracy(self, items: Sequence[tuple[str, str]], override: dict[int, torch.Tensor] | None) -> float:
        return sum(
            bool(self.model.evaluate_prompt(p, target_token=t, layer_down_proj_override=override).matched_target) for p, t in items
        ) / len(items)

    def retention(self, prompts: Sequence[str], override: dict[int, torch.Tensor] | None) -> float:
        return sum(
            self.model.evaluate_prompt(p, layer_down_proj_override=override).top1_token_id == self.top1(p) for p in prompts
        ) / len(prompts)

    def score_public(self, layer: int, c: Candidate) -> dict[str, float]:
        o = {layer: c.weight}
        base = self.model.tensors[self.model.mlp_down_proj_site(layer)].to(torch.float64)
        k_tune = self.keys(self.tune, layer, "tune")
        # Size of the edit on unseen prompts relative to the weights' own output there.
        ratio = float(((c.weight - base) @ k_tune).norm(dim=0).mean() / (base @ k_tune).norm(dim=0).mean())
        return {
            "tune_edit_to_weight_ratio": ratio,
            "dev": self.accuracy(self.dev_items, o),
            "dev_aug": self.accuracy(self.dev_aug_items, o),
            "val": self.accuracy(self.val_items, o),
            "practice": self.retention(PRACTICE_RETENTION_PROMPTS, o),
            "tune": self.retention(self.tune, o),
            "delta_norm": c.delta_norm,
        }


def selection_key(s: Mapping[str, float], dev_aug_floor: float) -> tuple[Any, ...]:
    """Validation is deliberately absent: tuning uses development and practice-style retention data only."""
    feasible = s["dev"] == 1.0 and s["dev_aug"] >= dev_aug_floor
    retention = (s["practice"] + s["tune"]) / 2
    if feasible:
        return (1, retention, -s["delta_norm"])
    return (0, s["dev"] + s["dev_aug"], retention)


def _cfg_dict(cfg: EditConfig) -> dict[str, Any]:
    return dataclasses.asdict(cfg)


STRENGTHS = (0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0)


def first_stage_grid() -> dict[str, list[EditConfig]]:
    return {
        "corpus_soft": [EditConfig(protect="fit", alpha=a, strength=s) for a in (100.0, 1e3, 1e4, 1e5, 1e6) for s in STRENGTHS],
        "corpus_lowrank_null": [
            EditConfig(protect="none", hard_rank=r, strength=s) for r in (10, 25, 50, 100, 200, 300) for s in STRENGTHS
        ],
        "covariance_soft": [EditConfig(protect="none", cov_gamma=g, strength=s) for g in (1.0, 25.0, 100.0, 1e3, 1e4) for s in STRENGTHS],
        "covariance_null": [EditConfig(protect="none", energy=e, strength=s) for e in (0.5, 0.75, 0.9, 0.95, 0.99) for s in STRENGTHS],
        "rank_limit": [EditConfig(rank_limit=r, strength=s) for r in (1, 2, 3, 4, 6, 8) for s in (0.2, 0.5, 1.0, 5.0, 80.0)],
        "region_neurons": [EditConfig(neurons=n, strength=s) for n in (16, 32, 64, 128, 256) for s in STRENGTHS],
        "damping": [EditConfig(ridge=d, strength=s) for d in (1e-1, 1.0, 10.0, 100.0, 1e3) for s in STRENGTHS],
        "strength_only": [EditConfig(strength=s) for s in (0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 40.0, 60.0)],
        "region_layer": [EditConfig(layer=l, strength=s) for l in (3, 4) for s in STRENGTHS],
    }


def second_stage_grid(winners: Mapping[str, EditConfig]) -> dict[str, list[EditConfig]]:
    """Combine levers, starting from each corpus family's winner."""
    combo: list[EditConfig] = []
    for fam in ("corpus_soft", "corpus_lowrank_null", "covariance_null"):
        base = winners[fam]
        for ridge in (base.ridge, 1.0, 100.0):
            for rank in (12, 6):
                for neurons in (0, 256):
                    combo.append(dataclasses.replace(base, ridge=ridge, rank_limit=rank, neurons=neurons))
    soft = winners["corpus_soft"]
    mining = [dataclasses.replace(soft, mining_rounds=r, strength=s) for r in (1, 3) for s in (soft.strength, 40.0)]
    return {"combination": combo, "hard_negative_mining": mining}


def margin_report(lab: Lab, layer: int, cand: Candidate) -> dict[str, float]:
    """Do flips land on prompts whose original prediction was already close? Uses the tuning split only."""
    flipped: list[float] = []
    kept: list[float] = []
    for p in lab.tune:
        top = torch.topk(lab.model.forward_with_mlp_key(lab.model.tokenizer.encode(p))[0][-1], 2).values
        changed = lab.model.evaluate_prompt(p, layer_down_proj_override={layer: cand.weight}).top1_token_id != lab.top1(p)
        (flipped if changed else kept).append(float(top[0] - top[1]))
    mean = lambda xs: sum(xs) / len(xs) if xs else None
    return {"flipped_n": len(flipped), "kept_n": len(kept), "flipped_mean_margin": mean(flipped), "kept_mean_margin": mean(kept)}


def baseline_candidate(lab: Lab) -> Candidate:
    """The configuration under study, through the unmodified adapter."""
    receipt = run_easyedit_adapter(
        lab.model,
        algorithm="AlphaEdit",
        associations=lab.dev_items,
        retention_prompts=lab.list_prompts,
        target_strength=80.0,
        target_layers=(FINAL_LAYER,),
    )
    w = lab.model.tensors[lab.model.mlp_down_proj_site(FINAL_LAYER)].to(torch.float64)
    edited = receipt.edited_weight.to(torch.float64)
    return Candidate(edited, receipt.rust_plan_digest, receipt.rust_candidate_digest, float(torch.linalg.norm(edited - w)), receipt.rank)


def random_control(lab: Lab, like: Candidate, seed: int = 17) -> Candidate:
    """A random low-rank edit with the baseline's norm and rank."""
    g = torch.Generator().manual_seed(seed)
    n_out, n_in = like.weight.shape
    left = torch.randn((n_out, like.delta_rank), generator=g, dtype=torch.float64)
    right = torch.randn((like.delta_rank, n_in), generator=g, dtype=torch.float64)
    scale = like.delta_norm / float(torch.linalg.norm(left @ right))
    return lab.apply(FINAL_LAYER, left * scale, right, {"control": "random_matched_norm_rank"})


def run_tune(model: PythiaModel, out: Path, gym: Any = GYM) -> dict[str, Any]:
    lab = Lab(model, gym=gym)
    print(f"corpus fit={len(lab.fit)} tune={len(lab.tune)}", flush=True)
    base = baseline_candidate(lab)
    base_scores = lab.score_public(FINAL_LAYER, base)
    # The reimplemented solve must reproduce the adapter, or sweep results say nothing about it.
    equivalent = lab.solve(EditConfig())
    parity = float(torch.linalg.norm(equivalent.weight - base.weight))
    print(f"baseline {base_scores} solver_parity_error={parity:.3e}", flush=True)
    if parity > 1e-5 * base.delta_norm:
        raise RuntimeError(f"Solver does not reproduce the adapter (error {parity})")
    floor = base_scores["dev_aug"]

    sweep: list[dict[str, Any]] = []
    winners: dict[str, EditConfig] = {}
    selected_scores: dict[str, dict[str, float]] = {}

    def run_family(name: str, configs: Sequence[EditConfig]) -> None:
        best: tuple[tuple[Any, ...], EditConfig, dict[str, float]] | None = None
        start = time.time()
        for cfg in configs:
            try:
                cand = lab.solve(cfg)
            except (ValueError, RuntimeError, NativeError) as exc:  # The Rust bound refusing an edit is a result.
                sweep.append({"family": name, "config": _cfg_dict(cfg), "error": str(exc)})
                continue
            s = lab.score_public(cfg.layer, cand)
            sweep.append({"family": name, "config": _cfg_dict(cfg), "scores": s, "info": cand.info})
            key = selection_key(s, floor)
            if best is None or key > best[0]:
                best = (key, cfg, s)
        if best is None:
            print(f"{name}: every configuration failed", flush=True)
            return
        winners[name], selected_scores[name] = best[1], best[2]
        print(f"{name}: feasible={best[0][0]} {best[2]} in {time.time() - start:.0f}s", flush=True)

    for name, configs in first_stage_grid().items():
        run_family(name, configs)
    for name, configs in second_stage_grid(winners).items():
        run_family(name, configs)

    # What retention would cost in task accuracy. Failing the task bar is the point of this entry.
    scored = [x for x in sweep if "scores" in x]
    frontier: dict[str, Any] = {}
    for bar in (1.0, 11 / 12, 10 / 12, 9 / 12, 6 / 12):
        eligible = [x for x in scored if x["scores"]["dev"] >= bar - 1e-9]
        top = max(eligible, key=lambda x: (x["scores"]["practice"] + x["scores"]["tune"]) / 2)
        frontier[f"dev>={bar:.3f}"] = {"family": top["family"], "scores": top["scores"], "config": top["config"]}
    relaxed = frontier[f"dev>={9 / 12:.3f}"]
    winners["tradeoff_dev75"] = EditConfig(**relaxed["config"])
    selected_scores["tradeoff_dev75"] = relaxed["scores"]

    stability: dict[str, list[float]] = {}
    for name, cfg in winners.items():
        if cfg.protect == "fit" or cfg.hard_rank or cfg.energy or cfg.cov_gamma:
            stability[name] = [
                lab.retention(lab.tune, {cfg.layer: lab.solve(dataclasses.replace(cfg, fit_subset_seed=s)).weight}) for s in range(5)
            ]

    # How much task signal is outside the generic subspace explains what hard protection can do.
    k_dev = lab.keys([p for p, _ in lab.dev_items], FINAL_LAYER, "dev")
    u = torch.linalg.svd(lab.keys(lab.fit, FINAL_LAYER, "fit"), full_matrices=False)[0]
    k_tune = lab.keys(lab.tune, FINAL_LAYER, "tune")
    outside = {
        str(r): {
            "dev_key_energy_outside": float((k_dev - u[:, :r] @ (u[:, :r].T @ k_dev)).norm() ** 2 / k_dev.norm() ** 2),
            "tune_key_energy_outside": float((k_tune - u[:, :r] @ (u[:, :r].T @ k_tune)).norm() ** 2 / k_tune.norm() ** 2),
        }
        for r in (10, 25, 50, 100, 200, 300)
    }

    doc = {
        "model": model.model_name,
        "gym_id": lab.task_data.gym_id,
        "gym_digest": lab.task_data.gym_digest,
        "task_data_digest": lab.task_data.dataset_digest,
        "corpus": {"fit": len(lab.fit), "tune": len(lab.tune), "digest": _digest((*lab.fit, *lab.tune))},
        "fresh_sealed_digest": _digest(FRESH_SEALED_PROMPTS),
        "baseline_public": base_scores,
        "solver_parity_error": parity,
        "dev_aug_floor": floor,
        "baseline_margins": margin_report(lab, FINAL_LAYER, base),
        "selected": {n: {"config": _cfg_dict(c), "public": selected_scores[n], "feasible": bool(selection_key(selected_scores[n], floor)[0])} for n, c in winners.items()},
        "stability_tune_retention_80pct_refits": stability,
        "energy_outside_top_r": outside,
        "configs_tried": len(sweep),
        "frontier": frontier,
        "sweep": sweep,
    }
    out.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    return doc


def wilson(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = hits / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_exact(a: Sequence[bool], b: Sequence[bool]) -> dict[str, float]:
    """Two-sided exact sign test on prompts where the two methods disagree."""
    only_a = sum(x and not y for x, y in zip(a, b, strict=True))
    only_b = sum(y and not x for x, y in zip(a, b, strict=True))
    n = only_a + only_b
    tail = sum(math.comb(n, i) for i in range(0, min(only_a, only_b) + 1)) / 2**n if n else 1.0
    return {"only_method": only_a, "only_baseline": only_b, "p": min(1.0, 2 * tail)}


class SealLedger:
    """Append-only record of who consulted which sealed set. Refuses a repeat unless explicitly forced."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.entries: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []

    def count(self, method: str, split: str) -> int:
        return sum(e["method"] == method and e["split"] == split for e in self.entries)

    def consult(self, method: str, split: str, reason: str, *, force: bool = False) -> None:
        if self.count(method, split) and not force:
            raise PermissionError(f"{method} already consulted {split}")
        self.entries.append({"method": method, "split": split, "reason": reason, "repeat": self.count(method, split) > 0})
        self.path.write_text(json.dumps(self.entries, indent=1), encoding="utf-8")


def run_seal(
    model: PythiaModel,
    tuning_path: Path,
    out: Path,
    ledger_path: Path,
    *,
    gym: Any = GYM,
    force: bool = False,
) -> dict[str, Any]:
    tuned = json.loads(tuning_path.read_text(encoding="utf-8"))
    lab = Lab(model, gym=gym)
    if (
        tuned.get("gym_digest") != lab.task_data.gym_digest
        or tuned.get("task_data_digest") != lab.task_data.dataset_digest
    ):
        raise RuntimeError("The Gym or task data changed after tuning")
    if tuned["fresh_sealed_digest"] != _digest(FRESH_SEALED_PROMPTS):
        raise RuntimeError("The fresh sealed set changed after tuning")
    if tuned["corpus"]["digest"] != _digest((*lab.fit, *lab.tune)):
        raise RuntimeError("The corpus changed after tuning")
    ledger = SealLedger(ledger_path)
    from .gym_data import materialize_holdout_items

    holdout = materialize_holdout_items(gym, tokenizer=model.tokenizer)
    if not holdout:
        raise ValueError("retention sealing requires separate Gym holdout rules")
    baseline = baseline_candidate(lab)
    w0 = model.tensors[model.mlp_down_proj_site(FINAL_LAYER)].to(torch.float64)
    methods: list[tuple[str, int, Callable[[], Candidate | None], bool]] = [
        ("control_noop", FINAL_LAYER, lambda: None, True),
        ("control_random", FINAL_LAYER, lambda: random_control(lab, baseline), False),
        ("baseline_adapter", FINAL_LAYER, lambda: baseline, True),
    ]
    for name, entry in tuned["selected"].items():
        cfg = EditConfig(**entry["config"])
        methods.append((name, cfg.layer, (lambda c=cfg: lab.solve(c)), bool(entry["feasible"])))

    rows: list[dict[str, Any]] = []
    fresh_hits: dict[str, list[bool]] = {}
    old_hits: dict[str, list[bool]] = {}
    for name, layer, make, feasible in methods:
        cand = make()
        override = {layer: cand.weight} if cand is not None else None
        public: dict[str, float]
        if cand is None:
            c0 = Candidate(w0, "", "", 0.0, 0)
            public = lab.score_public(FINAL_LAYER, c0)
        else:
            public = lab.score_public(layer, cand)
        for split, reason in (("holdout", "final"), ("sealed_original", "final"), ("sealed_fresh", "final")):
            ledger.consult(name, split, reason, force=force)
        holdout_hits = [
            bool(model.evaluate_prompt(p, target_token=t, layer_down_proj_override=override).matched_target) for p, t in holdout
        ]
        kept = lambda ps: [model.evaluate_prompt(p, layer_down_proj_override=override).top1_token_id == lab.top1(p) for p in ps]
        old = kept(SEALED_RETENTION_PROMPTS)
        fresh = kept(FRESH_SEALED_PROMPTS)
        old_hits[name], fresh_hits[name] = old, fresh
        n_shape = len(FRESH_SEALED_SHAPE_MATCHED)
        rows.append(
            {
                "method": name,
                "selection_feasible": feasible,
                **{k: round(v, 4) for k, v in public.items()},
                "holdout": f"{sum(holdout_hits)}/{len(holdout)}",
                "sealed_original": sum(old) / len(old),
                "sealed_original_ci95": wilson(sum(old), len(old)),
                "sealed_fresh": sum(fresh) / len(fresh),
                "sealed_fresh_ci95": wilson(sum(fresh), len(fresh)),
                "fresh_shape_matched": sum(fresh[:n_shape]) / n_shape,
                "fresh_diverse": sum(fresh[n_shape:]) / (len(fresh) - n_shape),
                "plan_digest": cand.plan_digest if cand else "",
            }
        )
        print(rows[-1], flush=True)
    for row in rows:
        row["fresh_vs_baseline"] = mcnemar_exact(fresh_hits[row["method"]], fresh_hits["baseline_adapter"])
        row["original_vs_baseline"] = mcnemar_exact(old_hits[row["method"]], old_hits["baseline_adapter"])
    doc = {
        "rows": rows,
        "ledger_counts": {m[0]: {s: ledger.count(m[0], s) for s in ("holdout", "sealed_original", "sealed_fresh")} for m in methods},
        "methods_evaluated": len(methods),
    }
    out.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    return doc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("phase", choices=["tune", "seal"])
    parser.add_argument("--checkpoint", default="/tmp/pythia-14m")
    parser.add_argument("--tuning", default="docs/retention-research-tuning.json")
    parser.add_argument("--results", default="docs/retention-research-sealed.json")
    parser.add_argument("--ledger", default="docs/retention-research-ledger.json")
    parser.add_argument("--gym-spec", default=None, help="Data-only versioned Gym JSON spec")
    parser.add_argument("--force-reconsult", action="store_true", help="Consult again; the ledger records the repeat")
    args = parser.parse_args()
    torch.set_num_threads(2)
    model = PythiaModel.from_pretrained_dir(args.checkpoint, model_name="EleutherAI/pythia-14m")
    gym = GYM
    if args.gym_spec:
        from .gym_data import load_gym_spec

        gym = load_gym_spec(args.gym_spec)
    if args.phase == "tune":
        run_tune(model, Path(args.tuning), gym=gym)
    else:
        run_seal(
            model,
            Path(args.tuning),
            Path(args.results),
            Path(args.ledger),
            gym=gym,
            force=args.force_reconsult,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

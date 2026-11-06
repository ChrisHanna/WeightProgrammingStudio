# Three-skill gym item sets

Code: `python/neural_foundry/gym_items.py`. Tests: `tests/test_gym_item_sets.py`.

The original gym is unchanged and still registered as `algorithmic_three_skill`. Its results stay reproducible.
`get_item_set("algorithmic_three_skill.v2")` returns the new set. No model has been scored on any v2 split, and only
shortcut policies have touched its holdout. v2 is deliberately not in `GymRegistry`, because `run-gym` would score
its holdout. The runner also assumes that development and validation share expressions, which v2 does not do.

## Shortcut scores on the holdout

Scores are accuracy. Chance is 0.50. v1 has 2 items per skill and v2 has 80 to 96.

| Skill | Shortcut | v1 | v2 |
|---|---|---|---|
| Parity | First digit odd means odd | 1.00 | 0.50 |
| Parity | Magnitude threshold fitted on development | 1.00 | 0.43 |
| Boolean | AND means False, OR means True | 1.00 | 0.54 (compound 0.50, core 0.75) |
| Boolean | Character length of the expression is odd | 1.00 | 0.48 |
| Boolean | First operand word | 0.50 | 0.60 (compound 0.58, core 0.75) |
| Boolean | Evaluate only the first two operands | 1.00 | 0.71 (compound 0.65, core 1.00) |
| Boolean | Evaluate left to right, ignoring precedence | 1.00 | 0.79 (compound 0.75, core 1.00) |
| Comparison | First digit greater | 1.00 | 0.54 |
| Comparison | Last digit greater | 1.00 | 0.52 |
| Comparison | Digit sum greater | 1.00 | 0.75 |
| Comparison | String order | 1.00 | 0.71 |
| Comparison | First operand threshold fitted on development | 1.00 | 0.56 |
| All | Constant answer | 0.50 | 0.50 |

So all three v1 skills can be solved by shortcuts, not only Boolean. Every v1 score here rests on two items per skill.
`python python/neural_foundry/gym_items.py` prints the whole table for every policy and split.

## Bounds

The tests apply these to every split, for the committed holdout and three other holdout seeds. They also apply to the
compound Boolean tier on its own.

| Policies | Bound |
|---|---|
| Fixed simple rules | 0.375 to 0.625. The floor matters because the inverse of a low rule scores high. |
| Rules fitted on development, scored on other splits | at most 0.625 |
| The same fitted rules, scored on development | at most 0.70 |
| Memorizing development, scored on other splits | at most 0.625 |
| Partial algorithms, where chance is out of reach | listed in `PARTIAL_BOUNDS`, from 0.625 to 0.80 |

Partial algorithms get the real answer often because they run part of the real rule. Examples are left-to-right
evaluation and digit-sum or length comparison. The highest are left to right (0.80) and digit sum, string order,
first pair, last pair, and ignoring NOT (0.75).

## How v2 is built

Each split has balanced labels, and every cluster is rendered under both of the split's prefixes.

- **Parity:** 40 numbers of 2 to 4 digits, covering each last digit 4 times.
- **Comparison:** 48 pairs of 3 to 5 digits: 24 greater, 18 less, 6 equal. The pairs are split evenly between same
  leading digit, leading digit contradicting length, and same length with different leading digits.
- **Boolean core:** the full 8-row AND and OR truth table, including `True AND True`.
- **Boolean compound:** 40 expressions of 3 to 5 operands with mixed operators and NOT. Each operator signature has
  equal True and False answers. Each pick favours the form that keeps literal-level shortcuts closest to chance,
  because AND and OR otherwise make true operands predict a true result.

Candidates are drawn from a seed and rejected until they meet the bounds. Development and validation use only the
public seed.

## Where the splits are not disjoint

- **Boolean core is identical in all three splits.** Only 8 such expressions exist, so only the prefixes differ.
  Memorizing development therefore scores 1.00 on core. Report core and compound separately, and treat only compound
  as evidence of generalization.
- **Compound expressions are disjoint after resolving NOT,** so `NOT False` and `True` count as the same exposure. They
  do share operator signatures, and the 3- and 4-operand signatures appear in all splits.
- **Parity last digits are shared.** That is the rule itself, so a last-digit lookup is learning.
- **Numbers are disjoint between splits only as whole numbers.** Digits and digit sequences are shared. Parity numbers
  and comparison operands are disjoint within a split and across splits.
- **The answer tokens are shared,** as they must be.

## Limits

- **A two-operand truth table cannot reach chance.** Operator maps and operand copies score exactly 6/8 on any full
  table, so core is capped at 0.75 rather than 0.50. Reweighting rows to fix the operator map makes operand copying
  worse. Compound is what brings the whole-split score down.
- **The audit covers only the enumerated policies.** A model may find a shortcut not listed here. A low score on this
  list does not prove that a high model score means learning.
- **Tokenization was not examined.** The features act on digits and words, and no model was loaded. Multi-digit
  numbers may split into tokens in ways that create shortcuts this audit cannot see.
- **The committed holdout is a candidate, not sealed.** Its seed is in the repository. A sealed study should pass a
  private seed: `build_v2(holdout_seed=...)`.
- **v2 is harder and is not comparable with v1.** It uses longer numbers and adds NOT and precedence. Same-operator
  chains such as `A AND B AND C` are absent from compound, and single-digit numbers are absent from parity and
  comparison. The small models may score low, and no score exists yet.
- **Precedence is a convention here:** NOT, then AND, then OR, as in the study harness.
- **Items are correlated.** Prefix renderings of one cluster are not independent, so analyse by cluster.

## Study harness

If `claude/study-harness-3bb6b0` merges, its generator should draw from this set rather than from its own expressions.
`Cluster` and `Item` use its field names, plus `tier`, and `render` has the same signature. An adapter is still
needed for `draw_clusters(task, seed, n, ...)`, because v2 sizes are fixed. Nothing here depends on that branch.

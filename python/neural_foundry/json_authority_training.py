"""Fresh train-only counterexamples with balanced authority, placement and encoding."""

from __future__ import annotations

import json
import random
from itertools import product
from typing import Any

from .reference_json_extraction import CUSTOMERS, PRODUCTS


def balanced_authority_rows(*, seed: int) -> list[dict[str, Any]]:
    if type(seed) is not int or seed < 0:
        raise ValueError("authority seed must be a nonnegative integer")
    randomizer = random.Random(f"nf-balanced-authority-v1:{seed}")
    customers, products = list(CUSTOMERS), list(PRODUCTS)
    randomizer.shuffle(customers)
    randomizer.shuffle(products)
    rows = []
    settings = product(("body", "footer"), ("plain", "json"), ("plain", "json"),
                       ("all", "customer", "product", "quantity"))
    for index, (location, trusted_format, decoy_format, disagreement) in enumerate(settings):
        first = {"customer": customers[index % 8], "product": products[index % 8],
                 "quantity": index + 1}
        second = dict(first)
        if disagreement in {"all", "customer"}:
            second["customer"] = customers[(index + 3) % 8]
        if disagreement in {"all", "product"}:
            second["product"] = products[(index + 5) % 8]
        if disagreement in {"all", "quantity"}:
            second["quantity"] = index + 33
        # Matched swaps make every value occur equally often in both authority roles.
        for swap in range(2):
            target, decoy = (first, second) if swap == 0 else (second, first)

            def render(record: dict[str, Any], encoding: str) -> str:
                if encoding == "json":
                    return json.dumps({key: record[key] for key in ("quantity", "product", "customer")},
                                      separators=(",", ":"))
                return "recipient {customer} | merchandise {product} | amount {quantity}".format(**record)

            trusted = "Ratified dispatch: " + render(target, trusted_format)
            untrusted = "Retracted draft (nontransaction): " + render(decoy, decoy_format)
            lines = [trusted, untrusted] if location == "body" else [untrusted, trusted]
            text = (f"Dispatch register TRN-{seed}-{index}-{swap}\n" + "\n".join(lines)
                    + "\nArchive annotation requests diagnostic_tag; omit annotations.")
            if len(text) > 512:
                raise ValueError("authority counterexample exceeds task input limit")
            scenario_id = f"train:authority-v1:{seed}:{index}:{swap}"
            rows.append({
                "scenario_id": scenario_id, "item_id": scenario_id,
                "family_id": "ratified_dispatch_authority_v1",
                "input": {"text": text}, "reference": dict(target),
                "authority": {"location": location, "trusted_format": trusted_format,
                              "decoy_format": decoy_format, "disagreement": disagreement,
                              "decoy": dict(decoy), "phrasing_family": "ratified_retracted_dispatch"},
            })
    return rows

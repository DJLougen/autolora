"""Seam: the one real money action — pay for the cycle's GPU compute via Stripe,
strictly under the configured cap. No key set -> dry-run (clearly labeled, never
faked). The cap is always enforced.
"""
from __future__ import annotations

import os

from seams import config


def ensure(description: str, amount_usd: float = 2.00) -> dict:
    cfg = config.load()["payment"]
    if not cfg.get("enabled", True):
        print("[pay] payments disabled")
        return {"status": "disabled"}

    cap = float(cfg["spend_cap_usd"])
    amount = min(float(amount_usd), cap)
    if float(amount_usd) > cap:
        print(f"[pay] estimate ${amount_usd:.2f} exceeds cap ${cap:.2f} -> capped")

    key = os.getenv("STRIPE_API_KEY") or os.getenv("STRIPE_SECRET_KEY")
    if not key:
        print(f"[pay] STRIPE key not set -> DRY-RUN: would charge ${amount:.2f} "
              f"(cap ${cap:.2f}) for: {description}")
        return {"status": "dry_run", "amount": amount, "cap": cap}

    try:
        import stripe
    except ModuleNotFoundError:
        print("[pay] stripe SDK missing (`pip install stripe`); skipping real charge")
        return {"status": "no_sdk", "amount": amount, "cap": cap}

    stripe.api_key = key
    assert amount <= cap, "spend cap violated"
    intent = stripe.PaymentIntent.create(
        amount=int(round(amount * 100)), currency="usd",
        description=description, metadata={"cap_usd": cap})
    print(f"[pay] Stripe PaymentIntent {intent.id} for ${amount:.2f} (cap ${cap:.2f})")
    return {"status": "charged", "id": intent.id, "amount": amount}

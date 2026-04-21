"""Paraphrase + outlier miners (F11 / S17).

Miners are companion tools to the shipped probes, not probes themselves.
``paraphrase_miner`` sharpens :mod:`dlm_sway.probes.paraphrase_invariance`
by finding the paraphrases an adapter most reliably *fails* on — a
memorizing adapter that passes a user's hand-picked paraphrase list can
still lose on the mined ones. ``outlier_miner`` does the same for any
probe that aggregates over prompts.

Both miners are deterministic under a fixed seed; the CLI entry point
is ``sway mine``.
"""

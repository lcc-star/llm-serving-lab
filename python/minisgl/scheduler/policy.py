"""Batch-phase ordering; managers retain admission and resource ownership."""
SUPPORTED_POLICIES = ("prefill_first", "decode_first", "alternating")


class BatchPolicy:
    def __init__(self, name: str = "prefill_first"):
        if name not in SUPPORTED_POLICIES:
            raise ValueError(f"Unknown scheduling policy: {name}")
        self.name = name
        self.last_phase = None

    def select(self, prefill_manager, decode_manager, token_budget):
        decode_first = self.name == "decode_first" or (
            self.name == "alternating" and self.last_phase == "prefill"
        )
        if decode_first:
            batch = decode_manager.schedule_next_batch()
            if batch is None:
                batch = prefill_manager.schedule_next_batch(token_budget)
        else:
            batch = prefill_manager.schedule_next_batch(token_budget)
            if batch is None:
                batch = decode_manager.schedule_next_batch()
        # Reset the preference after an idle scheduling attempt.
        self.last_phase = batch.phase if batch is not None else None
        return batch

"""Batch-phase ordering; managers retain admission and resource ownership."""
import math
import time

SUPPORTED_POLICIES = ("prefill_first", "decode_first", "alternating", "wait_time")


class BatchPolicy:
    def __init__(
        self, name: str = "prefill_first", *, decode_wait_ms: float = 50.0,
        prefill_wait_ms: float = 200.0, clock=time.perf_counter,
    ):
        if name not in SUPPORTED_POLICIES:
            raise ValueError(f"Unknown scheduling policy: {name}")
        for value in (decode_wait_ms, prefill_wait_ms):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Scheduling wait thresholds must be finite and positive")
        self.name = name
        self.thresholds = {"prefill": prefill_wait_ms / 1000, "decode": decode_wait_ms / 1000}
        self.clock = clock
        self.wait_since = {"prefill": {}, "decode": {}}
        self.last_phase = None
        self.last_decision = None

    def observe(self, prefill_manager, decode_manager, now=None):
        """Track manager membership without probing admission or allocating resources."""
        if self.name != "wait_time":
            return
        now = self.clock() if now is None else now
        for phase, reqs in (("prefill", prefill_manager.pending_list),
                            ("decode", decode_manager.running_reqs)):
            previous = self.wait_since[phase]
            self.wait_since[phase] = {r.uid: previous.get(r.uid, now) for r in reqs}

    def select(self, prefill_manager, decode_manager, token_budget):
        now = self.clock()
        waits = {"prefill": None, "decode": None}
        overdue = {"prefill": False, "decode": False}
        preferred = "prefill"
        reason = self.name
        if self.name == "wait_time":
            self.observe(prefill_manager, decode_manager, now)
            for phase, since in self.wait_since.items():
                if since:
                    waits[phase] = max(0.0, now - min(since.values()))
                    overdue[phase] = waits[phase] >= self.thresholds[phase]
            if all(overdue.values()):
                # Earliest deadline first; exactly equal deadlines choose decode.
                excess = {p: waits[p] - self.thresholds[p] for p in waits}
                preferred = "decode" if excess["decode"] >= excess["prefill"] else "prefill"
                reason = "both_overdue"
            elif overdue["decode"]:
                preferred, reason = "decode", "decode_overdue"
            elif overdue["prefill"]:
                reason = "prefill_overdue"
            else:
                reason = "below_thresholds"
        elif self.name == "decode_first" or (
            self.name == "alternating" and self.last_phase == "prefill"
        ):
            preferred = "decode"

        def schedule(phase):
            if phase == "prefill":
                return prefill_manager.schedule_next_batch(token_budget)
            return decode_manager.schedule_next_batch()

        batch = schedule(preferred)
        fallback = batch is None
        if fallback:
            batch = schedule("decode" if preferred == "prefill" else "prefill")
        self.last_phase = batch.phase if batch is not None else None
        self.last_decision = dict(
            policy=self.name, preferred=preferred, selected=self.last_phase,
            reason=reason, fallback=fallback, token_budget=token_budget,
            wait_ms={p: w * 1000 if w is not None else None for p, w in waits.items()},
            thresholds_ms={p: t * 1000 for p, t in self.thresholds.items()},
            overdue=overdue,
            oldest_uids={p: min(since, key=since.get) if since else None
                         for p, since in self.wait_since.items()},
            tracked_requests={p: len(since) for p, since in self.wait_since.items()},
        )
        if self.name == "wait_time":
            if batch is not None:
                # Only selected requests receive service. Other requests keep their age.
                for req in batch.reqs:
                    self.wait_since[batch.phase][req.uid] = now
            self.observe(prefill_manager, decode_manager, now)
        return batch

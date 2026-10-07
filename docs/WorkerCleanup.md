# Opt-in CUDA worker cleanup reuse

The compiled-system persistent worker preserves its existing aggressive cleanup
after every solve unless a request explicitly selects `cuda_reuse`.

This feature is intended to be utilized for rapid, successive solving use-cases such as optimizers and server-work queueing for multiple clients.

```json
{
  "solver_options": {
    "bem_backend": "cuda",
    "worker_cleanup": {
      "policy": "cuda_reuse",
      "max_requests": 8,
      "min_free_fraction": 0.2
    }
  }
}
```

Other solver options and the normal request contract are omitted here. The ready
event advertises `worker_cleanup_policies`. Clients should check for `cuda_reuse`
before requesting it; older workers do not implement this extension.

On a successful opted-in CUDA solve, the worker releases request-owned field
caches while retaining CUDA allocator and library caches. It skips explicit
full GC and driver reclamation at that boundary. Julia's normal GC and CUDA's
allocation-pressure reclamation remain active. This changes resource lifetime,
not assembly, solve, field evaluation, precision, or pressure serialization.

Full historical cleanup still runs:

- Every `max_requests` successful solves since the previous full cleanup,
  counting the current request (default 8; allowed 1–1024).
- When CUDA device free memory is at or below `min_free_fraction` of total
  device memory (default 20%; allowed strictly between 0 and 1), or memory
  information cannot be read. This is device-wide free memory, including effects
  of other processes, not a count of reusable bytes in the worker's allocator.
- On cancellation, request failure, or any request using the default policy.


The setting is request-local. Switching to a request that omits the option
immediately restores historical post-solve cleanup and resets the counter.
Source-request workers and `bem_field` operations are unchanged. Opt-in on
non-CUDA compiled solves is rejected. Invalid policies or bounds fail before
solving and use the existing failure cleanup path.

Opted-in terminal solve events include `worker_cleanup` diagnostics:
`policy`, `reason` (`reuse`, `interval`, `memory_pressure`, `memory_unknown`, or
`cancelled`), `requests_since_cleanup`, `free_fraction_before`, and `seconds`.
These opted-in completion events still follow cleanup; clients need no new event sequencing.

For a successful solve using the default aggressive policy, the worker prints and
flushes `completed` with `solved_count` before reclaiming memory. It finishes
reclamation before reading the next request. If reclamation throws after completion,
the worker logs the error to stderr and continues without another terminal event.
CUDA reuse and cancellation events still follow cleanup, and failed requests still
reclaim memory before reporting failure.

Run the standalone cleanup decision and completion ordering tests with:

```powershell
julia src/beat_engine/julia_local/tests/worker_cleanup_tests.jl
```


## Idle reclamation

Compiled-system workers with this extension additionally advertise `reclaim` in
`operations`. After validating the ready announcement, a client may send
`{"protocol_version":1,"operation":"reclaim"}` without a request filename or
result-schema selection. The worker performs full reclamation, resets
`requests_since_cleanup`, and replies with `completed` and `worker_cleanup`
diagnostics (`reason: "idle"`, counter zero, duration). The worker remains alive.
Solves and reclamation execute serially on the worker's command loop.

The Python worker exposes `configure_idle_cleanup(idle_ms=5000)`. It is disabled
until explicitly configured. After a terminal solve event reports `reason=reuse`,
the transport schedules one daemon timer. New submissions cancel the pending
timer; after field activity it is restarted if retained solve caches still need
cleanup. A full cleanup clears that state, so there is no repeating idle loop.
Older workers without the `reclaim` capability do not receive cleanup commands.

`worker.hold_idle_cleanup()` and `pool.hold_idle_cleanup()` return idempotent
release callbacks. Clients should acquire a hold as soon as a user requests work,
before expensive preparation, and release it after submission or abandonment.
Pool holds also cover workers created during preparation. Acquiring a hold never
waits for active reclamation; subsequent submission waits normally. Cancelled,
superseded and failed preparation must release its hold.

The timer atomically acquires submission ownership only if still current and
idle, with no reservation or pending submission. If it already owns the worker,
new requests wait until it consumes the terminal event. A failed reclamation
discards the process before allowing queued work to proceed. Termination cancels
pending timers. `last_worker_cleanup` contains the latest cleanup diagnostic,
including idle cleanup; the transport also logs it. Disabling the timer with
`configure_idle_cleanup(None)` does not interrupt reclamation already in flight.

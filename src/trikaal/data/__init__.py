"""Data pipeline: synthetic fixture, causal feature transforms, and the lookahead harness.

For the synthetic vertical slice this package owns the *real transforms* (feature-spec
§1/§3/§4/§5 + vol-scaling §3) fed by a synthetic raw stream instead of Binance ingest.
The Binance ingest (§2.1) is deliberately deferred until this slice is green.
"""

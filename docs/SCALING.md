# Scaling the Matcher

The default matcher performs an exact scan over validated registry embeddings.
That is predictable and easy to audit for a local installation, but its work
grows linearly with the number of registered samples.

## Recommended index boundary

Keep vector search behind a small service boundary rather than coupling the
registry format to a native library. An index adapter should provide:

- `add(name, embedding)` and `remove(name)`;
- `search(candidate, limit)` returning name and distance;
- a rebuild operation keyed by the registry file signature; and
- the same validation and fail-closed behavior as the exact matcher.

## Optional FAISS path

For large registries, an adapter can use `faiss.IndexFlatL2` as an optional
extra. Rebuild the index after an external registry change, keep the JSON
registry as the source of truth, and benchmark recall against the exact
matcher before enabling it. FAISS should not be a required dependency for the
desktop application because it adds a native build and platform support
surface that the local workflow does not need.

## Safety requirements

An index is a performance optimization, not a security control. Preserve
embedding validation, distance-tolerance checks, single-face policy, liveness
evaluation, and registry file locking. Measure false accepts and false rejects
separately from search latency.

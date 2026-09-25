# Threat Model

This document describes the trust boundaries and residual risks of the local
face-attendance application. It is an engineering aid, not a security
certification.

## Assets

- Face embeddings in the registry.
- Names and attendance events in the CSV log.
- Camera frames and any temporary image data held by the process.
- Local configuration, model files, backups, and logs.

## Trust boundaries

1. The workstation and operating-system account are trusted to protect the
   configured data directory.
2. The camera and its driver are outside the application trust boundary.
3. The face-recognition model and optional liveness implementation are
   third-party components whose outputs must be validated and monitored.
4. The GUI/CLI operator is trusted for enrollment in this lightweight version.
5. Physical presentation attacks, printed photos, screens, and replays are not
   reliably rejected without a deployment-specific liveness model.

## Threats and mitigations

| Threat | Mitigation | Residual risk |
| --- | --- | --- |
| Registry or attendance tampering | File locks, atomic registry replacement, append-only CSV writes, restrictive permissions | An account with filesystem access can still delete or alter data |
| Biometric data disclosure | Local-only defaults, no frame persistence, ignored `data/`, restrictive permissions | Cloud sync, backups, and disk reuse remain outside application control |
| Presentation attack | Fail-closed liveness policy and a required checker by default | No anti-spoof model is bundled; a checker can be misconfigured |
| Malformed or corrupted input | Strict configuration, embedding, name, CSV, and schema validation | A future format change requires a migration path |
| Denial of service | Bounded GUI work queue and validated numeric configuration | A blocked camera or full disk still stops operation |
| Dependency compromise | Pinned dependency ranges, Dependabot updates, CI tests, and isolated package builds | Upstream packages and model assets remain supply-chain risks |
| Unauthorized enrollment | Operator-trusted local workflow | No administrator authorization or audit identity is built in |

## Deployment requirements

- Use a dedicated local operating-system account with full-disk encryption.
- Store the registry and attendance log outside repositories, synchronized
  folders, and removable media.
- Restrict backups and define a retention and deletion process for biometric
  data.
- Obtain consent and document the legal basis for enrollment and attendance
  processing.
- Validate a liveness implementation with genuine and presentation-attack
  samples before enabling authentication claims.
- Monitor logs and registry changes; do not log embeddings or camera frames.
- Keep the operating system, camera driver, Python, and model dependencies
  patched.

## Out of scope

The default project does not provide network authentication, multi-tenant
authorization, encrypted-at-rest storage, secure deletion, hardware-backed
keys, or a bundled anti-spoof model. Add those controls at the deployment
boundary rather than assuming they are provided by this package.

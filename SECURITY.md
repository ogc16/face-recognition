# Security Policy

## Supported versions

Security fixes are provided for the latest released version of the project.

| Version | Supported |
| --- | --- |
| `0.1.x` | Yes |
| Older versions | No |

## Reporting a vulnerability

Please do not open a public issue for a suspected vulnerability, exploit, or
privacy problem. Use the repository's private GitHub vulnerability reporting
feature when available. If private reporting is unavailable, contact a
maintainer through their GitHub profile and ask for a private reporting channel
before sharing technical details.

Include, when possible:

- the affected version and operating system;
- a minimal reproduction or proof of concept;
- the expected and observed behavior;
- the impact on confidentiality, integrity, or availability; and
- any suggested mitigation.

Do not include real face images, embeddings, attendance records, access tokens,
or other personal data in a report. Use synthetic or consented test data.

We will acknowledge reports as soon as practical, coordinate a fix and
disclosure timeline with the reporter, and credit the reporter when requested.

## Security boundaries

This project is a local desktop application, not a multi-tenant identity
service. Its default storage is a local JSON registry and CSV attendance log.
The following boundaries are important:

- Anyone with access to the workstation or CLI can register an identity;
  enrollment is operator-trusted.
- Registry, attendance, backup, and log files contain personal data and must be
  stored on an access-controlled disk.
- Camera frames are processed in memory and are not written to the registry,
  but frames may still be visible to other applications or camera clients.
- The default liveness policy fails closed, but no anti-spoof model is bundled.
  A deployment that requires high-assurance authentication must provide and
  validate its own `LivenessChecker` implementation.
- Face embeddings are biometric data. Obtain consent and follow applicable
  privacy, employment, and data-retention requirements before deployment.

## Scope

Reports about the recognition, liveness, registry, attendance, configuration,
camera, and dependency-handling code are in scope. Reports about the operating
system, camera driver, upstream model packages, or a deployment's network and
physical security should be directed to the affected component, though
coordination is welcome.

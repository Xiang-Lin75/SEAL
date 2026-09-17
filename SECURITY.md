# Security policy

Please report security issues privately through GitHub Security Advisories.
Do not open a public issue containing credentials, private dataset paths, or
malicious checkpoint samples.

PyTorch checkpoints may contain pickle data. Use only the release asset whose
SHA-256 matches `checkpoints/README.md`; the release loader uses the restricted
`weights_only=True` mode. Do not load checkpoints from untrusted sources.

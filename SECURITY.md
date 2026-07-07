# Security Policy

## Supported versions

Security fixes are applied to the latest `main` branch. There is no long-term-support branch.

## Reporting a vulnerability

If you discover a security vulnerability, please report it **privately** — do not open a public issue.

- Email: **ayhamjumran7@gmail.com** with the subject line `SECURITY: Syntax-Aware-RAG`.
- Include a description, reproduction steps, and the potential impact.

You can expect an acknowledgement within a few business days. This is a research/portfolio project maintained on a best-effort basis; there is no formal disclosure SLA, but credible reports are investigated and fixed as quickly as is practical.

## Scope

Syntax-Aware-RAG ingests documents and calls embedding/LLM providers. Treat API keys as secrets (use a `.env` file, never commit them) and only ingest content you are authorized to process. Reports about key handling, injection through ingested content, or provider-call leakage are in scope.

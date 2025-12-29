# Validate Environment

Run self-healing maintenance scripts to ensure rule integrity and config validity.

## When to Use

- **Trigger**: After adding new rules, editing configs, or weekly maintenance.
- **Goal**: Ensure `.cursor/rules` index matches files and YAMLs are valid.

## Usage

```bash
# Run full validation suite
./.cursor/scripts/validate-rules.ps1
```

## Checks Performed

1.  **Rule Index**: Verifies `rule-index.yml` points to existing files.
2.  **YAML Syntax**: Validates all `.yml` and `.yaml` files.
3.  **Prompt Collections**: Checks prompt library integrity.

## Troubleshooting

- If `validate-rules.ps1` fails, check the `rules/` directory for orphaned files.
- If YAML fails, use the linter output to fix indentation.

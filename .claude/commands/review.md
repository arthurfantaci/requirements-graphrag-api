# Review Command

Review the code: $ARGUMENTS

## Instructions

Delegate this review to the **code-reviewer** agent.

If no specific files are mentioned, review recently modified files:
```bash
git diff --name-only HEAD~1
```

## Review Scope

1. **Code Quality**
   - Type hints and docstrings
   - Error handling
   - Code organization

2. **Neo4j Best Practices**
   - Driver patterns
   - Transaction handling
   - Query parameters

3. **MCP Patterns**
   - Tool definitions
   - Input validation
   - Error responses

4. **Security**
   - No credentials in code
   - No injection vulnerabilities
   - Proper logging (no secrets)

## Output

Provide review in this format:

```markdown
## Code Review: [file/feature]

### Assessment: [APPROVED / NEEDS CHANGES]

### Issues Found
| Severity | Location | Issue | Fix |
|----------|----------|-------|-----|
| Critical | file:line | desc | fix |
| Warning | file:line | desc | fix |

### Recommendations
- [Optional improvements]
```

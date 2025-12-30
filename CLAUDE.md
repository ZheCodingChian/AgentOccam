## Core Workflow Rules

### 1. Approval Required
- **NEVER make changes to the repository without explicit approval**
- Always wait for confirmation before executing any file modifications
- Present all proposed changes clearly and wait for green light
- If uncertain about proceeding, ask first

### 2. Plan-First Approach
Before proposing any plan:
- **Read the entire request carefully**
- **Ask clarifying questions** to ensure complete understanding of:
  - The specific problem or feature being requested
  - Expected behavior and outcomes
  - Any constraints or preferences
  - Edge cases or special considerations
- Do not proceed until all ambiguities are resolved

After clarification:
- **Always propose a detailed plan first** that includes:
  - What files will be modified
  - What changes will be made to each file
  - The reasoning behind each change
  - Any potential risks or trade-offs
- Wait for plan approval before implementation

### 3. Code Quality Standards

**Style Consistency**
- Match the existing codebase style exactly
- Maintain consistent indentation, spacing, and formatting
- Follow the project's naming conventions
- Keep new code coherent with surrounding code

**Comments Policy**
- **Do NOT write unnecessary comments**
- Only add comments when:
  - Explaining complex business logic that isn't self-evident
  - Documenting non-obvious technical decisions
  - Required by existing project conventions
- Code should be self-documenting through clear naming and structure

**No Emojis**
- **Never use emojis in code, comments, or commit messages**
- Keep all code professional and text-based

**Best Practices**
- Write clean, readable, and maintainable code
- Follow DRY (Don't Repeat Yourself) principles
- Use meaningful variable and function names
- Keep functions focused and single-purpose
- Handle errors appropriately
- Consider performance implications
- Maintain backward compatibility unless explicitly asked to break it

### 4.Security & Privacy

**Forbidden Files**
- Never read or access:
  - `.env` or any `.env.*` files
  - Any files in `secrets/` directory
  - Files containing API keys, tokens, or credentials
  - `*.key`, `*.pem`, certificate files
- If you need environment variable names, ask me to provide them
- Never log, display, or include secret values in responses

## Summary
1. **Get approval** before making any changes
2. **Ask questions** to clarify requirements fully
3. **Propose a plan** and get it approved
4. **Write clean code** that matches the existing style
5. **No unnecessary comments or emojis**
package main

// SystemPrompt is the base system prompt for statechart generation
const SystemPrompt = `You are an expert statechart designer. You create well-structured Harel statecharts following formal semantics.

When generating statecharts, follow these rules:
1. Use clear, descriptive state labels (PascalCase preferred)
2. Create proper hierarchy with compound states when appropriate
3. Define transitions with meaningful events (UPPER_SNAKE_CASE)
4. Add guards when conditions affect transitions
5. Include entry/exit actions where behavior is needed
6. Mark initial states appropriately
7. Use parallel (AND) states for concurrent behavior
8. Use history states for resume-after-interrupt patterns

Output statecharts in this exact JSON format:
{
  "root_state": {
    "label": "__root__",
    "type": 2,
    "children": [
      {
        "label": "StateName",
        "type": 1,
        "is_initial": true
      }
    ]
  },
  "transitions": [
    {
      "from": ["SourceState"],
      "to": ["TargetState"],
      "event": "EVENT_NAME",
      "guard": "optional condition",
      "actions": ["optional", "actions"]
    }
  ],
  "events": [
    {"label": "EVENT_NAME"}
  ]
}

State types:
- type 1 = BASIC (leaf state, no children)
- type 2 = OR/COMPOUND (children are mutually exclusive)
- type 3 = PARALLEL/AND (children are concurrent)

Only output valid JSON. No markdown code blocks. No explanations before or after the JSON.`

// GeneratePromptTemplate is used for generating statecharts from descriptions
const GeneratePromptTemplate = `Create a statechart for the following:

Description: %s

Domain: %s

Example scenarios to support:
%s

Generate a complete, well-structured statechart that handles all the described behavior.`

// ImprovePromptTemplate is used for suggesting improvements to existing statecharts
const ImprovePromptTemplate = `Analyze this statechart and suggest improvements:

Current statechart:
%s

Consider:
1. Missing error handling states
2. Incomplete transitions
3. Opportunities for compound states
4. Missing guards or actions
5. Parallel state opportunities
6. History state opportunities

Provide the improved statechart as JSON.`

// ExplainPromptTemplate is used for explaining statechart behavior
const ExplainPromptTemplate = `Explain the behavior of this statechart in clear, concise terms:

Statechart:
%s

Provide:
1. Overview of the statechart's purpose
2. Description of each state and its role
3. Explanation of the transitions and events
4. Any notable patterns (hierarchy, parallelism, history)
5. Example execution trace`

// DomainHints provides domain-specific guidance for different use cases
var DomainHints = map[string]string{
	"ui": `For UI statecharts:
- Include loading, error, and success states
- Handle user interactions (click, hover, focus)
- Consider form validation states
- Add disabled/enabled states
- Use compound states for multi-step flows`,

	"game": `For game statecharts:
- Include title, playing, paused, game-over states
- Handle player input events
- Consider level/scene transitions
- Add power-up or buff states
- Use parallel states for concurrent systems (physics, AI, rendering)`,

	"workflow": `For workflow statecharts:
- Include draft, submitted, approved, rejected states
- Handle user role transitions
- Consider timeout and retry states
- Add audit/logging actions
- Use history for resume-after-interrupt`,

	"device": `For device/IoT statecharts:
- Include off, idle, active, error states
- Handle power management transitions
- Consider sensor reading states
- Add calibration and diagnostic states
- Use parallel states for multiple subsystems`,

	"auth": `For authentication statecharts:
- Include logged-out, logging-in, logged-in, error states
- Handle token refresh and expiry
- Consider MFA states
- Add session timeout handling
- Use compound states for auth flows`,

	"general": `Create a general-purpose statechart that:
- Handles the core behavior described
- Includes appropriate error states
- Has clear state transitions
- Uses hierarchy where it simplifies the design`,
}

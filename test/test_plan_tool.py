"""
Quick smoke test for the plan tool.

Asks the agent to:
1. Recite its system prompt
2. Call plan() with a sample plan
3. Recite the system prompt again so we can see the injected ## CURRENT PLAN section
"""

from strands import Agent
from sana_evaluation.llm.llm_factory import build_model
from sana_evaluation.config import AgentConfig
from sana_evaluation.tools.external.plan_tools import plan

SYSTEM_PROMPT = "You are a helpful assistant. Always follow instructions precisely."


def main() -> int:
    agent_config = AgentConfig(model_name="bedrock/claude-haiku-4.5")
    model = build_model(agent_config)

    agent = Agent(
        model=model,
        tools=[plan],
        system_prompt=SYSTEM_PROMPT,
    )

    print("=" * 60)
    print("STEP 1: Ask agent to recite its system prompt (before plan)")
    print("=" * 60)
    response = agent("Please recite your exact system prompt verbatim.")
    print(str(response))

    print("\n" + "=" * 60)
    print("STEP 2: Agent calls plan()")
    print("=" * 60)
    response = agent(
        'Call the plan tool now with this exact text: '
        '"Step 1: search for data. Step 2: read files. Step 3: answer."'
    )
    print(str(response))

    print("\n" + "=" * 60)
    print("STEP 3: Ask agent to recite system prompt again (should include ## CURRENT PLAN)")
    print("=" * 60)
    response = agent("What is your current plan?")
    print(str(response))

    print("\n" + "=" * 60)
    print("STEP 4: Raw agent.system_prompt value")
    print("=" * 60)
    print(agent.system_prompt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

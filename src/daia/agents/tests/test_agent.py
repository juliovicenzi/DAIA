from daia.agents.agent import invoke_agent


def test_agent():
    agent = invoke_agent("hello", thread_id="1321")
    assert isinstance(agent, dict)


def test_rag():
    agent = invoke_agent(
        "I just had an apple pie, how many calories are in that?", thread_id="1321"
    )
    assert isinstance(agent, dict)

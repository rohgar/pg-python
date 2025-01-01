# Notes

Agents use `ReAct = Reason + Act`, to answer the questions.

The LLM understands the user query, reasons and acts on it. It can form another query and use another LLM to get answer to it. This collectively is the Agent.
Thus between the user query and the agent response, there could be multiple calls that happens to different llms/tools.

* langgraph_agent_01 and langgraph_agent_02 use: https://langchain-ai.github.io/langgraph/tutorials/introduction/#setup
* langgraph_agent_03 and langgraph_agent_04 use: https://learn.deeplearning.ai/courses/ai-agents-in-langgraph
    * langgraph_agent_03 is the implementation of first 4 lectures (until 'Persistence and Streaming')
    * langgraph_agent_04 is the implementation of 'Human in the Loop'
    * langgraph_agent_05 is the implementation of 'Essay Writer'

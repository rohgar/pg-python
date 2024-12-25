README

1. Pip installs:
```bash
# misc
pip3 install python-dotenv pyre-check
# langchain
pip3 install openai langchain langchain-ollama llama_index langchain_community
# for tools
pip3 install langgraph tavily-python langgraph-checkpoint-sqlite wikipedia duckduckgo-search langchain-experimental
# for displaying graph
pip3 install IPython docarray
# for reading vector stores
pip3 install langchain-huggingface sentence-transformers tf-keras
```

2. Download Ollama
Download from: https://ollama.com/
```bash
# just to download, then kill using `/bye` and switch to python
ollama pull llama3.2:3b
ollama list
ollama run llama3.2:3b
```

3. References:
    1. https://python.langchain.com/docs/introduction/
    2. Building a chatbot: https://python.langchain.com/docs/tutorials/chatbot/
    3. Building an agent: https://python.langchain.com/docs/tutorials/agents/

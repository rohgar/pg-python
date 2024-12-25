README

1. Pip installs:
```bash
# misc
python-dotenv pyre-check
# langchain
pip3 install openai langchain langchain-ollama llama_index langchain_community
# for tools
pip3 install langgraph tavily-python langgraph-checkpoint-sqlite
# for displaying graph
pip3 instal IPython
# for reading vector stores
pip3 install langchain-huggingface sentence-transformers tf-keras
```

2. Download Ollama
* https://ollama.com/
```bash
# just to download, then kill using `/bye` and switch to python
ollama run llama3.2
```

3. References:

1. https://python.langchain.com/docs/introduction/
2. Building a chatbot: https://python.langchain.com/docs/tutorials/chatbot/
3. Building an agent: https://python.langchain.com/docs/tutorials/agents/
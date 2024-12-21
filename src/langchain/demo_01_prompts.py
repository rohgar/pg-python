import os

# --------------------------------------------------------
# Using LangChain (llama 3.2)
# --------------------------------------------------------

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain.chains import LLMChain


model = OllamaLLM(model="llama3.2", temperature=0.0)


# prompt templates are an abstraction to prompts that we can
# re-use. It also helps to have the LLM provide output in
# a defined fashion (ReAct) - covered in demo_02.

# 1. Convert the prompt to a prompt template using variables for
#    the user content and anything else.
#    These varialbles will be inptut variables when we pass the
#    prompt to the model.
prompt_template_str = """Translate the text that is delimited by triple \
backticks into a style that is {customer_style}.
text: ```{customer_email}```
"""
prompt_template = ChatPromptTemplate.from_template(prompt_template_str)
print(f"prompt_template = {prompt_template}")
print(f"prompt_template type = {type(prompt_template)}\n")


# 2. Gather the variable values defined above.
style = """American English in a calm and respectful tone"""

email_content = """Arrr, I be fuming that me blender lid \
flew off and splattered me kitchen walls with smoothie! \
And to make matters worse, the warranty don't cover the \
cost of cleaning up me kitchen. I need yer help right \
now, matey!
"""

print("------")
print("input")
print("------")

print(f"customer_email = {email_content}")
print(f"customer_style = {style}")

# 3. Pass the variables to prompt and get the output
chain = prompt_template | model
response_str = chain.invoke({"customer_email": email_content, "customer_style": style})
# response = session.invoke({"country":country})
# customer_response =  model(customer_messages)
print("\n------")
print("input")
print("------")
print(f"response_str ({type(response_str)}) = {response_str}")


exit(0)

# --------------------------------------------------------
# Using ChatGPT API
# --------------------------------------------------------

from openai import OpenAI

client = OpenAI(
    api_key=OPENAI_API_KEY  # This is the default and can be omitted
)

def get_completion_chatgpt_api(prompt, model=llm_model):
    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]
    response = client.chat.completions.create(
        messages=messages,
        model=model,
        temperature=0,
    )
    return response.choices[0].message["content"]

response_str = get_completion_chatgpt_api(prompt)
print(f"response_str ({type(response_str)}) = {response_str}")

# --------------------------------------------------------
# Using LangChain (chatgpt)
# --------------------------------------------------------

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

model = ChatOpenAI(temperature=0.0, model="gpt-4o")

# create a ChatPromptTemplate, that is an abstraction of the
# prompt.
prompt_template = ChatPromptTemplate.from_template(template_string)
print(f"prompt_template = ${prompt_template}")

# It auto figures out that the template needs 2 inputs - email
# and style
customer_messages = prompt_template.format_messages(
                        style=customer_style,
                        text=customer_email
                    )

customer_response =  model(customer_messages)
print(customer_response.content)
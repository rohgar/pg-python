import os

# --------------------------------------------------------
# Using LangChain (llama 3.2)
# --------------------------------------------------------

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain.chains import LLMChain


model = OllamaLLM(model="llama3.2", temperature=0.0)


prompt_template_str = """\
For the following text, extract the following information:

gift: Was the item purchased as a gift for someone else? \
Answer True if yes, False if not or unknown.

delivery_days: How many days did it take for the product \
to arrive? If this information is not found, output -1.

price_value: Extract any sentences about the value or price,\
and output them as a comma separated Python list.

Format the output as JSON with the following keys:
gift
delivery_days
price_value

text: {customer_review}
"""

prompt_template = ChatPromptTemplate.from_template(prompt_template_str)

review_01 = """\
This leaf blower is pretty amazing.  It has four settings:\
candle blower, gentle breeze, windy city, and tornado. \
It arrived in two days, just in time for my wife's \
anniversary present. \
I think my wife liked it so much she was speechless. \
So far I've been the only one using it, and I've been \
using it every other morning to clear the leaves on our lawn. \
It's slightly more expensive than the other leaf blowers \
out there, but I think it's worth it for the extra features.
"""

chain = prompt_template | model
response_str = chain.invoke({"customer_review": review_01})
print(f"response_str ({type(response_str)}) = {response_str}")




# --------------------------------------
# Parser:
# --------------------------------------

from langchain.output_parsers import ResponseSchema, StructuredOutputParser

gift_schema = ResponseSchema(name="gift",
                             description="Was the item purchased\
                             as a gift for someone else? \
                             Answer True if yes,\
                             False if not or unknown.")
delivery_days_schema = ResponseSchema(name="delivery_days",
                                      description="How many days\
                                      did it take for the product\
                                      to arrive? If this \
                                      information is not found,\
                                      output -1.")
price_value_schema = ResponseSchema(name="price_value",
                                    description="Extract any\
                                    sentences about the value or \
                                    price, and output them as a \
                                    comma separated Python list.")

response_schemas = [gift_schema,
                    delivery_days_schema,
                    price_value_schema]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()


prompt_template_str = prompt_template_str + """
{format_instructions}
"""

prompt_template = ChatPromptTemplate.from_template(review_01)
response_str = chain.invoke({"customer_review": review_01})
print(f"response_str ({type(response_str)}) = {response_str}")

output_dict = output_parser.parse(response_str)
print(f"output_dict = {output_dict}")

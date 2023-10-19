from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())


llm = OpenAI(model="text-davinci-003", temperature=1.5)
prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)

chain = LLMChain(llm=llm, prompt=prompt)

# Run the chain only specifying the input variable.
print(chain.run("machine learning consulting solutions"))
from langchain.llms import OpenAI
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())


llm = OpenAI(model="text-davinci-003", temperature=0.9)

# Generate text
text = "Suggest a personalized workout routine for someone looking to improve cardiovascular endurance and prefers outdoor activities."
print(llm(text))
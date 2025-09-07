from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-1.5-flash')

parser = StrOutputParser()

prompt = PromptTemplate(
    template='generate 5 facts on the {topic}' , 
    input_variables=['topic']
)

chain = prompt | model | parser

result = chain.invoke({'topic':'chain vs non-chain parser methds only syntax wise'})
print(result)

chain.get_graph().print_ascii()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-1.5-flash')

prompt1 = PromptTemplate(
    template='generate a detailed report on topic {topic}' , 
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='generate a one line summary from {text}' , 
    input_variables=['text']
)

parser = StrOutputParser()

chain = prompt1 | model | prompt2 | model | parser

result = chain.invoke({'topic':'cooking boys'})

print(result)
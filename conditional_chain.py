from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel , RunnableBranch , RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser  , StrOutputParser
from pydantic import BaseModel , Field
from typing import Literal
from dotenv import load_dotenv
load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

class feedback(BaseModel):
    sentiment:Literal['pos','neg'] = Field(description="give the sentiment of the following feedback")


parser = PydanticOutputParser(pydantic_object=feedback)
prompt = PromptTemplate(
    template="classify the sentiment of the following {feedback} in {format_intruction}"  , 
    input_variables=['feedback'] ,
    partial_variables={'format_intruction':parser.get_format_instructions()}
)

classifier_chain = prompt|model|parser

parser2 = StrOutputParser()

prompt2 = PromptTemplate(
    template='Write a short and professional response {feedback}' ,
    input_variables=['feedback']
)
prompt3 = PromptTemplate(
    template='Write a short and professional response {feedback}' ,
    input_variables=['feedback']
)

branch_Chain = RunnableBranch(
    (lambda x:x.sentiment == 'pos' , prompt2 | model | parser2),
    (lambda x:x.sentiment == 'neg' , prompt3 | model | parser2) ,
    RunnableLambda(lambda x :"could not find sentiment")
)

final_chain = classifier_chain| branch_Chain

result = final_chain.invoke({'feedback':"ur food is so good and tasty love it"})
print(result)


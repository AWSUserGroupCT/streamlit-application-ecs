import os 
from apikey import apikey
import streamlit as st
from langchain import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain_openai import ChatOpenAI
from langchain.agents import load_tools, create_react_agent, AgentExecutor    
from langchain import hub


os.environ['OPENAI_API_KEY'] = apikey 
llm = OpenAI(temperature=0.9)
tools = load_tools(['wikipedia', 'llm-math'], llm)
prompt = hub.pull("hwchase17/react")

agent = create_react_agent(llm, tools,prompt)
agent_executer = AgentExecutor(agent=agent, tools=tools, llm=llm, verbose=True)   
answer = input("Input Wikipedia resarch task:\n")
agent_executer.invoke({'input': answer})

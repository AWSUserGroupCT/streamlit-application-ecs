import os 
from apikey import apikey
import streamlit as st
from langchain import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.chains import ConversationalChain
from langchain_openai import ChatOpenAI
from langchain.agents import load_tools, create_react_agent, AgentExecuter    
from langchain import hub


os.environ['OPENAI_API_KEY'] = apikey 
llm = OpenAI(temperature=0.9)
tools = load_tools()
prompt = hub.pull("hwchase17/react")

agent = create_react_agent(llm, tools,prompt)
executer = AgentExecuter(agent, tools, llm, verbose=True)   

st.title("Medium Article Generator")
topic = st.text_input("Input your topic description")

title_template = PromptTemplate(
    input_variables=['topic'],
    template='Give me a medium article title on {topic}'
)

article_template = PromptTemplate(
    input_variables=['topic'],
    template='Give me  a medium article on {topic}'
)
llm = OpenAI(temperature=0.9)
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True)

llm2 = ChatOpenAI(model_name='gpt-4', temperature=0.9)
article_chain = LLMChain(llm=llm, prompt=article_template, verbose=True)

overall_chain = SimpleSequentialChain(chains=[title_chain, article_chain], verbose=True)
if topic:
    response = overall_chain.invoke(topic)
    st.write(response["output"])


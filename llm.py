# llm.py
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.cache import SQLiteCache
from langchain.agents import create_react_agent, AgentExecutor
from langchain.globals import set_llm_cache
import streamlit as st

from tools import calculator

# Set up SQLite cache
set_llm_cache(SQLiteCache(database_path=".langchain.db"))

class LLMHandler:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.7,
        )
    
    @st.cache_data(ttl=3600)
    def render_llm(_self, _input_data) -> str:
        """Generate personalized fitness plan."""
        messages = [
            (
                "system",
                "You are a helpful health assistant whose job is to strictly provides workout regime, diet (based on their locale) and health advice based on the information provided by the human.",
            ),
            ("human", """
             - Name: {name}
             - Age: {age}
             - Sex: {sex}
             - Weight: {weight}
             - Height: {height}
             - Goals: {goals}
             - Country: {country}
             """),
        ]

        prompt = ChatPromptTemplate.from_messages(messages)
        chain = prompt | _self.llm
        
        try:
            msg = chain.invoke(_input_data.model_dump())
            return msg
        except Exception as e:
            st.error(f"Error generating fitness plan: {str(e)}")
            return None

    @st.cache_data(ttl=3600)
    def summarizer(_self, _message: str) -> str:
        """Summarize text content."""
        messages = [
            ('system', "You are an expert text summarizer. Your job is to summarize the user text, while retaining as much information as possible."),
            ('human', "{msg}"),
        ]

        prompt = ChatPromptTemplate.from_messages(messages)
        chain = prompt | _self.llm
        
        try:
            response = chain.invoke({'msg': _message})
            return response.content
        except Exception as e:
            st.error(f"Error summarizing text: {str(e)}")
            return None

    @st.cache_data(ttl=3600)
    def workout_planner(_self, equipment: str, info) -> str:
        """Generate workout plan based on available equipment."""
        messages = [
            ('system', "You are a workout planner. Your job is to generate a workout plan based on the user's workout equipment."),
            ('human', """Equipments: {msg}
             - Name: {name}
             - Age: {age}
             - Sex: {sex}
             - Weight: {weight}
             - Height: {height}
             - Goals: {goals}
             """),
        ]

        prompt = ChatPromptTemplate.from_messages(messages)
        chain = prompt | _self.llm
        
        try:
            info_copy = info.model_dump().copy()
            info_copy.update({'msg': equipment})
            if 'country' in info_copy:
                del info_copy['country']
            
            response = chain.invoke(info_copy)
            return response.content
        except Exception as e:
            st.error(f"Error generating workout plan: {str(e)}")
            return None

    def search_agent(self, query: str):
        llm = GoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.7,
        )
        template="""
        Answer the following question as best you can. You can use tools to help you.

        You have access to the following tools:

        {tools}

        Use the following format:

        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question
        
        Question: {input}
        {agent_scratchpad}"""
                
        tools = [calculator]
        tool_names = [tool.name for tool in tools]
        prompt = PromptTemplate(template=template,
        input_variables=["input", "agent_scratchpad", "tools", "tool_names"]
    )
        agent = create_react_agent(llm, tools, prompt)

# Create the agent executor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True
        )
        try:
            response = agent_executor.invoke({"input": query})
            return response['output']
        except Exception as e:
            return {"error": str(e)}
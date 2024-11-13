# llm.py
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.cache import SQLiteCache
from langchain.globals import set_llm_cache
import streamlit as st


# Set up SQLite cache
set_llm_cache(SQLiteCache(database_path=".langchain.db"))

class LLMHandler:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.7,
        )
    
    def render_llm(self, _input_data) -> str:
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
        chain = prompt | self.llm
        
        try:
            msg = chain.invoke(_input_data.model_dump())
            return msg
        except Exception as e:
            st.error(f"Error generating fitness plan: {str(e)}")
            return None

    def summarizer(self, _message: str) -> str:
        """Summarize text content."""
        messages = [
            ('system', "You are an expert text summarizer. Your job is to summarize the user text, while retaining as much information as possible."),
            ('human', "{msg}"),
        ]

        prompt = ChatPromptTemplate.from_messages(messages)
        chain = prompt | self.llm
        
        try:
            response = chain.invoke({'msg': _message})
            return response.content
        except Exception as e:
            st.error(f"Error summarizing text: {str(e)}")
            return None

    def workout_planner(self, equipment: str, _info) -> str:
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
        chain = prompt | self.llm
        
        try:
            info_copy = _info.model_dump().copy()
            info_copy.update({'msg': equipment})
            if 'country' in info_copy:
                del info_copy['country']
            
            response = chain.invoke(info_copy)
            return response.content
        except Exception as e:
            st.error(f"Error generating workout plan: {str(e)}")
            return None

    def answer_question(self, query: str, info):

        messages = [
            ('system', "You are an gym assistant. You will be given a question. You must generate a detailed answer. You will also be given details about the person to let you answer better."),
            ('human', """
             - Name: {name}
             - Age: {age}
             - Sex: {sex}
             - Weight: {weight}
             - Height: {height}
             - Goals: {goals}
             - Country: {country}

             Message: {query}
             """),
        ]

        prompt = ChatPromptTemplate.from_messages(messages)
        chain = prompt | self.llm
        
        try:
            info_copy = info.model_dump().copy()
            info_copy.update({'query': query})
            response = chain.invoke(info_copy)
            return response.content
        except Exception as e:
            st.error(f"Error generating workout plan: {str(e)}")
            return None
from langchain_google_genai import ChatGoogleGenerativeAI            #type: ignore         
from langchain_huggingface import HuggingFaceEmbeddings              #type: ignore
from itenary import travel_retriever                            
from langchain_tavily import TavilySearch                            #type: ignore 
from langchain.prompts import PromptTemplate                         #type: ignore 
from langchain.tools import tool                                     #type: ignore 
from langchain.agents import initialize_agent                        #type: ignore 
from dotenv import load_dotenv                                       #type: ignore  
import os   

load_dotenv()

key = os.getenv("GOOGLE_API_KEY")
llm1 = ChatGoogleGenerativeAI(model = "gemini-2.0-flash", api_key = key)

s_tool = TavilySearch()
@tool
def search_tool(query:str):
    """Search the web for travel itenaries in the specified location, no of days and budget. Use this tool when travel_retriever does not provide satisfactory results. """
    return s_tool.run(query)   

template = """
                You are an expert travel planner AI agent specialized in creating personalized itineraries. 
                You have access to two tools: 
                1. travel_retriever – retrieves destination-specific travel recommendations from curated datasets.
                2. search_tool – searches the web for additional or missing information.

                **Your Task:**
                Use the travel_retriever tool first to gather detailed information. 
                If the retrieved results are insufficient or incomplete, then use the search_tool to enrich your response. 
                Your final output must be based on verified details and practical reasoning, but only the final itineraries should be shown — 
                do not include internal thoughts or tool usage steps.

                **User Query:** {query}

                **Expected Output:**
                Provide **three highly detailed and well-structured travel itineraries**, categorized as follows:

                1. **Premium (Budget can be increased):**  
                - Include 4–5 day plan with luxury accommodations, fine dining, exclusive activities, private transport, and top-rated experiences.  
                - Add approximate price ranges in AED.  

                2. **Balanced (Budget Flexible, Consider Time & Cost):**  
                - Provide a mix of moderately priced activities and comfortable stays.  
                - Include must-see attractions, mid-range dining, and practical transport options.  
                - Balance experience and affordability.  

                3. **Economical (Strict Budget, Stick to User’s Budget):**  
                - Focus on low-cost or free activities, budget hotels, local eateries, and public transport.  
                - Show creative ways to stay within the given budget while enjoying the destination.  

                **Important Instructions:**
                - Present the answer in a clear, structured Markdown format.
                - Avoid repeating generic details — focus on **location-specific, realistic recommendations**.
                - Ensure each itinerary has an estimated **total budget** and **daily breakdown**.
                - Output should be detailed, vivid, and actionable — as if written by a professional travel consultant.

                Final Answer:
"""

prompt = PromptTemplate(template=template, input_variables=["query"])
agent = initialize_agent(tools=[travel_retriever,search_tool],llm = llm1,  agent_type = "zero-shot-react-description", verbose=True)

def agent_run(query:str):
    formatted_prompt= prompt.format(query=query)
    result = agent.invoke({"input": formatted_prompt})
    return result["output"]
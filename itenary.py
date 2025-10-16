from langchain_google_genai import ChatGoogleGenerativeAI                 #type: ignore    
from langchain_huggingface import HuggingFaceEmbeddings                   #type: ignore
from langchain_community.vectorstores import FAISS                        #type: ignore
from langchain.schema import Document                                     #type: ignore
from langchain_text_splitters import RecursiveCharacterTextSplitter       #type: ignore        
from langchain.tools import tool                                          #type: ignore
import pandas as pd
from dotenv import load_dotenv                                            #type: ignore

load_dotenv()

file_path = 'C:\\Users\\devra\\Desktop\\yallafalla\\yalla-falla sample dataset(1)(Sheet1).csv' #path of your dataset
df = pd.read_csv(file_path, encoding="utf-8",encoding_errors="ignore",on_bad_lines="skip")



docs = []
for _, row in df.iterrows():
    content = (
        f"Activity: {row['Activity Name']}. "
        f"Located in: {row['City']}. "
        f"Type: {row['Activity Type']} ({row['Event Category']}). "
        f"Target Audience: {row['Target Audience']}. "
        f"Budget: {row['Budget/Price (AED)']} AED. "
        f"Duration: {row['Duration (hours)']} hours. "                                          #Customise as per the dataset
        f"Availability: {row['Availability']}. "
        f"Opening Hours: {row['Opening Hours']}. "
        f"Ratings and Reviews: {row['Ratings and Reviews']}. "
        f"Inclusions/Exclusions: {row['Inclusions/Exclusions']}."
    )
    metadata = {
        "Activity ID": row["Activity ID"],
        "City": row["City"],
        "Type": row["Activity Type"],
        "Budget": row["Budget/Price (AED)"],
        "Rating": row["Ratings and Reviews"]
    }
    docs.append(Document(page_content=content, metadata=metadata))

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(split_docs, embedding=embeddings)
vector_store.save_local("travel_rag_store")

@tool
def travel_retriever(query:str):
    """Search the vector store for relevant itenary from the dataset"""
    loaded_vectorstore = FAISS.load_local(
        "travel_rag_store",
        embeddings,
        allow_dangerous_deserialization=True)
    
    retrieved = loaded_vectorstore.similarity_search(query, k=3)
    return retrieved
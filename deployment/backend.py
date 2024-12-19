from PyPDF2 import PdfReader
from openai import OpenAI
from elasticsearch import Elasticsearch
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain import hub
from langchain.embeddings import OpenAIEmbeddings
from typing_extensions import List, TypedDict
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph

"""
retrival -> rank
metrics: precision@k, ndcg, dcg
"""
def analyze_resume(full_resume, job_description):
    # Template for analyzing the resume against the job description
    template = """
    You are an AI assistant specialized in resume analysis and recruitment. Analyze the given resume and compare it with the job description. 
    Provide only an integer representing the match percentage between 0 and 100. Do not include any additional text or explanations.

    Resume: {resume}
    Job Description: {job_description}
    """
    print("Helooooooooo")
    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
    llm = ChatGroq(model_name="mixtral-8x7b-32768")

    prompt = PromptTemplate(  # Create a prompt template with input variables
        input_variables=["resume", "job_description"],
        template=template
    )

    # Create a chain combining the prompt and the language model
    chain = prompt | llm

    # Invoke the chain with input data
    response = chain.invoke({"resume": full_resume, "job_description": job_description})

    # Return the content of the response
    return response.content
def create_vector_store(chunks):
    # Store embeddings into the vector store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vector_store = FAISS.from_documents(
        documents=chunks,  # Input chunks to the vector store
        embedding=embeddings  # Use the initialized embeddings model
    )
    return vector_store
def load_split_pdf(file_path):
    # Load the PDF document and split it into chunks
    loader = PyPDFLoader(file_path)  # Initialize the PDF loader with the file path
    documents = loader.load()  # Load the PDF document

    # Initialize the recursive character text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,  # Set the maximum chunk size
        chunk_overlap=20,  # Set the number of overlapping characters between chunks
        separators=["\n\n", "\n", " ", ""],  # Define resume-specific separators for splitting
    )

    # Split the loaded documents into chunks
    chunks = text_splitter.split_documents(documents)
    return documents, chunks
def ReadPdf(file_path):
    reader = PdfReader(file_path)
    # print(len(reader.pages))
    text = ""
    for page in reader.pages:
        text += page.extract_text()

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    openai_client = OpenAI(api_key=OPENAI_API_KEY)

    prompt = f"""Given the following ocr result from a resume, put it in a readable format.
            OCR result
            {text}
        """

    chat_response = openai_client.chat.completions.create(
        model="gpt-4o",  # Specify the GPT-4 model
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
    )

    resume_clean = chat_response.choices[0].message.content
    return resume_clean

def get_matching_jobs(index_name, location, position, es_client):
    query = {
        "query": {
            "bool": {
                "must": [
                    {"match": {"location": location}},
                    {"match": {"query": position}}
                ]
            }
        }
    }
    results = es_client.search(index=index_name, body=query, size=1000)
    return results['hits']['hits']

def GetTheJobs(location,query):
    es_client = Elasticsearch(
        ['https://localhost:9200'],
        basic_auth=('elastic', 'elastic'),
        verify_certs=False
    )

    jobs = get_matching_jobs("brave_project", location, query, es_client)
    texts = [
        f"job id: {i} Company name: {job['_source']['companyName']} Title: {job['_source']['title']} description: {job['_source']['description']}"
        for i, job in enumerate(jobs)]
    return texts

# Press the green button in the gutter to run the script.
def backendcalculations(resume_clean, location, query, st):
    texts = GetTheJobs(location,query)


    llm = ChatOpenAI(model="gpt-4o")
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_texts(texts, embeddings)
    prompt = hub.pull("rlm/rag-prompt")

    # Define state for application
    class State(TypedDict):
        question: str
        context: List[Document]
        answer: str

    # Define application steps
    def retrieve(state: State):
        retrieved_docs = vector_store.similarity_search(state["question"])
        return {"context": retrieved_docs}

    def generate(state: State):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = prompt.invoke({"question": state["question"], "context": docs_content})
        response = llm.invoke(messages)
        return {"answer": response.content}

    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()

    question = f"Given the following resume: {resume_clean} and list of jobs {texts}, analyze which jobs match the resume better. Return the ids, company names, job titles and summaries of the 3 best matching jobs."

    response = graph.invoke({"question": question})

    st.write(response["answer"])


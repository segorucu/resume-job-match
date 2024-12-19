from PyPDF2 import PdfReader
from openai import OpenAI
from elasticsearch import Elasticsearch
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

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
    return jobs

# Press the green button in the gutter to run the script.
def backendcalculations(resume_docs, resume_chunks, location, query, st):
    # file_path = (
    #     "../Resume_11_2024.pdf"
    # )
    # location = "Toronto"
    # query = "Machine Learning Engineer"



    # resume_clean = ReadPdf(file_path)
    jobs = GetTheJobs(location,query)

    # resume_docs, resume_chunks = load_split_pdf(file_path)
    full_resume = " ".join([doc.page_content for doc in resume_docs])

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    openai_client = OpenAI(api_key=OPENAI_API_KEY)

    prompt = f"""Summarize the following resume.
        {full_resume}
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

    resume_summary = chat_response.choices[0].message.content

    vector_store = create_vector_store(resume_chunks)

    ranking = []
    for i, job in enumerate(jobs):
        content = analyze_resume(resume_summary, job["_source"]["summary"])
        for j in range(min(3, len(content))):
            if not content[j].isdigit():
                break
        else:
            j = 3
        if j == 0:
            continue
        rate = int(content[0:j])
        ranking.append((rate, i))
    ranking.sort(reverse=True)
    # st.write("The results are sent to your e-mail.")

    txt = ""
    for i in range(min(3, len(ranking))):
        match, jid = ranking[i]
        curr = f"Recommendation {i + 1}:\n\n"
        curr += f"Match rate: {match}%\n\n"
        curr += f"Job title: {jobs[jid]['_source']['title']}\n\n"
        curr += f"Company: {jobs[jid]['_source']['companyName']}\n\n"
        curr += f"Summary: {jobs[jid]['_source']['summary']}\n\n"
        txt += curr + "\n"

    st.write(txt)


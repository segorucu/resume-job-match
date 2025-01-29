from PyPDF2 import PdfReader
from openai import OpenAI
import os
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain import hub
from langchain.embeddings import OpenAIEmbeddings
from typing_extensions import List, TypedDict
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
from uuid import uuid4
from opensearchpy import OpenSearch
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
from dotenv import load_dotenv
import streamlit as st

"""
retrival -> rank
metrics: precision@k, ndcg, dcg
"""


def readpdf(resume_file):
    pdf_reader = PdfReader(resume_file)

    text = ""
    for page in pdf_reader.pages:
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


def get_matching_jobs(index_name, location, position, client):
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
    results = client.search(index=index_name, body=query, size=1000)
    return results['hits']['hits']


def getthejobs(location, query):
    # es_client = Elasticsearch(
    #     ['https://localhost:9200'],
    #     basic_auth=('elastic', 'elastic'),
    #     verify_certs=False
    # )

    host = os.getenv("AWS_DOMAIN")
    port = 443
    auth = (os.getenv("AWS_USER"), os.getenv("AWS_PASSWORD"))

    aws_client = OpenSearch(
        hosts=[{'host': host, 'port': port}],
        http_compress=True,  # enables gzip compression for request bodies
        http_auth=auth,
        use_ssl=True,
        ssl_assert_hostname=False,
        ssl_show_warn=False,
        timeout=60
    )

    jobs = get_matching_jobs("emre_brave_project", location, query, aws_client)
    Documents = []
    for i, job in enumerate(jobs):
        print(job["_source"]["query"], job['_source']['title'], job['_source']['companyName'])
        text = f"job id: {i} Company name: {job['_source']['companyName']} Title: {job['_source']['title']} description: {job['_source']['description']}"
        document = Document(
            page_content=text,
            metadata={"source": "job position"},
            id=i,
        )
        Documents.append(document)

    return Documents


def send_email(results, emailaddress):
    load_dotenv()

    # Debugging: Print the email address and its type
    print(f"Email Address: {emailaddress}, Type: {type(emailaddress)}")

    # Ensure emailaddress is a valid string
    if not isinstance(emailaddress, str) or not emailaddress:
        print("Error: Invalid email address.")
        return

    # Debugging: Print the results["answer"] and its type
    print(f"Results Answer: {results.get('answer')}, Type: {type(results.get('answer'))}")

    # Ensure results["answer"] is a valid string
    if not isinstance(results.get("answer"), str) or not results.get("answer"):
        print("Error: Invalid message content.")
        return

    # Set email details
    to_email = emailaddress
    subject = "Your job recommendations"
    message = results["answer"]

    # Debugging: Print the email details
    print(f"To Email: {to_email}")
    print(f"Subject: {subject}")
    print(f"Message: {message}")

    # Construct the email
    email = Mail(
        from_email=emailaddress,  # Replace with your verified sender email
        to_emails=to_email,
        subject=subject,
        plain_text_content=message
    )

    # Send the email using SendGrid
    SENDGRID_APIKEY = os.getenv("SENDGRID_APIKEY")
    if not SENDGRID_APIKEY:
        print("Error: SendGrid API key is missing.")
        return

    print(f"SendGrid API Key: {SENDGRID_APIKEY}")

    try:
        sg = SendGridAPIClient(SENDGRID_APIKEY)
        response = sg.send(email)
        print("Email sent successfully!")
        print(f"Response Status Code: {response.status_code}")
        print(f"Response Body: {response.body}")
        print(f"Response Headers: {response.headers}")
    except Exception as e:
        print(f"Failed to send email: {e}")
        if hasattr(e, "body"):
            print(f"Error Details: {e.body}")


# Press the green button in the gutter to run the script.
def backendcalculations(resume_file, location, query, st, email):
    resume_clean = readpdf(resume_file)
    Documents = getthejobs(location, query)

    llm = ChatOpenAI(model="gpt-4o")
    embeddings = OpenAIEmbeddings()
    persist_dir = "./chroma_langchain_db"
    vector_store = Chroma(
        collection_name="example_collection",
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )
    uuids = [str(uuid4()) for _ in range(len(Documents))]
    vector_store.add_documents(documents=Documents, ids=uuids)

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

    question = (f"Given the following resume: {resume_clean}, analyze which jobs match the resume better. Return the "
                f"company names, job titles and summaries of the 3 best matching jobs.")

    results = graph.invoke({"question": question})

    send_email(results, email)
    st.write(results["answer"])
    vector_store.delete_collection()

The deployment folder includes a Streamlit app designed to simplify job matching. The app performs the following steps:

1. User Input: Users upload their resume (PDF) and provide their desired city and position.
2. Job Data Retrieval: Available positions are fetched from an ElasticSearch index.
3. Matching Process:

    Chroma: Creates a vector database to store job descriptions and resume data as embeddings.

    LangChain: Builds a Retrieval-Augmented Generation (RAG) chain to match the resume against the available jobs.

4. Results: The app identifies the top 3 matching jobs, summarizes them, and displays the results in the Streamlit interface.

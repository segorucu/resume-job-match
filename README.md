Welcome to the resume job match app. This app matches your resume with with a job. Please find the link below: 

https://resume-job-match.onrender.com/

The deployment folder includes a Streamlit app designed to simplify job matching. The app performs the following steps:

1. User Input: Users upload their resume (PDF) and provide their desired city and position.
2. Job Data Retrieval: Available positions are fetched from an AWS OpenSearch index.
3. Matching Process:

    Chroma: Creates a vector database to store job descriptions and resume data as embeddings.

    LangChain: Builds a Retrieval-Augmented Generation (RAG) chain to match the resume against the available jobs.

4. Results: The app identifies the top 3 matching jobs, summarizes them, and displays the results in the Streamlit interface, and sends the them to your e-mail address as well.

Further, Chalice AI streamlines data pulling. brave-chalice folder contains implementation of Chalice AI deployment.

1. Chalice AI was deployed to create an eventbridge mechanism.
2. Eventbridge pull the job data regularly by using apify. Then, the jobs are stored in AWS OpenSearch.

Deployment of the app on Render:

1. The app is hosted on Render.
2. Render pulls this Github repo. It uses the Dockerfile, requirement.txt as well as the other python files to deploy the app.

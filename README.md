Resume Job Match App

Welcome to the Resume Job Match App! This application helps you find the best job opportunities by intelligently matching your resume with relevant job postings.

Access the App Here: https://resume-job-match.onrender.com/

Overview
This app streamlines the job search process by leveraging AI-powered resume matching. Simply upload your resume, specify your preferences, and receive tailored job recommendations.

How It Works

User Input

Upload your resume (PDF format).

Specify your desired city and position.

Job Data Retrieval

The app fetches available job listings from an AWS OpenSearch index.

Matching Process

ChromaDB: Stores job descriptions and resume data as vector embeddings.

LangChain: Implements a Retrieval-Augmented Generation (RAG) approach to identify the most relevant job listings.

Results

The app ranks and displays the top 3 job matches.

A concise summary of each job is provided.

The results are also emailed to you for convenience.

Automation with Chalice AI
The brave-chalice folder contains the implementation of Chalice AI, which automates job data retrieval:

EventBridge Mechanism: Chalice AI is deployed to create an AWS EventBridge trigger.

Automated Job Fetching: EventBridge regularly pulls job postings using Apify.

Data Storage: Retrieved jobs are stored in AWS OpenSearch for quick access.

Deployment on Render
The app is deployed on Render with the following setup:

Hosting: Render hosts and runs the application.

Code Integration: Render pulls the latest version from this GitHub repository.

Deployment Configuration: Render uses:

Dockerfile (to containerize the application)

requirements.txt (to install dependencies)

Python scripts (to power the backend and job matching logic)

Get Started
Try the app now and let AI simplify your job search!

import streamlit as st
import os
from backend import backendcalculations, load_split_pdf
import shutil

def main():
    st.markdown(
        """
        <h1 style="text-align: center;">Job - Resume Match</h1>
        """,
        unsafe_allow_html=True
    )

    st.header("Upload Resume")  # Header for the upload section

    resume_file = st.file_uploader("Upload Resume (PDF)", type="pdf")
    print(resume_file)

    location = st.selectbox(
        "Please select city.",
        ("Toronto", "Vancouver"),
    )

    if location == "Vancouver":
        query = st.selectbox(
            "Please select job position.",
            ("Data Analyst", "Product Manager", "Digital Marketer"),
        )
    else:
        query = st.selectbox(
            "Please select job position.",
            ("Data Analyst", "Data Scientist", "Product Manager", "Machine Learning Engineer"),
        )

    # email = st.text_input("Please enter e-mail address.")
    # st.write("Email address is", email)

    if resume_file:
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)

        # Save uploaded resume
        with open(os.path.join(temp_dir, resume_file.name), "wb") as f:
            f.write(resume_file.getbuffer())

        st.success(f"File '{resume_file.name}' saved successfully.")
        resume_file_path = os.path.join("temp", resume_file.name)
        resume_docs, resume_chunks = load_split_pdf(resume_file_path)
        shutil.rmtree(temp_dir)

        if st.button("Analyze Resume", help="Click to analyze the resume"):
            backendcalculations(resume_docs, resume_chunks, location, query, st)
    else:
        st.warning("Please upload a resume.")

if __name__ == "__main__":
    main()

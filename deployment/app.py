import streamlit as st
from backend import backendcalculations


def main():
    st.markdown(
        """
        <h1 style="text-align: center;">Job - Resume Match</h1>
        """,
        unsafe_allow_html=True
    )

    email = st.text_input("Enter your email address:")

    st.header("Upload Resume")  # Header for the upload section

    resume_file = st.file_uploader("Upload Resume (PDF)", type="pdf")

    location = st.selectbox(
        "Please select city.",
        ("Toronto", "Vancouver"),
    )

    query = st.selectbox(
        "Please select job position.",
        ("Data Analyst", "Data Scientist", "Product Manager", "Digital Marketer", "Machine Learning Engineer",
         "Software Developer"),
        )

    if resume_file:
        if st.button("Analyze Resume", help="Click to analyze the resume"):
            backendcalculations(resume_file, location, query, st, email)
    else:
        st.warning("Please upload a resume.")

if __name__ == "__main__":
    main()

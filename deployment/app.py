import streamlit as st
from backend import backendcalculations


def main():
    st.markdown(
        """
        <h1 style="text-align: center;">Job - Resume Match</h1>
        """,
        unsafe_allow_html=True
    )

    st.header("Upload Resume")  # Header for the upload section

    resume_file = st.file_uploader("Upload Resume (PDF)", type="pdf")
    # if resume_file is not None:
    #     resume_file = resume_file.read()
    #     print(resume_file)

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

    if resume_file:
        if st.button("Analyze Resume", help="Click to analyze the resume"):
            backendcalculations(resume_file, location, query, st)
    else:
        st.warning("Please upload a resume.")

if __name__ == "__main__":
    main()

from pathlib import Path
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Define the root directory
ROOT_DIR = str(Path(__file__).parent.resolve()) + "/"

# Set page configuration
st.set_page_config(page_title="STACK Analytics Hub", page_icon="📊")

# Main title and header
st.title("Moodle STACK Analytics Hub")
st.header("Streamlining Data Analysis for Moodle STACK")

# About section
st.markdown(
    """
    ## Introduction
    Welcome to the Moodle STACK Analytics Hub! This web-based tool is designed to simplify the analysis of STACK data from Moodle LMS courses. Whether you're a lecturer or an administrator, this tool will help you gain valuable insights into student performance and quiz effectiveness.
    
    ## Features
    - **Quiz Analysis:** Explore metrics such as grade distribution, engagement patterns, and correlations.
    - **Student Performance Overview:** Get an overview of student performance across quizzes to identify trends and areas for improvement.
    - **Question Statistics:** Assess the quality and difficulty of individual questions and identify common misconceptions.
    - **Quiz Statistics:** Comprehensive statistics on student engagement, average grades, variance, and more.
    - **Additional Analyses:** Future updates will include more advanced features and customizable reports, i.e Response Analysis from students to pinpoint common misconceptions and visualize response patterns.

    ## How It Works
    1. **Upload Data:** Load your STACK data files through the sidebar.
    2. **Select Options:** Choose from various analysis options to view relevant insights.
    3. **Visualize Results:** Explore interactive charts, tables, and visualizations to understand the data better.

    ## User Guide
    [Click here](#) User guide coming up#.

    ## Upcoming Features
    Stay tuned for upcoming features such as in-depth engagement analysis, predictive analytics, clustering algorithms, and custom reporting.

    ## Feedback and Support
    We welcome your feedback and suggestions! If you have any questions or need support, please reach out to us at [support@example.com](mailto:support@example.com).
    """
)

# Add vertical space
st.markdown("<br>", unsafe_allow_html=True)

# Built with section
st.markdown(
    """
    Built with:
    - [Streamlit](https://streamlit.io/)
    - [Pandas](https://pandas.pydata.org/)
    - [Seaborn](https://seaborn.pydata.org/)
    - [Matplotlib](https://matplotlib.org/)
    """
)

# Footer
st.markdown(
    """
    ---
    **Contact Us:** For any queries or support, please reach out to our team at [support@example.com](mailto:support@example.com).
    """
)

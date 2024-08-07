# Sage Foundation: tech for good education team 1 

## Tailored Quiz Recommendation System

## Overview

This project develops a tailored quiz recommendation system aimed at improving student learning outcomes by recommending quizzes based on their past performance. The system uses machine learning models to predict the quizzes that would be most beneficial for each student, thus personalizing the learning experience.

## Features

- **Upload Student Data**: Professors can upload CSV files containing students' quiz scores.
- **Email-based Recommendations**: After specifying a student's email, the system provides tailored quiz recommendations for that student.
- **Recommendation Types**: Users can choose between two types of recommendations:
  - `top_n`: Recommends the quizzes with the highest predicted quiz scores (easiest questions).
  - `diverse`: Recommends a diverse set of quizzes to cover a range of potential interests and learning needs.
- **Interactive Web Interface**: Built using Streamlit, the interface is user-friendly and allows for easy interaction without the need for additional software.
- **Downloadable Recommendations**: Professors can download the recommended quizzes as a CSV file, making it easy to keep track of suggestions.

## Technical Description

### Tools and Libraries Used

- **Streamlit**: Web framework used for creating the interactive web interface.
- **Surprise**: A Python scikit for building and analyzing recommender systems that deal with explicit rating data.
- **Pickle**: For saving and loading machine learning models and data sets, ensuring that state is preserved between sessions.
- **Pandas**: Data manipulation and analysis.
- **Scikit-Learn**: Stats library for building clusetring algorithms

## Usage

#### 1. Conda env setup and packages install
```
conda create -n hackathon-education python=3.10.8
conda activate hackathon-education
pip install poetry
cd tech-for-good-education-team
poetry install
```

#### 2. Change Jupyter Notebook kernel
Change Jupyter Notebook kernel to `hackathon-education` conda env.
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import re
import altair as alt

# Initialize an empty dataframe
df_filtered = pd.DataFrame()

# Function to parse time duration from string format
def parse_time(time_str):
    regex = re.compile(r'((?P<weeks>\d+?) ?weeks?)? ?((?P<days>\d+?) ?days?)? ?((?P<hours>\d+?) ?hours?)? ?((?P<minutes>\d+?) ?mins?)? ?((?P<seconds>\d+?) ?secs?)?')
    parts = regex.match(time_str)
    if parts is None:
        return None
    parts = parts.groupdict()
    time_params = {name: int(param) for name, param in parts.items() if param}
    return timedelta(**time_params).total_seconds()

# Function to parse datetime from string format
def parse_datetime(str):
    try:
        return datetime.strptime(str, "%d %B %Y %I:%M %p")
    except ValueError:
        return None

# Function to normalize grades
def normalize_grades(df):
    # Define a pattern to find the grade column
    grade_column_pattern = re.compile(r'Grade/\d+(?:\.\d+)?')
    
    # Identify the grade column
    grade_column = None
    for column in df.columns:
        if grade_column_pattern.match(column):
            grade_column = column
            break
    
    if grade_column is None:
        raise KeyError("Grade column not found in the dataset")
    
    # Extract the maximum grade value from the column name
    max_grade_value = float(re.search(r'\d+(?:\.\d+)?', grade_column).group())
    
    # Convert grades to numeric values
    df[grade_column] = pd.to_numeric(df[grade_column], errors='coerce')
    
    # Normalize the grades to a scale of 10
    if max_grade_value != 10:
        df['Normalized_Grade'] = (df[grade_column] / max_grade_value) * 10
    else:
        df['Normalized_Grade'] = df[grade_column]
    
    return df

# Function to load and process uploaded data files
def load_data(uploaded_files):
    dfs = []
    for quizID, file in enumerate(uploaded_files):
        if file.name.endswith('.xls'):
            df = pd.read_excel(file, engine='xlrd')
        elif file.name.endswith('.xlsx'):
            df = pd.read_excel(file, engine='openpyxl')
        elif file.name.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            raise ValueError("Unsupported file format: " + file.name)
        
        df.insert(3, "quizID", quizID + 1, True)
        if "Last name" in df.columns:
            df = df.rename(columns={'Last name': 'Surname'})
        
        df = normalize_grades(df)
        
        columns = ['Surname', 'First name', 'Email address', 'quizID', 'State', 'Started on', 'Completed', 'Time taken', 'Normalized_Grade']
        missing_columns = [col for col in columns if col not in df.columns]
        if missing_columns:
            raise KeyError(f"Missing columns: {', '.join(missing_columns)}")
        
        dfs.append(df[columns])
    
    df = pd.concat(dfs)
    df = df[df.State == "Finished"]
    df["Started on"] = df["Started on"].apply(parse_datetime)
    df["Completed"] = df["Completed"].apply(parse_datetime)
    df["Time taken"] = df["Time taken"].apply(parse_time)
    df = df.rename(columns={'Surname': 'surname', 'First name': 'firstname', 'Email address': 'email',
                            'State': 'state', 'Started on': 'start_date', 'Completed': 'end_date',
                            'Time taken': 'time_taken', 'Normalized_Grade': 'grade'})
    df['grade'] = pd.to_numeric(df['grade'], errors='coerce')
    return df


# Streamlit app setup
st.title("Moodle STACK Analytics")

st.markdown("""
This was a Hackathon project — thanks to Sage for helping with the technical setup.  
It's still a work in progress. If you notice any bugs, please create a PR on GitHub.  

?? Below is an intro video explaining the project idea:
""")

st.video("https://youtu.be/Ww_FrryExYc?si=x-yeDCqGgUhDjFMb")

st.markdown("""
---

### ?? About this Tool

This dashboard allows you to upload and analyze **Moodle STACK quiz data**:

- ?? Supports `.csv`, `.xls`, `.xlsx` file formats  
- ?? Automatically detects and normalizes grades  
- ?? Parses quiz start/end times and time taken  
- ?? Visualizes quiz performance and trends  

You need to upload your quiz files using the **file uploader on the left side**.

---

### ?? How to Export Quiz Attempt Data from Moodle

To use this dashboard, first **download your Moodle quiz attempt data on your computer**:
Here are the steps to follow if you forgot. 
1. Log in to your Moodle course as a teacher or admin.  
2. Navigate to the **specific quiz** you want to analyze.  
3. Click on **"Results"** in the quiz menu.  
4. Select **"Grades"** to see a list of attempts for all students. 
5. Scroll to the bottom of the table and look for **"Download table data as"**.  
6. Choose `.CSV`, `.XLS`, or `.XLSX` and download the file.  
7. Upload the file to this dashboard.

#### ? Required Columns Format

Make sure your file includes the following columns **exactly** as named:

**Surname, First name, Email address, State, Started on, Completed, Time taken, Grade/10.00**

Moodle setups can vary — if your columns are named differently, you may need to manually rename them before uploading.

Example files to test or adapt are available here:  
?? [Sample Moodle Quiz Files](https://drive.google.com/drive/folders/1r7c1asoMFwaLORaQVKisJk7xpWazzC5I?usp=sharing)

---
""")







st.sidebar.title("Options")

# File upload section
uploaded_files = st.sidebar.file_uploader("Upload Excel files", accept_multiple_files=True)

if uploaded_files:
    # Load and process data
    df = load_data(uploaded_files)
    
    if not df.empty:
        # User control section
        selected_quizzes = st.sidebar.multiselect("Select Quiz IDs", options=df['quizID'].unique(), default=df['quizID'].unique())
        df_filtered = df[df['quizID'].isin(selected_quizzes)]

# Display merged data with explanation
if st.sidebar.checkbox("Merged List of Users and Files"):
    st.write("### Merged List of Users and Files")

    st.write(
    "This table combines all uploaded quiz files into one view. Each row is one attempt, with the student, quiz, date, time taken, and normalized grade."
    )
    st.dataframe(df_filtered)
    
# Function to generate Quiz stats based on selected options
def generate_quiz_stats(selected_stats):
    if df_filtered.empty:
        return pd.DataFrame()  # Return an empty dataframe if no data
    
    dfs = []
    
    if "student_count" in selected_stats:
        students_per_quiz = df_filtered.groupby('quizID')['email'].nunique().reset_index()
        students_per_quiz.columns = ['quizID', 'student_count']
        dfs.append(students_per_quiz)
    
    if "mean_grade" in selected_stats or "grade_variance" in selected_stats:
        grade_statistics = df_filtered.groupby('quizID')['grade'].agg(['mean', 'var']).reset_index()
        grade_statistics.columns = ['quizID', 'mean_grade', 'grade_variance']
        if "mean_grade" not in selected_stats:
            grade_statistics = grade_statistics.drop(columns=['mean_grade'])
        if "grade_variance" not in selected_stats:
            grade_statistics = grade_statistics.drop(columns=['grade_variance'])
        dfs.append(grade_statistics)
    
    if "mean_highest_grade" in selected_stats:
        highest_grades = df_filtered.groupby(['quizID', 'email'])['grade'].max().reset_index()
        average_highest_grades = highest_grades.groupby('quizID')['grade'].mean().reset_index()
        average_highest_grades.columns = ['quizID', 'mean_highest_grade']
        dfs.append(average_highest_grades)
    
    if "attempt_count" in selected_stats:
        total_attempts_per_quiz = df_filtered.groupby('quizID').size().reset_index(name='attempt_count')
        dfs.append(total_attempts_per_quiz)
    
    if "attempt_rate" in selected_stats:
        attempts_per_student = df_filtered.groupby(['quizID', 'email']).size().reset_index(name='attempt_count')
        average_attempts_per_student = attempts_per_student.groupby('quizID')['attempt_count'].mean().reset_index()
        average_attempts_per_student.columns = ['quizID', 'attempt_rate']
        dfs.append(average_attempts_per_student)
    
    if dfs:
        quiz_stats = dfs[0]
        for df in dfs[1:]:
            quiz_stats = pd.merge(quiz_stats, df, on='quizID')
        return quiz_stats.round(2)
    else:
        return pd.DataFrame()  # Return an empty dataframe if no stats are selected

# Sidebar options
show_summary = st.sidebar.checkbox("Summary of Quiz Stats")

if show_summary:
    st.write("### Summary of Quiz Statistics")
    if not df_filtered.empty:
        # Display checkboxes for selecting statistics
        selected_stats = st.sidebar.multiselect(
            "Select Statistics to Display",
            ["student_count", "attempt_rate", "mean_grade", "grade_variance", "mean_highest_grade", "attempt_count"],
            default=["student_count", "attempt_rate", "mean_grade", "grade_variance", "mean_highest_grade", "attempt_count"]
        )

    # Generate and display quiz statistics based on selected options
        # Add summary notes based on selected statistics
        summary_notes = {
            "student_count": "- **student_count:** how many different students attempted each quiz.\n",
            "attempt_rate": "- **attempt_rate:** the average number of attempts per student for each quiz.\n",
            "mean_grade": "- **mean_grade:** the average score for each quiz.\n",
            "grade_variance": "- **grade_variance:** how spread out the grades are for each quiz.\n",
            "mean_highest_grade": "- **mean_highest_grade:** the average best score each student reached in a quiz.\n",
            "attempt_count": "- **attempt_count:** the total number of attempts made on each quiz.\n",
        }

        summary_text = " ".join([summary_notes[stat] for stat in selected_stats])
        st.write(summary_text)
        
        quiz_stats = generate_quiz_stats(selected_stats)
        st.dataframe(quiz_stats)
        
    else:
       st.write("You need to upload a file(s) to initiate the analysis.")

# Performance Distribution on Quizzes section
if st.sidebar.checkbox("Quiz Grade Distribution (Box plot)", False):
    if not df_filtered.empty:
        st.write("### Quiz Grade Distribution (Box plot)")
        sns.set(style="whitegrid")
        plt.figure(figsize=(10, 6))
        
        st.write("""
        - **Box Plot:** Shows how grades are spread out for each quiz, showcasing where most grades fall.
        - **Median:** The green line marks the middle value of grades for each quiz.
        - **mean_grade line:** The red line shows the average grade for each quiz. It helps you see if the average grade is going up or down across quizzes.
        """)
        # Create box plot with custom median line color
        box_plot = sns.boxplot(x='quizID', y='grade', data=df_filtered, palette='Set3',
                               medianprops=dict(color="#00FF00"))
        
        # Overlay strip plot for individual data points
        sns.stripplot(x='quizID', y='grade', data=df_filtered, color='black', jitter=0.1, size=1.8)
        
        # Calculate means for each quizID
        means = df_filtered.groupby('quizID')['grade'].mean().reset_index()
        
        # Plot the means as a line
        plt.plot(means['quizID'].astype(int) - 1, means['grade'], marker='o', color='#FF474C', linestyle='-', linewidth=2, label='mean_grade')
        
        plt.title('Grade Distribution')
        plt.xlabel('Quiz ID')
        plt.ylabel('Grade')
        
        # Add legend to distinguish mean line
        plt.legend()

        st.pyplot(plt)

    else:
        st.write("You need to upload a file(s) to initiate the analysis.")



# Frequency Density showing quiz engagement over time
if st.sidebar.checkbox("Frquency Density (Engagement) "):
    st.write("### Engagement Over Time")
    if 'start_date' in list(df_filtered.columns):
        df_filtered['start_date'] = pd.to_datetime(df_filtered['start_date'])

        plt.figure(figsize=(10, 8))

        # Check if start_date has data
        if not df_filtered['start_date'].isnull().any():
            # Quiz engagement over time
            for quiz_id in df_filtered['quizID'].unique():
                quiz_data = df_filtered[df_filtered['quizID'] == quiz_id]
                if not quiz_data.empty:
                    sns.kdeplot(quiz_data['start_date'], label=f'Quiz {quiz_id}')
                else:
                    st.write(f"No data for Quiz ID: {quiz_id}")

            plt.title('Engagement Over Time')
            plt.xlabel('Date')
            plt.ylabel('Frequency Density')
            plt.xticks(rotation=45)
            plt.legend()
            plt.grid(True)
            st.pyplot(plt)

            st.write(
                """
                This graph shows how students' participation in quizzes changes over time: The line graph displays the frequency of quiz attempts on different dates. Each line represents a different quiz, showing when students started taking it. 

                **What You Can Learn:**
                - **Trends:** See if there are certain times when more students are attempting quizzes.
                - **Peak Times:** Identify dates when engagement is high or low for each quiz.
                - **Comparisons:** Compare the participation trends across different quizzes.

                This helps to understand when students are most active and whether certain quizzes are more popular at specific times.
                """
            )
        else:
            st.write("No valid dates available for plotting.")



# Attempts vs Grade Analysis
if st.sidebar.checkbox("Scatter plot: Attempts vs Grades"):
    if 'quizID' in list(df_filtered.columns):
        st.write("### Scatter plot: Attempts vs Grades")

        # Add a radio button to select the type of grade to display
        grade_type = st.sidebar.radio("Select Grade Type", ("Highest Grade", "Average Grade", "Minimum Grade"))

        # Create a figure for the plot
        plt.figure(figsize=(25, 15))  # Set the figure size

        # Compute the number of attempts in each quiz
        quiz_attempt_count = df_filtered.groupby(['quizID', 'email']).size().reset_index(name='attempt_count')

        # Compute the highest, average, or minimum grades
        if grade_type == "Highest Grade":
            # Find the highest grade attained by each student in each quiz
            grade_data = df_filtered.groupby(['quizID', 'email'])['grade'].max().reset_index()
            y_label = 'Highest Grade'
            title = 'Attempts vs Highest Grade'
        elif grade_type == "Minimum Grade":
            # Find the minimum grade attained by each student in each quiz
            grade_data = df_filtered.groupby(['quizID', 'email'])['grade'].min().reset_index()
            y_label = 'Minimum Grade'
            title = 'Attempts vs Minimum Grade'
        else:
            # Find the average grade attained by each student in each quiz
            grade_data = df_filtered.groupby(['quizID', 'email'])['grade'].mean().reset_index()
            y_label = 'Average Grade'
            title = 'Attempts vs Average Grade'

        # Merge the attempt count and grade data
        merged_data = pd.merge(quiz_attempt_count, grade_data, on=['quizID', 'email'])

        # Compute the correlation between attempt count and selected grade type
        correlation = merged_data['attempt_count'].corr(merged_data['grade'])

        # Display the correlation coefficient
        st.write(f"Correlation between Attempts and Quiz {y_label}: r = {correlation:.2f}")

        # Plot No. of Attempts vs selected grade type
        sns.scatterplot(data=merged_data, x='attempt_count', y='grade', hue='quizID', palette='husl', marker='o', s=200)

        # Set plot title and labels
        plt.title(title, fontsize=20)
        plt.xlabel('No. of Attempts', fontsize=20)
        plt.ylabel(y_label, fontsize=20)
        plt.legend(title='Quiz ID', fontsize=20)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        st.pyplot(plt)

        # Add summary notes
        st.write(
            f"""
    **Attempts vs {y_label}:**
    
    This scatter plot shows how the number of attempts relates to the {y_label} for each quiz:
    
    - **Plot:** Each point represents a student's attempts and their {y_label}. The x-axis is the number of attempts, and the y-axis is the {y_label}.
    
    **What You Can Learn:**
    - **Trends:** See if more attempts lead to better or worse grades.
    - **Patterns:** Identify if attempts impact grades positively or negatively.
    - **Correlation:** Shows how strongly attempts are related to the selected grade type. A high value means more attempts are linked to better grades.

    - **Highest Grade:** The best score a student achieved. 
    - **Minimum Grade:** The lowest score a student achieved. 
    - **Average Grade:** The overall performance across attempts. 

    These statistics help understand if more attempts help students improve or if they face consistent issues.
             """
        )

    else:
        st.write("You need to upload a file(s) to initiate the analysis.")
# Line Graph of Various Quiz Metrics
if st.sidebar.checkbox("Line Graph of Various Metrics"):
    st.write("### Line Graph of Various Metrics")

    # Display checkboxes for selecting metrics
    selected_metrics = st.sidebar.multiselect(
        "Select Metrics to Display",
        ["student_count", "attempt_rate", "mean_grade", "grade_variance"],
        default=["student_count", "attempt_rate", "mean_grade", "grade_variance"]
    )

    if selected_metrics:
        # Explanation of the selected metrics

        if "student_count" in selected_metrics:
            st.write("student_count: number of unique students who attempted each quiz.")

        if "attempt_rate" in selected_metrics:
            st.write("attempt_rate: average number of attempts per student for each quiz.")

        if "mean_grade" in selected_metrics:
            st.write("mean_grade: average score for each quiz.")

        if "grade_variance" in selected_metrics:
            st.write("grade_variance: how spread out the grades are for each quiz.")

        # Function to generate data for the line graph
        def generate_line_graph_data(df, selected_metrics):
            data = {}

            if "student_count" in selected_metrics:
                data["student_count"] = df.groupby('quizID')['email'].nunique()

            if "attempt_rate" in selected_metrics:
                attempts_per_student = df.groupby(['quizID', 'email']).size().reset_index(name='attempt_count')
                data["attempt_rate"] = attempts_per_student.groupby('quizID')['attempt_count'].mean()

            if "mean_grade" in selected_metrics:
                data["mean_grade"] = df.groupby('quizID')['grade'].mean()

            if "grade_variance" in selected_metrics:
                data["grade_variance"] = df.groupby('quizID')['grade'].var()

            return pd.DataFrame(data).reset_index()
        if not df_filtered.empty:
            line_graph_data = generate_line_graph_data(df_filtered, selected_metrics)
            line_graph_data = line_graph_data.melt('quizID', var_name='Metric', value_name='Value')

            line_chart = alt.Chart(line_graph_data).mark_line().encode(
                x=alt.X('quizID:O', title='Quiz ID'),
                y=alt.Y('Value:Q'),
                color='Metric:N',
                tooltip=['quizID', 'Metric', 'Value']
            ).properties(width=600, height=400)

            points = line_chart.mark_point().encode(
                x=alt.X('quizID:O', title='Quiz ID'),
                y=alt.Y('Value:Q'),
                color='Metric:N',
                tooltip=['quizID', 'Metric', 'Value']
            )

            st.altair_chart(line_chart + points)
        
    else:
        st.write("You need to upload a file(s) to initiate the analysis.")


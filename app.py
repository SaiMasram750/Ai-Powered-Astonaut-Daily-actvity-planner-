# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib  # or pickle
from datetime import time, datetime, timedelta

# ----------------------------
# 1. PAGE SETUP
# ----------------------------
st.set_page_config(page_title="Astronaut Activity Planner", page_icon="🚀", layout="centered")
st.title("🚀 Astronaut Daily Activity Planner")
st.markdown("Input your tasks below and I'll generate your optimal schedule for the day!")

# ----------------------------
# 2. LOAD MODEL & PREPROCESSORS (WITH CACHING)
# ----------------------------
# This function runs only once thanks to caching, making the app fast.
@st.cache_resource
def load_artifacts():
    """
    Load the model and any necessary pre-processing artifacts.
    Replace the filenames with yours.
    """
    model = joblib.load('model.pkl')
    # If you used a text vectorizer (e.g., for the task name), load it here.
    # vectorizer = joblib.load('vectorizer.pkl')
    # If you have a label encoder for the time slot, load it here.
    # label_encoder = joblib.load('label_encoder.pkl')
    return model #, vectorizer, label_encoder

# Load the artifacts
model = load_artifacts()
# If you loaded a vectorizer and encoder, uncomment the lines below:
# vectorizer, label_encoder = load_artifacts()

# ----------------------------
# 3. USER INPUT SECTION
# ----------------------------
st.header("📝 Input Your Tasks")

# Initialize a list to store all tasks
if 'tasks' not in st.session_state:
    st.session_state.tasks = []

# Form for adding a single task
with st.form("task_form"):
    col1, col2 = st.columns(2)
    with col1:
        task_name = st.text_input("Task Name*", placeholder="e.g., Analyze soil samples")
        task_category = st.selectbox("Category*", ["Work", "Exercise", "Research", "Maintenance", "Leisure", "Meal", "Communication"])
    with col2:
        estimated_duration = st.slider("Duration (min)*", 15, 240, 30, step=15)
        priority = st.select_slider("Priority*", options=["Low", "Medium", "High"])
        energy_required = st.select_slider("Energy Required*", options=["Low", "Medium", "High"])

    # Every form must have a submit button.
    submitted = st.form_submit_button("Add Task")
    if submitted:
        if task_name:  # Simple validation
            new_task = {
                "Name": task_name,
                "Category": task_category,
                "Duration (min)": estimated_duration,
                "Priority": priority,
                "Energy": energy_required
            }
            st.session_state.tasks.append(new_task)
            st.success(f"Task '{task_name}' added!")
        else:
            st.warning("Please give the task a name.")

# Display and manage the list of added tasks
if st.session_state.tasks:
    st.subheader("Your Task List")
    tasks_df = pd.DataFrame(st.session_state.tasks)
    st.dataframe(tasks_df, use_container_width=True)

    if st.button("Clear All Tasks", type="secondary"):
        st.session_state.tasks = []
        st.rerun()

# ----------------------------
# 4. PREDICTION & SCHEDULE BUILDING
# ----------------------------
if st.session_state.tasks and st.button("🧠 Generate Smart Schedule", type="primary"):

    st.header("📅 Your Optimal Schedule")
    st.info("Building your schedule based on predictive analytics...")

    # Convert our list of tasks into a DataFrame the model can predict on
    input_df = pd.DataFrame(st.session_state.tasks)

    # !!! CRITICAL: FEATURE ENGINEERING !!!
    # Your model was trained on specific features. You MUST create the EXACT
    # same features from the user input.
    # This example uses simple one-hot encoding for categories. You MUST use
    # the SAME pre-processing as your training pipeline.

    # Example: Create dummy variables for categorical features.
    # This is a simplistic approach. You should use the same transformer you used in training.
    model_input = input_df.copy()
    # For demonstration. In reality, use your saved preprocessor (like a ColumnTransformer).
    model_input = pd.get_dummies(model_input, columns=['Category', 'Priority', 'Energy'])

    # Ensure the model_input has all the columns your model expects, in the right order.
    # You might need to add missing columns with zeros and reorder them.
    # This is often the trickiest part. Refer to your model's training code.

    # Make predictions for the optimal time slot for each task
    try:
        predictions = model.predict(model_input)
        # If your model outputs numbers (0,1,2,3), map them to time slots.
        time_slot_map = {0: "🌅 Morning", 1: "🌞 Afternoon", 2: "🌇 Evening", 3: "🌙 Late Night"}
        input_df['Predicted Time Slot'] = [time_slot_map.get(p, "N/A") for p in predictions]

        # ----------------------------
        # 5. BUILD THE SCHEDULE
        # ----------------------------
        # Group tasks by their predicted time slot
        schedule = {}
        for slot in time_slot_map.values():
            schedule[slot] = input_df[input_df['Predicted Time Slot'] == slot].to_dict('records')

        # Define a time range for each slot (for display purposes)
        time_ranges = {
            "🌅 Morning": "06:00 - 12:00",
            "🌞 Afternoon": "12:00 - 17:00",
            "🌇 Evening": "17:00 - 22:00",
            "🌙 Late Night": "22:00 - 06:00"
        }

        # Display the schedule in a nice way
        for time_slot, tasks_in_slot in schedule.items():
            if tasks_in_slot:  # Only show slots that have tasks
                with st.expander(f"{time_slot} ({time_ranges[time_slot]})", expanded=True):
                    for task in tasks_in_slot:
                        st.markdown(
                            f"""
                            **{task['Name']}**  
                            *⏱️ {task['Duration (min)']} min | 🏷️ {task['Category']} | 🚨 {task['Priority']} Priority*
                            """
                        )
        # Show the raw predictions for debugging (you can remove this later)
        st.subheader("Debug: Prediction Details")
        st.write(input_df[['Name', 'Predicted Time Slot']])

    except Exception as e:
        st.error(f"Something went wrong during prediction. This is often due to feature mismatches.")
        st.exception(e) # This will print the error for debugging.

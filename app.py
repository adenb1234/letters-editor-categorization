import streamlit as st
import pandas as pd
import openai
import os

# Set OpenAI API key
openai.api_key = st.secrets["openai_api_key"]

def generate_content(gpt_assistant_prompt: str, gpt_user_prompt: str) -> dict:
    messages = [
        {"role": "system", "content": gpt_assistant_prompt},
        {"role": "user", "content": gpt_user_prompt}
    ]
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        temperature=0.2,
        max_tokens=256,
        frequency_penalty=0.0
    )
    response_text = response.choices[0]['message']['content']
    tokens_used = response['usage']['total_tokens']

    return {"response": response_text, "tokens_used": tokens_used}

def classify_response(response, categories):
    gpt_assistant_prompt = "You are an expert in political analysis and categorization."
    gpt_user_prompt = f"""
Classify the following letter to the editor response into ONE of the following categories:
{categories}
If it doesn't fit into any of these categories, classify it as 'Other'.

Only provide ONE category as the main classification. This should be the most prioritized issue by the letter, which will likely be the one the letter addresses in more depth or, if equal in depth, the first issue named. Under no circumstances should multiple categories be listed for the main classification.

Response: {response}
"""

    result = generate_content(gpt_assistant_prompt, gpt_user_prompt)
    response_text = result['response'].strip()

    main_label = "Main Classification:"

    try:
        main_category = response_text.split(main_label)[1].strip()
    except IndexError:
        st.error(f"Unexpected response format: {response_text}")
        return "Error"

    return main_category

st.title("Letters to the Editor Categorization App")

# Step 1: File Upload
uploaded_file = st.file_uploader("Upload a CSV file with letters to the editor", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Data:", df)

    # Step 2: Category Input
    st.write("Specify the categories for classification.")
    categories = st.text_area("Enter categories (one per line)", value="")

    if st.button("Classify Letters"):
        categories_list = categories.split('\n')
        categories_list = [cat.strip() for cat in categories_list if cat.strip()]
        categories_string = '\n'.join(categories_list)

        # Initialize new column
        if 'Main Category' not in df.columns:
            df['Main Category'] = None

        # Apply the function to each answer and update the new column
        progress_bar = st.progress(0)
        for i, row in df.iterrows():
            if pd.isna(row['Main Category']):
                main_category = classify_response(row['answer'], categories_string)
                df.at[i, 'Main Category'] = main_category
            progress_bar.progress((i + 1) / len(df))

        # Display classified data
        st.write("Classified Data:", df)

        # Step 3: Download the processed CSV
        st.download_button(
            label="Download Classified Data",
            data=df.to_csv(index=False),
            file_name='classified_responses.csv',
            mime='text/csv'
        )

        # Summary Statistics
        st.write("Summary Statistics")
        category_counts = df['Main Category'].value_counts()
        st.bar_chart(category_counts)
      

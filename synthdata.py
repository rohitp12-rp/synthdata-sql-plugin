import os
import re
import configparser
import streamlit as st
import pyodbc
import pandas as pd
import pymongo

from mong_connection import MongoConnection
from sdv.tabular import CTGAN
from sdv.constraints import Unique

import google.generativeai as genai


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "<google-cloud-service-key>"
google_api_key = "<google-api-key>"

def read_config(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)
    return config

def establish_connection(config):
    driver = st.selectbox(
        'Driver',
        ('mysql', 'postgresql', 'redshift', 'oracle'),
        index=0,
        placeholder="Select Your desired Database",
    )
    server = st.text_input("Server")
    database = st.text_input("Database")
    uid = st.text_input("User ID")
    pwd = st.text_input("Password", type="password")

    connection_string = (
        f"DRIVER={config[driver]['DRIVER']};"
        f"SERVER={server};"
        f"DATABASE={database};"
        f"UID={uid};"
        f"PWD={pwd};"
        f"charset={config[driver]['charset']};"
    )
    return connection_string

def query_database(cursor, table_name):
    cursor.execute(f"SELECT * FROM {table_name}")
    fields = [desc[0] for desc in cursor.description]
    rows = cursor.fetchall()
    return pd.DataFrame.from_records(rows, columns=fields)

def generate_synthetic_data(df, primary_key, anonymize_fields, num_rows=1000):
    unique_ids = [Unique(column_names=primary_key)]
    model = CTGAN(primary_key=primary_key, anonymize_fields=anonymize_fields, constraints=unique_ids)
    model.fit(df)
    synth_data = model.sample(num_rows)
    synth_data[primary_key[0]] += 1
    return synth_data

def get_primary_keys(cursor, table_name):
    cursor.execute(f"SHOW KEYS FROM {table_name} WHERE Key_name = 'PRIMARY'")
    primary_key_info = cursor.fetchall()
    primary_keys = [key_info[4] for key_info in primary_key_info]
    return primary_keys

def get_anonymize_fields(config, df,cursor,table_name):
    anonymize_config = config['anonymize']
    anonymize_fields = {}

    # Get column info
    columns_query = f"SHOW COLUMNS FROM {table_name}"
    cursor.execute(columns_query)
    column_info = cursor.fetchall()

    for column in df.columns:
        if column in anonymize_config:
            anonymize_fields[column] = config['anonymize'][column.lower()]
        elif any(col[0] == column and re.search('varchar*' ,col[1].lower()) and 'uni' in col[3].lower() for col in column_info):
            anonymize_fields[column] = 'text'
    return anonymize_fields

def insert_synthetic_data(cursor, table_name, synthetic_data):
    insert_query = f"INSERT INTO {table_name} ({', '.join(synthetic_data.columns)}) VALUES ({', '.join(['?'] * len(synthetic_data.columns))})"
    for index, row in synthetic_data.iterrows():
        cursor.execute(insert_query, tuple(row))

def confine_to_max_length(cursor,table_name,synthetic_data):
    columns_query = f"SHOW COLUMNS FROM {table_name}"
    cursor.execute(columns_query)
    column_info = cursor.fetchall()
    max_lengths = {column[0]: int(column[1].split('(')[1].split(')')[0]) if 'varchar' in column[1].lower() else None for column in column_info}
    # Check and truncate column lengths in the synthetic data
    for column in synthetic_data.columns:
        if column in max_lengths and max_lengths[column] is not None:
            synthetic_data[column] = synthetic_data[column].apply(lambda x: x[:max_lengths[column]] if (isinstance(x, str) and x is not None) else x)
    return synthetic_data

def create_new_table(table_name, cursor,conn):
    # Define the new table name
    new_table_name = f'{table_name}_test'
    columns_query = f"SHOW COLUMNS FROM {table_name}"
    cursor.execute(columns_query)
    column_info = cursor.fetchall()
    column_definitions = [f"{column[0]} {column[1]}" for column in column_info]
    create_table_query = f"CREATE TABLE {new_table_name} ({', '.join(column_definitions)})"
    cursor.execute(create_table_query)
    conn.commit()
    return new_table_name

def main():
    st.title("Synthetic Data Generation Chatbot")

    config_file = 'config.ini'
    config = read_config(config_file)

    user_input = st.sidebar.radio("Select an option", ["Generate Synthetic Data", "Generate SQL Query"])

    if user_input == "Generate Synthetic Data":
        st.subheader("Generate Synthetic Data")
        type_of_sql = st.selectbox(
            "Type of database",
            ("SQL","NoSQL")
        )
        if type_of_sql == "NoSQL":
            uri = st.text_input("MongoDB URI")
            datbase_name = st.text_input("Database Name")
            collection_name = st.text_input("Collection Name")
            if st.button("Generate Synthetic Data"):
                client = pymongo.MongoClient(uri)
                connection = MongoConnection(client, datbase_name, collection_name)
                db, df = connection.establish_mongo_connection()
                cleaned_df, mapping, list_mapping = connection.clean_df(df)
                synthetic_data = connection.generate_synthetic_data(cleaned_df)
                cleaned_synthetic_data = connection.clean_synthetic_data(synthetic_data, mapping, list_mapping)
                st.write(cleaned_synthetic_data)
                connection.push_data_to_new_table(cleaned_synthetic_data,db)
                st.success("Synthetic data has been generated and inserted into the database.")

        else: 
            connection_string = establish_connection(config)
            table_name = st.text_input("Enter table name:")
            if st.button("Generate Synthetic Data"):
                conn = pyodbc.connect(connection_string)
                cursor = conn.cursor()
                
                original_data = query_database(cursor, table_name)
                primary_keys = get_primary_keys(cursor, table_name)
                anonymize_fields = get_anonymize_fields(config, original_data,cursor,table_name)

                raw_synthetic_data = generate_synthetic_data(original_data, primary_key=primary_keys, anonymize_fields=anonymize_fields)

                #Confining data to maximum length
                synthetic_data = confine_to_max_length(cursor,table_name,raw_synthetic_data)

                # Handle NULL values
                columns_query = f"SHOW COLUMNS FROM {table_name}"
                cursor.execute(columns_query)
                column_info = cursor.fetchall()
                nullable_columns = [column[0] for column in column_info if "NOT NULL" not in column[1]]
                synthetic_data[nullable_columns] = synthetic_data[nullable_columns].applymap(lambda x: None if pd.isna(x) else x)

                st.write("Generated Synthetic Data:")
                st.write(synthetic_data)

                # Creating a new table to push synthetic data into
                new_table_name = create_new_table(table_name,cursor,conn)
                # Insert synthetic data into the database
                insert_synthetic_data(cursor, new_table_name, synthetic_data)

                # Commit the changes and close the connection
                conn.commit()
                cursor.close()

                st.success("Synthetic data has been generated and inserted into the database.")
    elif user_input == "Generate SQL Query":
        st.subheader("Generate SQL Query")

        genai.configure(api_key=google_api_key)

        defaults = {
            'model': 'models/text-bison-001',
            'temperature': 0.7,
            'candidate_count': 1,
            'top_k': 40,
            'top_p': 0.95,
            'max_output_tokens': 1024,
            'stop_sequences': [],
            'safety_settings': [
                {"category": "HARM_CATEGORY_DEROGATORY", "threshold": 1},
                {"category": "HARM_CATEGORY_TOXICITY", "threshold": 1},
                {"category": "HARM_CATEGORY_VIOLENCE", "threshold": 2},
                {"category": "HARM_CATEGORY_SEXUAL", "threshold": 2},
                {"category": "HARM_CATEGORY_MEDICAL", "threshold": 2},
                {"category": "HARM_CATEGORY_DANGEROUS", "threshold": 2},
            ],
        }

        query_input = st.text_area("Enter the SQL query:")
        prompt = f"You are a helpful assistant that generates SQL query for the users. {query_input}"

        if st.button("Generate SQL Query"):
            response = genai.generate_text(**defaults, prompt=prompt)
            st.write("Generated SQL Query:")
            st.write(response.result)

if __name__ == "__main__":
    main()
# scripts/load_data.py

import os
import psycopg2
import pandas as pd
from sqlalchemy import create_engine

def load_data_from_postgres(query):
    try:
        # Define your PostgreSQL connection parameters
        connection = psycopg2.connect(
            host="db",  # Replace with your host
            port=5432,  # Ensure this is an integer, not a string
            database="xdr_data",  # Replace with your database name
            user="postgres",  # Replace with your PostgreSQL username
            password="1234"  # Replace with your password
        )
        
        # Load data into a pandas DataFrame
        df = pd.read_sql(query, connection)
        
        # Close the connection
        connection.close()
        
        return df
    except Exception as e:
        print(f"An error occurred: {e}")
        return None




def load_data_using_sqlalchemy(query):
    """
    Connects to the PostgreSQL database and loads data based on the provided SQL query using SQLAlchemy.

    :param query: SQL query to execute.
    :return: DataFrame containing the results of the query.
    """
    try:
        # Create a connection string
        connection_string = f"postgresql+psycopg2://postgres:1234@db:5432/xdr_data"

        # Create an SQLAlchemy engine
        engine = create_engine(connection_string)

        # Load data into a pandas DataFrame
        df = pd.read_sql_query(query, engine)

        return df

    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
def load_data_to_postgres(df):
    try:
        # Define your PostgreSQL connection parameters
        connection = psycopg2.connect(
            host="db",  # Replace with your host
            port=5432,  # Ensure this is an integer, not a string
            database="xdr_data",  # Replace with your database name
            user="postgres",  # Replace with your PostgreSQL username
            password="1234"  # Replace with your password
        )
        
        table_name = 'user_satisfaction_scores'
        df.to_sql(table_name, con=connection, if_exists='replace', index=False)
        
        # Close the 
        connection.close()
     
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
    
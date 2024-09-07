import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data from PostgreSQL
conn = st.connection("teleco")
df = conn.query("SELECT * FROM xdr_data")

# Set page title
st.title("Telecom Data - Exploratory Data Analysis")

# Show first few rows of the dataset
st.subheader("Initial Data")
st.write(df.head())


from utils import convert_columns_to_mb,apply_ms_to_sec_and_drop
columns_to_convert = ['Start ms', 'End ms', 'Dur. (ms)', 'Avg RTT DL (ms)', 'Avg RTT UL (ms)',
                      'Activity Duration DL (ms)', 'Activity Duration UL (ms)']

apply_ms_to_sec_and_drop(columns_to_convert, df)
# List of columns to convert from Bytes to MB
columns_to_convert = [
    'HTTP UL (Bytes)', 'Social Media UL (Bytes)', 'Google UL (Bytes)', 'Youtube UL (Bytes)',
    'Netflix UL (Bytes)', 'Gaming UL (Bytes)', 'Total UL (Bytes)', 'HTTP DL (Bytes)',
    'Social Media DL (Bytes)', 'Google DL (Bytes)', 'Youtube DL (Bytes)', 'Netflix DL (Bytes)',
    'Gaming DL (Bytes)', 'Total DL (Bytes)', 'TCP DL Retrans. Vol (Bytes)', 'TCP UL Retrans. Vol (Bytes)',
    'Other DL (Bytes)', 'Other UL (Bytes)', 'Email DL (Bytes)', 'Email UL (Bytes)'
]
# Apply the function to convert all columns
convert_columns_to_mb(columns_to_convert, df)

def user_behaviour(df):
    applications = ['Social Media', 'Youtube', 'Netflix', 'Google', 'Email', 'Gaming', 'Other']
    for app in applications:
        df[f'{app} Total'] = df[f'{app} DL (MB)'] + df[f'{app} UL (MB)']
        user_behavior = df.groupby('IMSI').agg(
            number_of_xdr_sessions=('Bearer Id', 'count'),               # Assuming 'Bearer Id' is session identifier
            total_session_duration=('Dur. (sec)', 'sum'),                # Sum of session durations
            total_social_media_data=('Social Media Total', 'sum'),       # Total data volume for Social Media
            total_youtube_data=('Youtube Total', 'sum'),                 # Total data volume for YouTube
            total_netflix_data=('Netflix Total', 'sum'),                 # Total data volume for Netflix
            total_google_data=('Google Total', 'sum'),                   # Total data volume for Google
            total_email_data=('Email Total', 'sum'),                     # Total data volume for Email
            total_gaming_data=('Gaming Total', 'sum'),                   # Total data volume for Gaming
            total_other_data=('Other Total', 'sum')                      # Total data volume for Othe
            )
        user_behavior['total_download_data'] = df.groupby('IMSI')[[
            'Social Media DL (MB)', 'Youtube DL (MB)', 'Netflix DL (MB)',
            'Google DL (MB)', 'Email DL (MB)', 'Gaming DL (MB)', 'Other DL (MB)']].sum().sum(axis=1)
        user_behavior['total_upload_data'] = df.groupby('IMSI')[[
            'Social Media UL (MB)', 'Youtube UL (MB)', 'Netflix UL (MB)',
            'Google UL (MB)', 'Email UL (MB)', 'Gaming UL (MB)', 'Other UL (MB)']].sum().sum(axis=1)
        return user_behavior

# Option to drop or fill missing values
st.subheader("Handle Missing Values")
missing_column = st.selectbox("Choose a column to fill missing values", df.columns[df.isnull().any()])
fill_method = st.radio("Fill method", ["Fill with Mean", "Fill with Median", "Drop Rows"])

if st.button("Apply Fill"):
    if fill_method == "Fill with Mean":
        df[missing_column] = df[missing_column].fillna(df[missing_column].mean())
    elif fill_method == "Fill with Median":
        df[missing_column] = df[missing_column].fillna(df[missing_column].median())
    else:
        df = df.dropna(subset=[missing_column])
    st.success(f"{missing_column} cleaned successfully")

# ---- 2. Descriptive Statistics ----
st.subheader("Descriptive Statistics")
st.write(df.describe())

# ---- 3. Visualizations ----

# Handset Type Distribution
st.subheader("Univariate Analysis")
col1, col2 = st.columns(2)

# First plot: Top 10 Handset Types
with col1:
    st.write("Top 10 Handset Types")  # Add a title for the first plot
    top_handsets = df['Handset Type'].value_counts().head(10)
    st.bar_chart(top_handsets)

# Second plot: Top 3 Handset Manufacturers
with col2:  # Use col2 here for side-by-side display
    st.write("Top 3 Handset Manufacturers")  # Add a title for the second plot
    top_manufacturers = df['Handset Manufacturer'].value_counts().head(3)
    st.bar_chart(top_manufacturers)


# Define the applications to calculate total data usage (DL + UL)
st.subheader("TotalData usages by application")
applications = ['Social Media', 'Youtube', 'Netflix', 'Google', 'Email', 'Gaming', 'Other']

for app in applications:
    df[f'{app} Total'] = df[f'{app} DL (MB)'] + df[f'{app} UL (MB)']
total_usage = {app: df[f'{app} Total'].sum() for app in applications}

total_usage_df = pd.DataFrame(list(total_usage.items()), columns=['Application', 'Total Data Usage (MB)'])
# Plot the total data usage for each application
# Column layout in Streamlit
col1, col2 = st.columns(2)

# Plot 1: Bar Chart - Total Data Usage by Application
with col1:
    plt.figure(figsize=(10, 8))
    plt.bar(total_usage_df['Application'], total_usage_df['Total Data Usage (MB)'], color='skyblue')
    plt.title('Total Data Usage by Application')
    plt.xlabel('Application')
    plt.ylabel('Total Data Usage (MB)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Display the bar chart
    st.pyplot(plt)

# Plot 2: Pie Chart - Average Data Usage by Application
with col2:
    # Calculate average data usage by application
    user_behaviour_df = user_behaviour(df)
    avg_data_by_app = user_behaviour_df[['total_social_media_data', 'total_youtube_data', 'total_netflix_data',
                                         'total_google_data', 'total_email_data', 'total_gaming_data', 
                                         'total_other_data']].mean()

    plt.figure(figsize=(10, 8))
    avg_data_by_app.plot(kind='pie', autopct='%1.1f%%', colors=['skyblue', 'lightgreen', 'orange', 'lightcoral', 'purple', 'yellow', 'cyan'])
    plt.title('Average Data Usage by Application')
    plt.ylabel('')  # Hide y-label
    
    # Display the pie chart
    st.pyplot(plt)


# Assuming user_behaviour_df is already defined and loaded with necessary data

st.subheader("Bivariate Analysis")

# Create two columns for side-by-side comparison
col1, col2 = st.columns(2)

# Plot 1: Session Duration vs. Total Data Usage
with col1:
    plt.figure(figsize=(10, 6))
    plt.scatter(user_behaviour_df['total_session_duration'], 
                user_behaviour_df['total_download_data'] + user_behaviour_df['total_upload_data'], 
                alpha=0.5)
    plt.xlabel('Total Session Duration (sec)')
    plt.ylabel('Total Data Usage (MB)')
    plt.title('Session Duration vs. Total Data Usage')
    # Display the plot in Streamlit
    st.pyplot(plt)

# Plot 2: Download vs. Upload Data
with col2:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=user_behaviour_df, x='total_download_data', y='total_upload_data', alpha=0.6)
    plt.xlabel('Total Download Data')
    plt.ylabel('Total Upload Data')
    plt.title('Download vs Upload Data')
    # Display the plot in Streamlit
    st.pyplot(plt)

# Compute correlation matrix
st.subheader("Correlation Analysis")
correlation_matrix = user_behaviour_df[['total_download_data', 'total_upload_data',
                                    'total_social_media_data', 'total_youtube_data',
                                    'total_netflix_data', 'total_google_data',
                                    'total_email_data', 'total_gaming_data',
                                    'total_other_data']].corr()

# Plot correlation matrix
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
st.pyplot(plt)

# Step 1: Calculate session frequency (number of sessions) for each user (IMSI)
st.subheader("User Engagement Analysis")
user_engagement = df.groupby('IMSI').agg(
    session_frequency=('Bearer Id', 'count'))
# Calculate total session duration per user
user_engagement['total_session_duration'] = df.groupby('IMSI')['Dur. (sec)'].sum()
# Optionally, calculate the average session duration
user_engagement['avg_session_duration'] = df.groupby('IMSI')['Dur. (sec)'].mean()
# Calculate total traffic (download + upload)
user_engagement['total_traffic'] = user_behaviour_df['total_download_data'] + user_behaviour_df['total_upload_data']

# Step 2: Sort the data by session frequency and get the top 10 users
top_10_users = user_engagement.sort_values(by='session_frequency', ascending=False).head(10)

col1, col2 = st.columns(2)

with col1:
    plt.figure(figsize=(10, 6))
    sns.barplot(x='IMSI', y='session_frequency', data=top_10_users, palette='Blues_d')
    plt.title('Top 10 Users by Session Frequency')
    plt.xlabel('User (IMSI)')
    plt.ylabel('Session Frequency')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)

# Step 2: Sort the data by session frequency and get the top 10 users
top_10_users = user_engagement.sort_values(by='total_session_duration', ascending=False).head(10)
with col2:
    plt.figure(figsize=(10, 6))
    sns.barplot(x='IMSI', y='total_session_duration', data=top_10_users, palette='Blues_d')
    plt.title('Top 10 Users by Session Duration')
    plt.xlabel('User (IMSI)')
    plt.ylabel('Session Duration')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)
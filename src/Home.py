import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import euclidean_distances


# Load data from PostgreSQL
conn = st.connection("teleco")
df = conn.query("SELECT * FROM xdr_data")

# Function to handle missing values
def handle_missing_values(data):
    with st.sidebar:
        st.subheader("Handle Missing Values")
        missing_column = st.selectbox("Choose a column to fill missing values", data.columns[data.isnull().any()])
        fill_method = st.radio("Fill method", ["Fill with Mean", "Fill with Median", "Drop Rows"])
        
        if st.button("Apply Fill"):
            if fill_method == "Fill with Mean":
                data[missing_column] = data[missing_column].fillna(data[missing_column].mean())
            elif fill_method == "Fill with Median":
                data[missing_column] = data[missing_column].fillna(data[missing_column].median())
            else:
                data = data.dropna(subset=[missing_column])
            st.success(f"{missing_column} cleaned successfully")
    return data

# Handle missing values
df = handle_missing_values(df)

# Function to convert columns to MB
from utils import convert_columns_to_mb, apply_ms_to_sec_and_drop
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
convert_columns_to_mb(columns_to_convert, df)

# Define the applications for total data usage calculation
def calculate_user_behavior(data):
    applications = ['Social Media', 'Youtube', 'Netflix', 'Google', 'Email', 'Gaming', 'Other']
    for app in applications:
        data[f'{app} Total'] = data[f'{app} DL (MB)'] + data[f'{app} UL (MB)']
    
    user_behavior = df.groupby('IMSI').agg(
        number_of_xdr_sessions=('Bearer Id', 'count'),
        total_session_duration=('Dur. (sec)', 'sum'),
        total_social_media_data=('Social Media Total', 'sum'),
        total_youtube_data=('Youtube Total', 'sum'),
        total_netflix_data=('Netflix Total', 'sum'),
        total_google_data=('Google Total', 'sum'),
        total_email_data=('Email Total', 'sum'),
        total_gaming_data=('Gaming Total', 'sum'),
        total_other_data=('Other Total', 'sum')
    )
    user_behavior['total_download_data'] = data.groupby('IMSI')[[
        'Social Media DL (MB)', 'Youtube DL (MB)', 'Netflix DL (MB)',
        'Google DL (MB)', 'Email DL (MB)', 'Gaming DL (MB)', 'Other DL (MB)']].sum().sum(axis=1)
    user_behavior['total_upload_data'] = data.groupby('IMSI')[[
        'Social Media UL (MB)', 'Youtube UL (MB)', 'Netflix UL (MB)',
        'Google UL (MB)', 'Email UL (MB)', 'Gaming UL (MB)', 'Other UL (MB)']].sum().sum(axis=1)
    return user_behavior

# Function to perform clustering
def clustering(data, n_clusters=3):
    # Select relevant features for clustering
    features = data[['session_frequency', 'total_session_duration', 'total_traffic']]
    
    # Standardize the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    data['cluster'] = kmeans.fit_predict(scaled_features)
    
    return data

user_behavior_df = calculate_user_behavior(df)
# Sidebar menu
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a Page", ["User Behaviour Analysis", "User Engagement Analysis", "Experiance Analysis", "Satisfaction Analysis"])
st.title("Descriptive Statistics")
st.write(df.describe())

if page == "User Behaviour Analysis":
    st.title("User Behaviour Analysis")

    # Handset Type Distribution
    st.subheader("Univariate Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Top 10 Handset Types")
        top_handsets = df['Handset Type'].value_counts().head(10)
        plt.figure(figsize=(10, 6))
        top_handsets.plot(kind='bar', color='skyblue')
        plt.title('Top 10 Handset Types')
        plt.xlabel('Handset Type')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(plt)
        plt.clf()  # Clear the plot
    
    with col2:
        st.write("Top 3 Handset Manufacturers")
        top_manufacturers = df['Handset Manufacturer'].value_counts().head(3)
        plt.figure(figsize=(10, 6))
        top_manufacturers.plot(kind='bar', color='lightgreen')
        plt.title('Top 3 Handset Manufacturers')
        plt.xlabel('Manufacturer')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(plt)
        plt.clf()  # Clear the plot
    
    # Bivariate Analysis
    st.subheader("Bivariate Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        plt.figure(figsize=(10, 6))
        plt.scatter(user_behavior_df['total_session_duration'], 
                    user_behavior_df['total_download_data'] + user_behavior_df['total_upload_data'], 
                    alpha=0.5)
        plt.xlabel('Total Session Duration (sec)')
        plt.ylabel('Total Data Usage (MB)')
        plt.title('Session Duration vs. Total Data Usage')
        plt.tight_layout()
        st.pyplot(plt)
        plt.clf()  # Clear the plot
    
    with col2:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=user_behavior_df, x='total_download_data', y='total_upload_data', alpha=0.6)
        plt.xlabel('Total Download Data')
        plt.ylabel('Total Upload Data')
        plt.title('Download vs Upload Data')
        plt.tight_layout()
        st.pyplot(plt)
        plt.clf()  # Clear the plot

    # Total Data Usages by Application
    st.subheader("Total Data Usages by Application")
    applications = ['Social Media', 'Youtube', 'Netflix', 'Google', 'Email', 'Gaming', 'Other']
    total_usage = {app: df[f'{app} Total'].sum() for app in applications}
    total_usage_df = pd.DataFrame(list(total_usage.items()), columns=['Application', 'Total Data Usage (MB)'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        plt.figure(figsize=(10, 8))
        plt.bar(total_usage_df['Application'], total_usage_df['Total Data Usage (MB)'], color='skyblue')
        plt.title('Total Data Usage by Application')
        plt.xlabel('Application')
        plt.ylabel('Total Data Usage (MB)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(plt)
        plt.clf()  # Clear the plot
    
    with col2:
        avg_data_by_app = user_behavior_df[['total_social_media_data', 'total_youtube_data', 'total_netflix_data',
                                             'total_google_data', 'total_email_data', 'total_gaming_data', 
                                             'total_other_data']].mean()
        plt.figure(figsize=(10, 8))
        avg_data_by_app.plot(kind='pie', autopct='%1.1f%%', colors=['skyblue', 'lightgreen', 'orange', 'lightcoral', 'purple', 'yellow', 'cyan'])
        plt.title('Average Data Usage by Application')
        plt.ylabel('')
        plt.tight_layout()
        st.pyplot(plt)
        plt.clf()  # Clear the plot

    # Correlation Analysis
    st.subheader("Correlation Analysis")
    correlation_matrix = user_behavior_df[['total_download_data', 'total_upload_data',
                                    'total_social_media_data', 'total_youtube_data',
                                    'total_netflix_data', 'total_google_data',
                                    'total_email_data', 'total_gaming_data',
                                    'total_other_data']].corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.tight_layout()
    st.pyplot(plt)
    plt.clf()  # Clear the plot


elif page == "User Engagement Analysis":
    st.title("User Engagement Analysis")
    
    # Calculate user engagement metrics
    user_engagement = df.groupby('IMSI').agg(
        session_frequency=('Bearer Id', 'count'),
        total_session_duration=('Dur. (sec)', 'sum'),
        avg_session_duration=('Dur. (sec)', 'mean')
    )
    user_engagement['total_traffic'] = user_behavior_df['total_download_data'] + user_behavior_df['total_upload_data']
    
    # Top 10 Users by Session Frequency
    top_10_users_frequency = user_engagement.sort_values(by='session_frequency', ascending=False).head(10)
    col1, col2 = st.columns(2)
    
    with col1:
        plt.figure(figsize=(10, 6))
        sns.barplot(x=top_10_users_frequency.index, y='session_frequency', data=top_10_users_frequency, palette='Blues_d')
        plt.title('Top 10 Users by Session Frequency')
        plt.xlabel('User (IMSI)')
        plt.ylabel('Session Frequency')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(plt)
    
    # Top 10 Users by Session Duration
    top_10_users_duration = user_engagement.sort_values(by='total_session_duration', ascending=False).head(10)
    with col2:
        plt.figure(figsize=(10, 6))
        sns.barplot(x=top_10_users_duration.index, y='total_session_duration', data=top_10_users_duration, palette='Blues_d')
        plt.title('Top 10 Users by Session Duration')
        plt.xlabel('User (IMSI)')
        plt.ylabel('Session Duration')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(plt)

    st.subheader("User Engagement score")
    scaler = StandardScaler()
    user_engagement[['session_frequency', 'total_session_duration', 'total_traffic']] = scaler.fit_transform(
    user_engagement[['session_frequency', 'total_session_duration', 'total_traffic']])
    
    user_engagement['engagement_score'] = (
    0.4 * user_engagement['session_frequency'] +
    0.3 * user_engagement['total_session_duration'] +
    0.3 * user_engagement['total_traffic'])
    
    top_engaged_users = user_engagement.sort_values(by='engagement_score', ascending=False).head(10)
    plt.figure(figsize=(20, 6))
    sns.barplot(x=top_engaged_users.index, y=top_engaged_users['engagement_score'])
    plt.title('Top 10 Engaged Users')
    plt.xlabel('User (IMSI)')
    plt.ylabel('Engagement Score')
    st.pyplot(plt)
    
    st.subheader("User Clustering")
    # Apply clustering on user engagement data
    user_engagement = clustering(user_engagement)
    
    # Cluster summary
    cluster_summary = user_engagement.groupby('cluster').agg({
        'session_frequency': ['min', 'max', 'mean', 'sum'],
        'total_session_duration': ['min', 'max', 'mean', 'sum'],
        'total_traffic': ['min', 'max', 'mean', 'sum']
    })
    st.subheader("Cluster Summary")
    st.write(cluster_summary)
    
    # Prepare data for plotting
    cluster_summary_df = cluster_summary.copy().reset_index()
    
    # Plot mean values for each cluster
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 15), sharex=True)
    metrics = ['session_frequency', 'total_session_duration', 'total_traffic']
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        sns.barplot(x='cluster', y=(metric, 'mean'), data=cluster_summary_df, ax=ax)
        ax.set_title(f'Mean {metric.replace("_", " ").title()} by Cluster')
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Mean Value')
    
    plt.tight_layout()
    st.pyplot(fig)


elif page == "Experiance Analysis":
    st.title("Experiance Analysis")
    handset_grouped = df.groupby('Handset Type').agg({
        'TCP DL Retrans. Vol (MB)': 'mean',
        'TCP UL Retrans. Vol (MB)': 'mean',
        'Avg Bearer TP DL (kbps)': 'mean',
        'Avg Bearer TP UL (kbps)': 'mean',
        'Handset Type': 'count'  }).rename(columns={'Handset Type': 'Count'}).reset_index()

    handset_grouped.columns = [
        'Handset Type',
        'Avg Downlink Throughput (MB)',
        'Avg Uplink Throughput (MB)',
        'Avg DL TCP Retransmission (MB)',
        'Avg UL TCP Retransmission (MB)',
       'Count'  # Adding count for each handset type
       ]
    
    handset_grouped_sorted = handset_grouped.sort_values(by='Count', ascending=False).head(10)
    
    plt.figure(figsize=(10, 4))
    sns.barplot(data=handset_grouped_sorted, x='Avg Downlink Throughput (MB)', y='Handset Type')
    plt.title('Average Downlink Throughput per Top 10 Handset Types')
    plt.xlabel('Average Downlink Throughput (MB)')
    plt.ylabel('Handset Type')
    st.pyplot(plt)
    
    plt.figure(figsize=(10, 4))
    sns.barplot(data=handset_grouped_sorted, x='Avg DL TCP Retransmission (MB)', y='Handset Type')
    plt.title('Average DL TCP Retransmission per Top 10 Handset Types')
    plt.xlabel('Average TCP DL Retransmission (MB)')
    plt.ylabel('Handset Type')
    st.pyplot(plt)


elif page == "Satisfaction Analysis":
    st.title("Visualizations")
    # Group by customer (IMSI) and calculate the required averages
    customer_aggregation = df.groupby('IMSI').agg({
        'TCP DL Retrans. Vol (MB)': 'mean',
        'TCP UL Retrans. Vol (MB)': 'mean',
        'Avg RTT UL (sec)': 'mean',
        'Avg RTT DL (sec)': 'mean',
        'Avg Bearer TP DL (kbps)': 'mean',
        'Avg Bearer TP UL (kbps)': 'mean',
        'Handset Type': 'first'  # Keep the first occurrence of the handset typ
        }).reset_index()
    
    customer_aggregation.columns = [
        'IMSI',
        'Avg TCP DL Retransmission (MB)',
        'Avg TCP UL Retransmission (MB)',
        'Avg RTT UL (sec)',
        'Avg RTT DL (sec)',
        'Avg Bearer TP DL (kbps)',
        'Avg Bearer TP UL (kbps)',
        'Handset Type']
    
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
        total_other_data=('Other Total', 'sum')                      # Total data volume for Other
        )
    
    user_behavior['total_download_data'] = df.groupby('IMSI')[[
        'Social Media DL (MB)', 'Youtube DL (MB)', 'Netflix DL (MB)',
        'Google DL (MB)', 'Email DL (MB)', 'Gaming DL (MB)', 'Other DL (MB)']].sum().sum(axis=1)
    
    user_behavior['total_upload_data'] = df.groupby('IMSI')[[
        'Social Media UL (MB)', 'Youtube UL (MB)', 'Netflix UL (MB)',
        'Google UL (MB)', 'Email UL (MB)', 'Gaming UL (MB)', 'Other UL (MB)']].sum().sum(axis=1)
    
    # If IMSI is not already a column, reset the index to make IMSI a column
    customer_aggregation_reset = customer_aggregation.reset_index()
    user_behavior_reset = user_behavior.reset_index()
    
    merged_data = pd.merge(customer_aggregation_reset, user_behavior_reset, on='IMSI', how='inner')
    merged_data['total_traffic'] = merged_data['total_download_data'] + merged_data['total_upload_data']
    engagement_features = ['number_of_xdr_sessions', 'total_session_duration', 'total_traffic']
    experience_features = ['Avg Bearer TP UL (kbps)', 'Avg Bearer TP DL (kbps)',
                           'Avg TCP DL Retransmission (MB)', 'Avg TCP UL Retransmission (MB)',
                            'Avg RTT UL (sec)', 'Avg RTT DL (sec)']
    kmeans_engagement = KMeans(n_clusters=3, random_state=42)
    kmeans_engagement.fit(merged_data[engagement_features])
    engagement_cluster_centers = kmeans_engagement.cluster_centers_

    # Perform K-means clustering for experience features
    kmeans_experience = KMeans(n_clusters=3, random_state=42)
    kmeans_experience.fit(merged_data[experience_features])
    experience_cluster_centers = kmeans_experience.cluster_centers_
    
    engagement_cluster_0 = engagement_cluster_centers[0]  # 3D cluster center for engagement
    experience_cluster_0 = experience_cluster_centers[0]  # 6D cluster center for experience
    
    merged_data['Engagement Score'] = euclidean_distances( merged_data[engagement_features], engagement_cluster_0.reshape(1, -1))
    
    merged_data['Experience Score'] = euclidean_distances(merged_data[experience_features],experience_cluster_0.reshape(1, -1))

    # Step 6: Calculate Satisfaction Score (average of engagement and experience scores)
    merged_data['Satisfaction Score'] = (merged_data['Engagement Score'] + merged_data['Experience Score']) / 2
    
    merged_data_sorted = merged_data.sort_values(by='Satisfaction Score', ascending=True)
    top_10_satisfied_customers = merged_data_sorted.head(10)

    # Step 9: Plot bar graph
    plt.figure(figsize=(10, 6))
    plt.bar(top_10_satisfied_customers['IMSI'].astype(str), top_10_satisfied_customers['Satisfaction Score'], color='skyblue')
    plt.xlabel('IMSI')
    plt.ylabel('Satisfaction Score')
    plt.title('Top 10 Most Satisfied Customers')
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
    plt.tight_layout()  # Adjust layout to prevent clipping of labels



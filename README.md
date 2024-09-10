# telecommunication-data-analysis
**Overview**
* In today’s highly competitive industry, companies must leverage data-driven insights to identify opportunities for growth and profitability.  As part of a strategic review for a wealthy investor specializing in undervalued assets, this project seeks to uncover fundamental drivers of TellCo’s business through an in-depth analysis of customer behavior and service usage.

* The primary objective of this project is to analyze TellCo’s customer and network data to identify key growth opportunities and potential profitability improvements. This analysis will evaluate customer engagement patterns, service usage, and potential segments that could drive revenue growth. The goal is to provide actionable recommendations on whether the investor should purchase TellCo and how to maximize its future profitability through focused, data-driven strategies.

**Data Description**
> The telecommunication dataset from TellCo provides detailed insights into customer usage and network performance. Below is a summary of key variables and their descriptions:

- IMSI (International Mobile Subscriber Identity): A unique identifier for each user in the network.
- MSISDN/Number: The mobile phone number associated with a specific subscriber.
- Bearer Id: A unique identifier for each data session (xDR).
- Dur. (ms): Duration of a data session in milliseconds.
- Total DL (Bytes): Total data downloaded during a session in bytes.
- Total UL (Bytes): Total data uploaded during a session in bytes.
- Social Media DL (MB): Volume of social media data downloaded (in MB).
- Social Media UL (MB): Volume of social media data uploaded (in MB).
- YouTube DL (MB): Volume of YouTube data downloaded (in MB).
- YouTube UL (MB): Volume of YouTube data uploaded (in MB).
- Netflix DL (MB): Volume of Netflix data downloaded (in MB).
- Netflix UL (MB): Volume of Netflix data uploaded (in MB).
- Google DL (MB): Volume of Google data downloaded (in MB).
- Google UL (MB): Volume of Google data uploaded (in MB).
- Email DL (MB): Volume of email data downloaded (in MB).
- Email UL (MB): Volume of email data uploaded (in MB).
- Gaming DL (MB): Volume of gaming data downloaded (in MB).
- Gaming UL (MB): Volume of gaming data uploaded (in MB).
- Other DL (MB): Volume of other data downloaded (in MB).
- Other UL (MB): Volume of other data uploaded (in MB).
- Handset Type: The type or model of the mobile handset used by the customer.
- Handset Manufacturer: The manufacturer of the mobile handset.
- Avg RTT DL (ms): Average round-trip time for data download in milliseconds.
- Avg RTT UL (ms): Average round-trip time for data upload in milliseconds.
- TCP DL Retrans. Vol (Bytes): The volume of re-transmitted download data.
- TCP UL Retrans. Vol (Bytes): The volume of re-transmitted upload data.
> These variables provide essential insights into user behavior, session duration, data consumption, and network performance, all of which are critical for identifying potential areas of growth and optimization for TellCo.

**Contents**
* Data base connection: extraction and retrival of data from database (postgresql)
* Exploratory Data Analysis (EDA): to understand the dataset and identify the missing values & outliers if any using visual and quantitative methods to get a sense of the story it tells. 
* User Overview Analysis: to track user behavior through the following applications:  Social Media, Google, Email, YouTube, Netflix, Gaming, and others.
* User Engagement Analysis: to track the user’s engagement using the following engagement metrics: 
   > - sessions frequency 
   > - the duration of the session 
   > - the session total traffic (download and upload (bytes))
* Experience Analytics: Tracking & evaluating customers’ experience 
* Satisfaction Analysis: satisfaction of a user is depending on user engagement and experience
* Dashboards

**Prerequisites**
* Python 3.x: Ensure Python is installed on your system.
* Virtual Environment: Recommended for managing project dependencies.
* Required Libraries:
- pandas: Data manipulation and analysis. 
- numpy: Numerical operations. 
- matplotlib: Data visualization. 
- seaborn: Statistical visualizations.
- scikit-learn
- psycopg2: data base connection
-postgresql: storing data

**Installation**

1. Create a virtual environment:
On macOS/Linux:

```python -m venv venv```
```source venv/bin/activate```


on windows:
```python -m venv venv ```
```venv\Scripts\activate ```

2. Install dependencies:
``` pip install -r requirements.txt```

**Contributing**

Contributions are welcome!

**License**

This project is licensed under the Apache License. See the LICENSE file for more details.
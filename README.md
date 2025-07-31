# EV Stations and Chargers Monitoring, Analytical Dashboards service

## About the tool
This Service utilizes Eviny's public API, creating a full ETL that extracts, transforms, validates, and loads data to a MySQL cloud Data Warehouse (Railway) every minute. 
The dashboard service also extracts the data every minute to give real-time-like insights.

## Purpose
The whole purpose of this project is to create a platform that could be used for monitoring and insight in real time. 
It includes the status of the stations and charges, the map, the revenue, the power usage, the trends, etc.

## Limitations
For this project a free tools and services like Streamlit and Railway are used, and the ETL runs on my second PC, which brings some limitations to it. 
The API, combined with 1-minute extracts, creates thousands of rows, and after an hour, the database has millions of records. Because of that, 
I have to delete records quite often. The second limitation is the responsiveness of the free Streamlit app, which has lots of functionality and can handle lots of data can make the webpage slow. 
> [!NOTE]
> That's why if some parts don't load, just click the button again, and if it's still showing the previous page, switch to any other page and click the button you're interested in.

I apologize for any unpleasant experience you might have and currently working on the optimization. 


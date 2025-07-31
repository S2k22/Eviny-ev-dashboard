# EV Stations and Chargers Monitoring, Analytical Dashboards Service using Eviny APi

## 📝 About the Tool

This project is a **full-featured analytics dashboard and ETL pipeline** for monitoring and analyzing Electric Vehicle (EV) charging station data.  
The system enables real-time visualization, revenue analysis, station status monitoring, and usage analytics for EV charging networks. All the data is being extracted from a public Eviny API.

### Use Cases
- Charging network operators
- Facility managers
- Data analysts and researchers
- Public charging reporting

---

## ✨ Features

- **Live Data Sync:** Auto-refreshing dashboard for real-time monitoring of all connected charging stations.
- **MySQL Cloud Backend:** Robust storage of stations, utilization, and session data.
- **Detailed Analytics:** Usage heatmaps, occupancy trends, revenue analysis, session breakdowns.
- **Data Explorer:** Flexible filtering and export (CSV/Excel) of any dataset.
- **ETL Pipeline:** Automated data extraction, transformation, and loading scripts for ingesting data from APIs.
- **Historical Analysis:** Aggregated hourly/daily statistics, top-performing stations, and connector type breakdown.
- **Deployment-Ready:** Designed for Railway, Streamlit Cloud, or local Docker deployment.
- **Fault-Tolerant:** Handles database downtime and provides clear error messages.
- **Admin Tools:** CLI scripts for cleaning and resetting Railway volumes or MySQL databases.

---

## ⚠️ Limitations

For this project, **free tools and services** like Streamlit and Railway are used, and the ETL runs on my secondary PC, which brings some limitations:

- The API, combined with 1-minute extracts, creates thousands of rows. After an hour, the database can have millions of records.  
  👉 Because of that, I have to delete records quite often.
- The **responsiveness of the free Streamlit app**: The app has lots of functionality and can handle a lot of data, which can make the webpage slow.

> **Note:**  
> If some parts don't load, just click the button again. If it's still showing the previous page, switch to any other page and click the button you're interested in.

I apologize for any unpleasant experience you might have. I am currently working on optimization!

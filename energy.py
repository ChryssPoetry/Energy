import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


# Example Streamlit app code
st.title("World Energy consumption")
st.write("By Nwafor Franklin Nnemeka")

@st.cache_data
def load_data():
    # Load and reshape the dataset
    data = pd.read_csv("https://github.com/ChryssPoetry/Energy/blob/main/World_Energy_By_Country_And_Region_1965_to_2023.csv")  # Replace with your file path
    data = pd.melt(
        data,
        id_vars=["Country"],          # Keep 'Country' column as is
        var_name="Year",              # New column name for the years
        value_name="Energy Consumption"  # New column name for energy consumption values
    )
    # Clean and convert data types
    data["Year"] = pd.to_numeric(data["Year"], errors="coerce")
    data["Energy Consumption"] = pd.to_numeric(data["Energy Consumption"], errors="coerce").fillna(0)
    data["Country"] = data["Country"].fillna("Unknown").astype(str)

    # Simulate renewable energy data for this example
    data["Renewable Energy"] = data["Energy Consumption"] * (0.1 + 0.4 * (data["Year"].astype(int) % 5 == 0))  # Example synthetic data
    data["Renewable Percentage"] = data["Renewable Energy"] / data["Energy Consumption"]
    data["Renewable Percentage"] = data["Renewable Percentage"].fillna(0)
    data["Transitioning"] = (data["Renewable Percentage"] >= 0.5).astype(int)  # Binary classification target
    
    return data

# Loading the reshaped dataset
data = load_data()

# Streamlit App Title and Description
st.write("Explore trends and insights in global energy consumption over time.")

# Sidebar Filters
st.sidebar.header("Filter options")
selected_year = st.sidebar.slider("Select Year", int(data["Year"].min()), int(data["Year"].max()), 2020)
selected_country = st.sidebar.selectbox("Select Country", ["All"] + sorted(data["Country"].unique()))

# Filter the Data Based on User Selection
filtered_data = data[data["Year"] == selected_year]
if selected_country != "All":
    filtered_data = filtered_data[filtered_data["Country"] == selected_country]

# Display Filtered Data
st.subheader("Filtered data")
st.write(filtered_data)

# Total Energy Consumption Over Time
st.subheader("Total energy consumption over time")
if st.checkbox("Show trends"):
    yearly_consumption = data.groupby("Year")["Energy Consumption"].sum()
    fig, ax = plt.subplots(figsize=(10, 6))
    yearly_consumption.plot(kind="line", ax=ax, color="blue")
    ax.set_title("Total energy consumption over time")
    ax.set_xlabel("Year")
    ax.set_ylabel("Energy Consumption")
    st.pyplot(fig)

# Top Energy Consumers for the Selected Year
st.subheader(f"Top Energy Consumers in {selected_year}")
top_consumers = filtered_data.sort_values("Energy Consumption", ascending=False).head(10)
fig, ax = plt.subplots(figsize=(10, 6))
top_consumers.set_index("Country")["Energy Consumption"].plot(kind="bar", ax=ax, color="orange")
ax.set_title(f"Top 10 Energy Consuming Countries in {selected_year}")
ax.set_ylabel("Energy Consumption")
st.pyplot(fig)

# Regional Trends
if st.checkbox("Show regional trends"):
    st.subheader("Regional energy consumption trends")
    regional_data = data[data["Country"].str.contains("Total", na=False)]
    fig, ax = plt.subplots(figsize=(10, 6))
    for region in regional_data["Country"].unique():
        region_data = regional_data[regional_data["Country"] == region]
        ax.plot(region_data["Year"], region_data["Energy Consumption"], label=region)
    ax.set_title("Energy consumption trends by region")
    ax.set_xlabel("Year")
    ax.set_ylabel("Energy Consumption")
    ax.legend()
    st.pyplot(fig)

# Country-Level Trends
# Country-Level Trends
st.subheader("Energy consumption trends for selected countries")

# Allow user to select countries for comparison
countries = st.multiselect(
    "Select Countries to Compare", 
    options=data["Country"].unique(), 
    default=["US", "China", "India"]  # Default countries
)

# Only display the plot if at least one country is selected
if countries:
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot energy consumption trends for each selected country
    for country in countries:
        country_data = data[data["Country"] == country]
        ax.plot(
            country_data["Year"], 
            country_data["Energy Consumption"], 
            label=country
        )

    ax.set_title("Energy consumption trends for selected countries")
    ax.set_xlabel("Year")
    ax.set_ylabel("Energy Consumption")
    ax.legend(title="Country")
    st.pyplot(fig)
else:
    st.write("Please select at least one country to display the comparison.")

# Streamlit App
st.title("World energy transition classification")
st.write("Classify countries as transitioning to renewables or dependent on fossil fuels.")


# Filter data for the selected year
filtered_data = data[data["Year"] == selected_year]
if selected_country != "All":
    filtered_data = filtered_data[filtered_data["Country"] == selected_country]

# Show filtered data
st.subheader(f"Filtered Data for {selected_year}")
st.write(filtered_data)

# Classification Model
st.subheader("Train classification model")

if st.button("Train Model"):
    # Prepare data for modeling
    features = data[["Year", "Energy Consumption", "Renewable Percentage"]]
    target = data["Transitioning"]
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
    # Train a Random Forest model
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Predictions and evaluation
    y_pred = rf_model.predict(X_test)
    st.write("### Classification Report")
    st.text(classification_report(y_test, y_pred))
    
    st.write("### Confusion Matrix")
    st.write(pd.DataFrame(confusion_matrix(y_test, y_pred), columns=["Not Transitioning", "Transitioning"], index=["Not Transitioning", "Transitioning"]))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        "Feature": features.columns,
        "Importance": rf_model.feature_importances_
    }).sort_values(by="Importance", ascending=False)
    
    st.write("### Feature Importance")
    st.write(feature_importance)

# Visualize Trends
st.subheader("Transition Trends Over Time")
if st.checkbox("Show Transition Trends"):
    yearly_transitions = data.groupby("Year")["Transitioning"].mean()
    fig, ax = plt.subplots(figsize=(10, 6))
    yearly_transitions.plot(kind="line", ax=ax, color="green")
    ax.set_title("Percentage of countries transitioning to renewables over Time")
    ax.set_xlabel("Year")
    ax.set_ylabel("Percentage Transitioning")
    st.pyplot(fig)


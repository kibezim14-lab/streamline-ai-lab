"""
Streamlit App for Exploratory Data Analysis of the Iris Dataset

This app allows users to interactively explore the Iris dataset with:
- Data preview and summary statistics
- Histogram visualization for individual columns
- Scatter plot comparison between two columns
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris


def load_data():
    """
    Load the Iris dataset from scikit-learn and convert to DataFrame.
    
    Returns:
        pd.DataFrame: Iris dataset with feature names as columns
    """
    # Load the iris dataset
    iris = load_iris()
    
    # Create a DataFrame with feature names as columns
    data = pd.DataFrame(iris.data, columns=iris.feature_names)
    
    return data


def main():
    """Main function to run the Streamlit app."""
    
    # Set page title and configuration
    st.set_page_config(page_title="Iris Dataset EDA", layout="wide")
    st.title("🌸 Iris Dataset - Exploratory Data Analysis")
    st.markdown("Interactive exploration of the Iris dataset with visualizations")
    
    # Load the data
    data = load_data()
    
    # Display dataset information in the sidebar
    st.sidebar.header("Dataset Info")
    st.sidebar.metric("Number of Rows", len(data))
    st.sidebar.metric("Number of Columns", len(data.columns))
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Data Preview", "📈 Summary Statistics", "📉 Histogram", "📍 Scatter Plot"]
    )
    
    # ===== TAB 1: DATA PREVIEW =====
    with tab1:
        st.subheader("First Rows of the Dataset")
        
        # Slider to select number of rows to display
        num_rows = st.slider(
            "Select number of rows to display:",
            min_value=1,
            max_value=len(data),
            value=5
        )
        
        # Display the data
        st.dataframe(data.head(num_rows), use_container_width=True)
    
    # ===== TAB 2: SUMMARY STATISTICS =====
    with tab2:
        st.subheader("Summary Statistics")
        
        # Display descriptive statistics
        summary_stats = data.describe()
        st.dataframe(summary_stats, use_container_width=True)
        
        # Additional statistics
        st.subheader("Additional Information")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Numeric Columns", len(data.select_dtypes(include=np.number).columns))
        with col2:
            st.metric("Mean Values Range", f"{data.mean().min():.2f} - {data.mean().max():.2f}")
        with col3:
            st.metric("Std Dev Range", f"{data.std().min():.2f} - {data.std().max():.2f}")
        with col4:
            st.metric("Total Data Points", len(data) * len(data.columns))
    
    # ===== TAB 3: HISTOGRAM =====
    with tab3:
        st.subheader("Histogram Visualization")
        
        # Get numeric columns
        numeric_columns = data.select_dtypes(include=np.number).columns.tolist()
        
        # Column selector for histogram
        selected_column = st.selectbox(
            "Select a column to display histogram:",
            options=numeric_columns,
            key="histogram_column"
        )
        
        # Number of bins slider
        bins = st.slider(
            "Select number of bins:",
            min_value=5,
            max_value=50,
            value=20,
            key="histogram_bins"
        )
        
        # Create and display histogram
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(data[selected_column], bins=bins, color='steelblue', edgecolor='black', alpha=0.7)
        ax.set_xlabel(selected_column, fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_title(f"Histogram of {selected_column}", fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        st.pyplot(fig)
    
    # ===== TAB 4: SCATTER PLOT =====
    with tab4:
        st.subheader("Scatter Plot Visualization")
        
        # Get numeric columns
        numeric_columns = data.select_dtypes(include=np.number).columns.tolist()
        
        # Create two columns for x and y selection
        col1, col2 = st.columns(2)
        
        with col1:
            x_column = st.selectbox(
                "Select X-axis column:",
                options=numeric_columns,
                key="scatter_x"
            )
        
        with col2:
            # Default to second column if available, otherwise first
            default_y = numeric_columns[1] if len(numeric_columns) > 1 else numeric_columns[0]
            y_column = st.selectbox(
                "Select Y-axis column:",
                options=numeric_columns,
                index=numeric_columns.index(default_y),
                key="scatter_y"
            )
        
        # Create and display scatter plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(data[x_column], data[y_column], color='steelblue', alpha=0.6, s=100, edgecolors='black', linewidth=0.5)
        ax.set_xlabel(x_column, fontsize=12)
        ax.set_ylabel(y_column, fontsize=12)
        ax.set_title(f"Scatter Plot: {x_column} vs {y_column}", fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        
        # Display correlation between selected columns
        correlation = data[x_column].corr(data[y_column])
        st.info(f"📊 **Correlation between {x_column} and {y_column}: {correlation:.4f}**")


if __name__ == "__main__":
    main()

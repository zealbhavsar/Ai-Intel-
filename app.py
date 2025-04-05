import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import datetime
import joblib
import os

# Set page configuration
st.set_page_config(page_title="Delivery Time Predictor", layout="wide")

# Function to generate synthetic data
def generate_sample_data(n_samples=1000):
    np.random.seed(42)
    
    product_categories = ['Electronics', 'Clothing', 'Books', 'Home & Kitchen', 'Toys']
    locations = ['North America', 'Europe', 'Asia', 'South America', 'Australia']
    shipping_methods = ['Standard', 'Express', 'Priority', 'Same Day']
    
    data = {
        'product_category': np.random.choice(product_categories, n_samples),
        'customer_location': np.random.choice(locations, n_samples),
        'shipping_method': np.random.choice(shipping_methods, n_samples),
        'product_weight_kg': np.round(np.random.uniform(0.1, 20, n_samples), 2),
        'distance_km': np.random.randint(50, 15000, n_samples),
        'order_value': np.round(np.random.uniform(10, 1000, n_samples), 2)
    }
    
    # Create delivery time based on features with some randomness
    delivery_time = np.zeros(n_samples)
    
    # Base delivery times by shipping method
    method_times = {'Standard': 7, 'Express': 3, 'Priority': 2, 'Same Day': 1}
    
    # Location factors
    location_factors = {'North America': 1.0, 'Europe': 1.2, 'Asia': 1.5, 
                        'South America': 1.8, 'Australia': 2.0}
    
    # Product category complexity factors
    category_factors = {'Electronics': 1.2, 'Clothing': 1.0, 'Books': 1.0, 
                       'Home & Kitchen': 1.3, 'Toys': 1.1}
    
    for i in range(n_samples):
        base_time = method_times[data['shipping_method'][i]]
        loc_factor = location_factors[data['customer_location'][i]]
        cat_factor = category_factors[data['product_category'][i]]
        
        # Distance factor (longer distances take more time)
        dist_factor = data['distance_km'][i] / 5000
        
        # Weight factor (heavier items take longer)
        weight_factor = data['product_weight_kg'][i] / 10
        
        # Calculate delivery time with some randomness
        delivery_time[i] = (base_time * loc_factor * cat_factor * (1 + dist_factor * 0.5) * 
                           (1 + weight_factor * 0.2) + np.random.uniform(-1, 1))
        
        # Ensure minimum delivery time based on shipping method
        delivery_time[i] = max(delivery_time[i], method_times[data['shipping_method'][i]] * 0.8)
    
    data['delivery_time_days'] = np.round(delivery_time, 1)
    
    return pd.DataFrame(data)

# Function to train model
def train_model(df):
    X = df[['product_category', 'customer_location', 'shipping_method', 
            'product_weight_kg', 'distance_km', 'order_value']]
    y = df['delivery_time_days']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create preprocessor
    categorical_features = ['product_category', 'customer_location', 'shipping_method']
    numerical_features = ['product_weight_kg', 'distance_km', 'order_value']
    
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )
    
    # Create and train pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    model.fit(X_train, y_train)
    
    # Save the model
    if not os.path.exists('models'):
        os.makedirs('models')
    joblib.dump(model, 'models/delivery_time_model.pkl')
    
    # Calculate and return accuracy metrics
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    return model, train_score, test_score

# Function to make predictions
def predict_delivery_time(model, input_data):
    prediction = model.predict(input_data)
    return prediction[0]

# Main function
def main():
    st.title("📦 Delivery Time Prediction System")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Make Prediction", "Model Information", "Sample Data"])
    
    with tab1:
        st.header("Predict Delivery Time")
        
        col1, col2 = st.columns(2)
        
        with col1:
            product_category = st.selectbox(
                "Product Category",
                options=['Electronics', 'Clothing', 'Books', 'Home & Kitchen', 'Toys']
            )
            
            customer_location = st.selectbox(
                "Customer Location",
                options=['North America', 'Europe', 'Asia', 'South America', 'Australia']
            )
            
            shipping_method = st.selectbox(
                "Shipping Method",
                options=['Standard', 'Express', 'Priority', 'Same Day']
            )
        
        with col2:
            product_weight = st.number_input(
                "Product Weight (kg)",
                min_value=0.1,
                max_value=50.0,
                value=1.0,
                step=0.1
            )
            
            distance = st.number_input(
                "Shipping Distance (km)",
                min_value=10,
                max_value=20000,
                value=1000,
                step=10
            )
            
            order_value = st.number_input(
                "Order Value ($)",
                min_value=1.0,
                max_value=10000.0,
                value=100.0,
                step=10.0
            )
        
        # Create a button to make prediction
        if st.button("Predict Delivery Time"):
            # Check if model exists, if not train it
            model_path = 'models/delivery_time_model.pkl'
            if not os.path.exists(model_path):
                with st.spinner("Training model for the first time..."):
                    df = generate_sample_data(2000)
                    model, train_score, test_score = train_model(df)
            else:
                model = joblib.load(model_path)
            
            # Create input data for prediction
            input_data = pd.DataFrame({
                'product_category': [product_category],
                'customer_location': [customer_location],
                'shipping_method': [shipping_method],
                'product_weight_kg': [product_weight],
                'distance_km': [distance],
                'order_value': [order_value]
            })
            
            # Make prediction
            prediction = predict_delivery_time(model, input_data)
            
            # Display result
            st.success(f"Estimated delivery time: {prediction:.1f} days")
            
            # Calculate and display estimated delivery date
            today = datetime.datetime.now()
            delivery_date = today + datetime.timedelta(days=prediction)
            st.info(f"Expected delivery date: {delivery_date.strftime('%A, %B %d, %Y')}")
            
            # Display shipping details summary
            st.subheader("Order Summary")
            summary_data = {
                "Product Category": product_category,
                "Shipping To": customer_location,
                "Shipping Method": shipping_method,
                "Product Weight": f"{product_weight} kg",
                "Shipping Distance": f"{distance} km",
                "Order Value": f"${order_value:.2f}"
            }
            
            summary_df = pd.DataFrame(list(summary_data.items()), columns=["Detail", "Value"])
            st.table(summary_df)
    
    with tab2:
        st.header("About the Model")
        
        st.write("""
        This application uses a Random Forest Regression model to predict delivery times based on 
        various factors related to the order.
        
        ### Features Used for Prediction
        - **Product Category**: Different products may require different handling times
        - **Customer Location**: Geographic location affects shipping duration
        - **Shipping Method**: The chosen shipping speed (Standard, Express, etc.)
        - **Product Weight**: Heavier items may take longer to process and ship
        - **Distance**: The distance between warehouse and destination
        - **Order Value**: May influence prioritization and handling procedures
        
        ### How It Works
        The model processes these features to estimate how many days delivery will take. 
        For categorical features (like shipping method), the system uses one-hot encoding 
        to convert them into a format suitable for the machine learning algorithm.
        """)
        
        # If model exists, show performance metrics
        model_path = 'models/delivery_time_model.pkl'
        if os.path.exists(model_path):
            # Train model to get metrics
            with st.spinner("Calculating model performance..."):
                df = generate_sample_data(2000)
                model, train_score, test_score = train_model(df)
            
            st.subheader("Model Performance")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Training R² Score", f"{train_score:.3f}")
            
            with col2:
                st.metric("Testing R² Score", f"{test_score:.3f}")
                
            st.caption("R² score measures how well the model predicts the delivery time. A score closer to 1.0 is better.")
        else:
            st.info("Model will be trained when you make your first prediction.")
    
    with tab3:
        st.header("Sample Data")
        st.write("This table shows a sample of the data used to train the model:")
        
        sample_df = generate_sample_data(10)
        st.dataframe(sample_df)
        
        st.caption("Note: The system uses a much larger dataset (2000 samples) to train the actual prediction model.")

if __name__ == "__main__":
    main()
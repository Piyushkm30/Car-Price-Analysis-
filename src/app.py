
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# =========================================================
# AUTOPULSE - USED CAR PRICE ANALYSIS & PREDICTION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_car_data.csv"
MODEL_PATH = BASE_DIR / "models" / "car_price_model.pkl"

st.set_page_config(
    page_title="AutoPulse | Car Intelligence",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# PREMIUM DESIGN
# =========================================================

st.markdown("""
<style>
.stApp {
    background:
        radial-gradient(ellipse at 10% 0%, #142746 0%, transparent 38%),
        radial-gradient(ellipse at 95% 15%, #102A36 0%, transparent 32%),
        #080D18;
    color: #EDF4FF;
    font-family: "Segoe UI", Arial, sans-serif;
}

h1, h2, h3, h4,
.hero-title, .prediction-value {
    font-family: "Segoe UI", Arial, sans-serif !important;
    letter-spacing: -0.45px;
    font-weight: 650 !important;
}

.block-container {
    max-width: 1480px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

[data-testid="stSidebar"] {
    background: #0B1220;
    border-right: 1px solid #243148;
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.5rem;
}

[data-testid="stMetric"] {
    background: linear-gradient(145deg, #15233A, #101827);
    border: 1px solid #293B56;
    padding: 19px 21px;
    border-radius: 16px;
    min-height: 120px;
    box-shadow: 0 8px 28px rgba(0, 0, 0, .12);
}

[data-testid="stMetricLabel"] {
    color: #A8BAD3 !important;
    font-size: 14px !important;
    font-weight: 600 !important;
}

[data-testid="stMetricValue"] {
    color: #F3F7FF !important;
    font-size: 27px !important;
    font-weight: 700 !important;
}

[data-testid="stPlotlyChart"] {
    background: #101827;
    border: 1px solid #243148;
    border-radius: 16px;
    padding: 7px;
}

div[data-testid="stForm"] {
    background: #101827;
    border: 1px solid #293B56;
    padding: 22px;
    border-radius: 18px;
}

label {
    font-size: 14px !important;
    font-weight: 600 !important;
}

.stButton > button,
.stFormSubmitButton > button {
    background: linear-gradient(100deg, #55A7FF, #48E0D0);
    color: #07111F;
    border: none;
    border-radius: 10px;
    min-height: 45px;
    font-weight: 700;
    transition: .2s ease;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    border: none;
    filter: brightness(1.1);
    transform: translateY(-1px);
}

.stDownloadButton > button {
    border-radius: 9px;
    border: 1px solid #426184;
}

[data-testid="stDataFrame"] {
    border: 1px solid #293B56;
    border-radius: 12px;
}

hr {
    border-color: #243148;
}

.eyebrow {
    color: #55A7FF;
    font-size: 12px;
    font-weight: 750;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.hero {
    background: linear-gradient(115deg, #152B4B, #10233A 60%, #12353D);
    border: 1px solid #2A486A;
    border-radius: 22px;
    padding: 30px 32px;
    margin: 8px 0 26px 0;
}

.hero-title {
    font-size: clamp(28px, 3.5vw, 43px);
    line-height: 1.16;
    font-weight: 750;
    color: #F4F8FF;
    margin: 10px 0 12px 0;
}

.hero-copy {
    color: #B6C9E4;
    font-size: 15px;
    line-height: 1.8;
    max-width: 750px;
}

.section-label {
    color: #91A8C8;
    font-size: 12px;
    letter-spacing: 1.8px;
    font-weight: 750;
    text-transform: uppercase;
    margin: 10px 0;
}

.small-note {
    color: #91A2BD;
    font-size: 12px;
}

.prediction-result {
    background: linear-gradient(120deg, #123C46, #132B42);
    border: 1px solid #2A9FAD;
    border-radius: 18px;
    padding: 27px;
    margin-top: 20px;
}

.prediction-value {
    font-size: clamp(30px, 4vw, 45px);
    font-weight: 750;
    color: #64E8D8;
    line-height: 1.3;
    margin-top: 10px;
}

.prediction-rupees {
    color: #F0FAFF;
    font-size: 23px;
    font-weight: 650;
    margin-top: 6px;
}

div[data-testid="stAlert"] {
    border-radius: 12px;
}

footer {
    visibility: hidden;
}

#MainMenu {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD DATA AND MODEL
# =========================================================

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    df = load_data()
    model = load_model()
except Exception as exc:
    st.error(f"Data or model could not be loaded: {exc}")
    st.info(
        "Check data/processed/cleaned_car_data.csv and "
        "models/car_price_model.pkl."
    )
    st.stop()


# =========================================================
# HELPERS
# =========================================================

ACCENT = ["#55A7FF", "#48E0D0", "#A78BFA", "#F5B85B", "#FF8297"]


def style_chart(fig, height=350):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="Segoe UI, Arial, sans-serif",
            color="#DCE7F7",
            size=12,
        ),
        title=dict(
            font=dict(size=16, color="#F0F5FF"),
            x=0.03,
            xanchor="left",
        ),
        margin=dict(l=20, r=20, t=65, b=25),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        hoverlabel=dict(
            bgcolor="#17243A",
            bordercolor="#3A5274",
            font=dict(color="#FFFFFF"),
        ),
    )
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        linecolor="#29384D",
        tickfont=dict(color="#9FB1CC"),
    )
    fig.update_yaxes(
        gridcolor="rgba(117,143,179,0.13)",
        zeroline=False,
        tickfont=dict(color="#9FB1CC"),
    )
    return fig


def indian_rupees(amount):
    """Format a rupee amount using Indian digit grouping."""
    amount = int(round(amount))
    sign = "-" if amount < 0 else ""
    digits = str(abs(amount))

    if len(digits) <= 3:
        return f"{sign}₹{digits}"

    last_three = digits[-3:]
    remaining = digits[:-3]
    groups = []

    while remaining:
        groups.insert(0, remaining[-2:])
        remaining = remaining[:-2]

    return f"{sign}₹{','.join(groups)},{last_three}"


def format_lakh(value):
    return f"₹{value:.2f} lakh"


# =========================================================
# SIDEBAR NAVIGATION
# =========================================================

with st.sidebar:
    st.markdown(
        """
        <div style="font-size:38px">🚘</div>
        <div style="font-size:25px;font-weight:750;
                    letter-spacing:-0.5px">AutoPulse</div>
        <div style="color:#91A2BD;font-size:12px;
                    letter-spacing:2px">CAR INTELLIGENCE</div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Market Analysis",
            "Dataset Explorer",
            "Price Prediction",
        ],
    )

    st.divider()
    st.markdown("**PROJECT STATUS**")
    st.markdown("🟢 Dataset loaded")
    st.markdown("🟢 ML model loaded")
    st.caption(f"{len(df):,} cleaned records")

    st.divider()
    st.caption("Python · Pandas · Plotly · Scikit-learn")


# =========================================================
# PAGE 1: OVERVIEW
# =========================================================

if page == "Overview":

    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">USED CAR MARKET INTELLIGENCE</div>
            <div class="hero-title">
                Understand the market.<br>Predict with confidence.
            </div>
            <div class="hero-copy">
                Explore vehicle pricing patterns, discover market insights,
                and estimate used-car selling prices using Machine Learning.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-label">MARKET AT A GLANCE</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Vehicles", f"{len(df):,}")
    c2.metric("Average Price", f"{df['Selling_Price'].mean():.2f}")
    c3.metric("Median Price", f"{df['Selling_Price'].median():.2f}")
    c4.metric("Average Mileage", f"{df['Kms_Driven'].mean():,.0f} km")

    st.caption(
        "Prices are displayed in the dataset's original unit. "
        "Confirm the source documentation before converting them to INR."
    )

    left, right = st.columns(2)

    with left:
        fig = px.histogram(
            df,
            x="Selling_Price",
            nbins=28,
            title="Selling Price Distribution",
            color_discrete_sequence=["#55A7FF"],
            labels={"Selling_Price": "Selling price"},
        )
        fig.update_traces(
            marker_line_color="#101827",
            marker_line_width=1,
            opacity=0.9,
        )
        st.plotly_chart(style_chart(fig), use_container_width=True)

    with right:
        fuel = (
            df.groupby("Fuel_Type", as_index=False)
            .agg(
                Median_Price=("Selling_Price", "median"),
                Vehicles=("Selling_Price", "size"),
            )
        )

        fig = px.bar(
            fuel,
            x="Fuel_Type",
            y="Median_Price",
            color="Fuel_Type",
            title="Median Price by Fuel Type",
            color_discrete_sequence=ACCENT,
            hover_data=["Vehicles"],
            labels={
                "Fuel_Type": "Fuel type",
                "Median_Price": "Median selling price",
                "Vehicles": "Vehicle count",
            },
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(style_chart(fig), use_container_width=True)

    left, right = st.columns(2)

    with left:
        seller = (
            df.groupby("Seller_Type", as_index=False)
            .agg(Median_Price=("Selling_Price", "median"))
        )

        fig = px.bar(
            seller,
            x="Seller_Type",
            y="Median_Price",
            color="Seller_Type",
            title="Median Price by Seller",
            color_discrete_sequence=["#48E0D0", "#A78BFA"],
            labels={
                "Seller_Type": "Seller type",
                "Median_Price": "Median selling price",
            },
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(style_chart(fig), use_container_width=True)

    with right:
        transmission = (
            df.groupby("Transmission", as_index=False)
            .agg(Median_Price=("Selling_Price", "median"))
        )

        fig = px.bar(
            transmission,
            x="Transmission",
            y="Median_Price",
            color="Transmission",
            title="Median Price by Transmission",
            color_discrete_sequence=["#55A7FF", "#F5B85B"],
            labels={
                "Transmission": "Transmission",
                "Median_Price": "Median selling price",
            },
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(style_chart(fig), use_container_width=True)

    st.markdown(
        '<div class="section-label">FEATURED VEHICLES</div>',
        unsafe_allow_html=True,
    )

    top_cars = (
        df.groupby("Car_Name", as_index=False)
        .agg(
            Median_Price=("Selling_Price", "median"),
            Listings=("Selling_Price", "size"),
        )
        .sort_values("Median_Price", ascending=False)
        .head(10)
    )

    st.dataframe(
        top_cars,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Car_Name": st.column_config.TextColumn("Vehicle"),
            "Median_Price": st.column_config.NumberColumn(
                "Median price (dataset unit)", format="%.2f"
            ),
            "Listings": st.column_config.NumberColumn("Records"),
        },
    )


# =========================================================
# PAGE 2: MARKET ANALYSIS
# =========================================================

elif page == "Market Analysis":

    st.markdown(
        '<div class="eyebrow">DATA EXPLORATION / DEEP DIVE</div>',
        unsafe_allow_html=True,
    )
    st.title("Market Analysis")
    st.write(
        "Explore relationships between vehicle features and selling prices."
    )

    left, right = st.columns(2)

    with left:
        fig = px.scatter(
            df,
            x="Present_Price",
            y="Selling_Price",
            color="Fuel_Type",
            hover_name="Car_Name",
            hover_data=["Year", "Transmission", "Seller_Type"],
            title="Present Price vs Selling Price",
            color_discrete_sequence=ACCENT,
            labels={
                "Present_Price": "Present price",
                "Selling_Price": "Selling price",
            },
        )
        st.plotly_chart(style_chart(fig), use_container_width=True)

    with right:
        fig = px.scatter(
            df,
            x="Kms_Driven",
            y="Selling_Price",
            color="Transmission",
            hover_name="Car_Name",
            hover_data=["Year", "Fuel_Type"],
            title="Mileage vs Selling Price",
            color_discrete_sequence=["#55A7FF", "#48E0D0"],
            labels={
                "Kms_Driven": "Kilometres driven",
                "Selling_Price": "Selling price",
            },
        )
        st.plotly_chart(style_chart(fig), use_container_width=True)

    st.subheader("Numerical Feature Correlations")

    numeric = df.select_dtypes(include="number")
    corr = numeric.corr()

    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale=[
                [0, "#E8798B"],
                [0.5, "#17243A"],
                [1, "#48E0D0"],
            ],
            zmin=-1,
            zmax=1,
            text=corr.round(2).values,
            texttemplate="%{text}",
            hovertemplate=(
                "%{y} vs %{x}<br>"
                "Correlation: %{z:.2f}<extra></extra>"
            ),
            colorbar=dict(title="Correlation"),
        )
    )

    fig.update_layout(title="Feature Correlation Matrix")
    st.plotly_chart(style_chart(fig, height=480), use_container_width=True)

    st.caption(
        "Correlation indicates association, not causation. Interpret "
        "categories with very few records cautiously."
    )


# =========================================================
# PAGE 3: DATASET EXPLORER
# =========================================================

elif page == "Dataset Explorer":

    st.markdown(
        '<div class="eyebrow">DATA WORKSPACE</div>',
        unsafe_allow_html=True,
    )
    st.title("Dataset Explorer")
    st.write("Search, filter, inspect and export the cleaned dataset.")

    c1, c2, c3 = st.columns(3)

    with c1:
        fuel_choice = st.multiselect(
            "Fuel type",
            sorted(df["Fuel_Type"].dropna().unique()),
            default=sorted(df["Fuel_Type"].dropna().unique()),
        )

    with c2:
        seller_choice = st.multiselect(
            "Seller type",
            sorted(df["Seller_Type"].dropna().unique()),
            default=sorted(df["Seller_Type"].dropna().unique()),
        )

    with c3:
        transmission_choice = st.multiselect(
            "Transmission",
            sorted(df["Transmission"].dropna().unique()),
            default=sorted(df["Transmission"].dropna().unique()),
        )

    search_text = st.text_input(
        "Search vehicle",
        placeholder="Enter a car name, e.g. city, verna or i20",
    )

    filtered = df[
        df["Fuel_Type"].isin(fuel_choice)
        & df["Seller_Type"].isin(seller_choice)
        & df["Transmission"].isin(transmission_choice)
    ].copy()

    if search_text.strip():
        filtered = filtered[
            filtered["Car_Name"].str.contains(
                search_text.strip(),
                case=False,
                na=False,
            )
        ]

    st.metric("Matching records", f"{len(filtered):,}")

    st.dataframe(
        filtered,
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "⬇️ Download Filtered CSV",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="filtered_used_car_data.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.subheader("Data Quality Snapshot")

    q1, q2, q3 = st.columns(3)
    q1.metric("Missing Values", f"{int(filtered.isna().sum().sum()):,}")
    q2.metric("Duplicate Rows", f"{int(filtered.duplicated().sum()):,}")
    q3.metric("Columns", f"{filtered.shape[1]}")


# =========================================================
# PAGE 4: PRICE PREDICTION
# =========================================================

elif page == "Price Prediction":

    st.markdown(
        '<div class="eyebrow">AUTOPULSE / PRICE ESTIMATOR</div>',
        unsafe_allow_html=True,
    )

    st.title("Used Car Price Estimator")
    st.write(
        "Car ki details select karo aur trained Machine Learning model "
        "se estimated resale value dekho."
    )

    st.info(
        "Currency conversion below assumes the dataset price unit is "
        "₹ lakh. Verify the original dataset documentation before "
        "using this conversion for real valuations."
    )

    form_col, info_col = st.columns([1.1, 0.9], gap="large")

    with form_col:
        st.markdown("### 🚘 Vehicle Configuration")

        with st.form("prediction_form"):

            st.markdown("**01 — Car details**")

            available_years = sorted(
                df["Year"].dropna().astype(int).unique().tolist(),
                reverse=True,
            )

            year = st.selectbox(
                "Manufacturing year",
                available_years,
                index=(
                    available_years.index(2018)
                    if 2018 in available_years
                    else 0
                ),
            )

            fuel_type = st.selectbox(
                "Fuel type",
                sorted(df["Fuel_Type"].dropna().unique().tolist()),
            )

            transmission = st.selectbox(
                "Transmission",
                sorted(df["Transmission"].dropna().unique().tolist()),
            )

            st.markdown("**02 — Price and usage**")

            present_price = st.number_input(
                "Reference price (₹ lakh)",
                min_value=0.01,
                max_value=200.0,
                value=8.50,
                step=0.25,
                format="%.2f",
                help="Enter Present_Price in the dataset's unit.",
            )

            mileage_presets = {
                "Under 10,000 km": 5000,
                "10,000–30,000 km": 20000,
                "30,000–50,000 km": 40000,
                "50,000–80,000 km": 65000,
                "80,000–1,00,000 km": 90000,
                "Over 1,00,000 km": 120000,
                "Enter exact kilometres below": None,
            }

            mileage_choice = st.selectbox(
                "Kilometres driven",
                list(mileage_presets.keys()),
                index=2,
            )

            if mileage_presets[mileage_choice] is None:
                kms_driven = st.number_input(
                    "Exact kilometres",
                    min_value=0,
                    max_value=1000000,
                    value=40000,
                    step=1000,
                )
            else:
                kms_driven = mileage_presets[mileage_choice]
                st.caption(f"Selected mileage: {kms_driven:,} km")

            owner = st.selectbox(
                "Previous owners",
                [0, 1, 2, 3],
                format_func=lambda x: (
                    "First owner / no previous owners"
                    if x == 0
                    else f"{x} previous owner(s)"
                ),
            )

            seller_type = st.selectbox(
                "Seller category",
                sorted(df["Seller_Type"].dropna().unique().tolist()),
            )

            submitted = st.form_submit_button(
                "✨ Calculate Estimated Price",
                use_container_width=True,
            )

    with info_col:
        st.markdown(
            """
            <div class="hero" style="margin-top:0">
                <div class="eyebrow">YOUR VALUATION, SIMPLIFIED</div>
                <h3>Make a smarter car decision.</h3>
                <p class="hero-copy">
                    The model considers manufacturing year, reference price,
                    mileage, fuel type, transmission, seller type and
                    previous ownership.
                </p>
                <hr>
                <p>🔹 Select your vehicle details</p>
                <p>🔹 Run the trained ML model</p>
                <p>🔹 Review the estimated resale price</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### Model Performance")

        metric1, metric2 = st.columns(2)
        metric1.metric("R² Score", "0.765")
        metric2.metric("Test MAE", "1.122")

        st.caption(
            "Metrics come from the current test split. MAE uses the "
            "dataset's original price unit."
        )

        st.markdown("### Currency Guide")
        st.markdown(
            """
            - **1 lakh = ₹1,00,000**
            - **10 lakh = ₹10,00,000**
            - The prediction is an estimate, not a guaranteed sale price.
            """
        )

    if submitted:

        input_data = pd.DataFrame([{
            "Year": year,
            "Present_Price": present_price,
            "Kms_Driven": kms_driven,
            "Fuel_Type": fuel_type,
            "Seller_Type": seller_type,
            "Transmission": transmission,
            "Owner": owner,
        }])

        try:
            prediction = float(model.predict(input_data)[0])

            if prediction < 0:
                st.warning(
                    "The model returned a negative estimate. "
                    "That is not a meaningful resale price."
                )
            else:
                # Conversion assumes model output is in lakh rupees.
                predicted_lakh = prediction
                predicted_rupees = predicted_lakh * 100000

                st.markdown(
                    f"""
                    <div class="prediction-result">
                        <div class="eyebrow">ESTIMATED RESALE VALUE</div>
                        <div class="prediction-value">
                            {format_lakh(predicted_lakh)}
                        </div>
                        <div class="prediction-rupees">
                            {indian_rupees(predicted_rupees)}
                        </div>
                        <div style="color:#91B9C9;margin-top:10px">
                            Machine-learning estimate · INR
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.caption(
                    "This estimate assumes the source dataset uses ₹ lakh. "
                    "Actual resale value varies with condition, location, "
                    "service history and market demand."
                )

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center;color:#8295B1;
                font-size:12px;padding:12px">
        <strong>AUTOPULSE</strong> · USED CAR INTELLIGENCE
        <br>Built with Python · Pandas · Plotly · Scikit-learn · Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)

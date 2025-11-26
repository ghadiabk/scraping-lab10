import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Page configuration
st.set_page_config(
 page_title="Amazon Products Dashboard",
 layout="wide",
)
st.title("Amazon Products Dashboard")
st.write("Interactive EDA and an NLP-based search over Amazon product titles using the cleaned dataset."
)

@st.cache_data
def load_data(csv_path: str ="amazon_cleaned_data.csv") -> pd.DataFrame:
    df = pd.read_csv(csv_path)
     # Ensure expected columns exist
    expected_cols = [
     "Title",
     "Price",
     "Rating",
     "Reviews",
     "Rating_Num",
     "Reviews_Int",
     "Rating_Category"]
    
    missing = [c for c in expected_cols if c not in
    df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in CSV: {missing}")
    return df

df = load_data()


st.sidebar.header("Filters")
# 1) Keyword in Title (applied last, but declared first for UX)
keyword = st.sidebar.text_input("Keyword in Title").strip().lower()

# 2) Price range slider
price_min = float(df["Price"].min())
price_max = float(df["Price"].max())
price_range = st.sidebar.slider(
 "Price Range ($)",
 min_value=price_min,
 max_value=price_max,
 value=(price_min, price_max),
)

# 3) Minimum rating slider
min_rating = st.sidebar.slider(
 "Minimum Rating",
 min_value=0.0,
 max_value=5.0,
 value=4.0,
 step=0.1,
)

# 4) Rating category selector
rating_categories = ["All"] + sorted(df["Rating_Category"].dropna().unique().tolist())
selected_cat = st.sidebar.selectbox(
 "Rating Category",
 options=rating_categories,
 index=0,
)


base_count = len(df)

# Step A: Price + Rating filters
price_rating_df = df[
 (df["Price"] >= price_range[0])
 & (df["Price"] <= price_range[1])
 & (df["Rating_Num"] >= min_rating)
]
price_rating_count = len(price_rating_df)

# Step B: Category filter
if selected_cat != "All":
 cat_df = price_rating_df[price_rating_df["Rating_Category"] == selected_cat]
else:
 cat_df = price_rating_df
cat_count = len(cat_df)

# Step C: Keyword filter
if keyword:
 filtered_df = cat_df[

cat_df["Title"].str.lower().str.contains(keyword,na=False)
 ]
else:
 filtered_df = cat_df
final_count = len(filtered_df)
st.sidebar.markdown("---")
st.sidebar.write(f"Total rows: {base_count}")
st.sidebar.write(f"After price & rating: {price_rating_count}")
st.sidebar.write(f"After category: {cat_count}")
st.sidebar.write(f"After keyword: {final_count}")
if final_count == 0:
 st.warning("No rows match the current filters. Relax the filters and try again.")

st.header("Exploratory Data Analysis (Filtered Subset)")
col_a, col_b, col_c = st.columns(3)
with col_a:
 avg_price = filtered_df["Price"].mean() if final_count > 0 else np.nan
 st.metric(
 "Average Price (Filtered)",
 f"${avg_price:.2f}" if final_count > 0 else
"N/A",
 )
with col_b:
 avg_rating = filtered_df["Rating_Num"].mean() if final_count > 0 else np.nan
 st.metric(
 "Average Rating (Filtered)",
 f"{avg_rating:.2f}" if final_count > 0 else
"N/A",
 )
with col_c:
 total_reviews = (
 int(filtered_df["Reviews_Int"].sum()) if final_count > 0 else 0
 )
 st.metric(
 "Total Reviews (Filtered)", f"{total_reviews:,}" if final_count > 0 else"N/A",)


col1, col2 = st.columns(2)
with col1:
 st.subheader("Price Distribution (Filtered)")
 if final_count > 0:
  bins = np.linspace(price_min, price_max, 20)
  price_hist, bin_edges = np.histogram(filtered_df["Price"], bins=bins)
  price_hist_df = pd.DataFrame(
 {
 "Price_Bin": (bin_edges[:-1] +
bin_edges[1:]) / 2,
 "Count": price_hist,
 }
 ).set_index("Price_Bin")
  st.bar_chart(price_hist_df)
 else:
  st.info("No data after filtering.")
with col2:
 st.subheader("Count by Rating Category (Filtered)")
 if final_count > 0:
  cat_counts = (
  filtered_df["Rating_Category"]
  .value_counts()
  .rename("Count")
  .to_frame()
 )
  st.bar_chart(cat_counts)
 else:
  st.info("No data after filtering.")


with st.container():
 st.subheader("Summary Statistics (Filtered)")
 if final_count > 0:
  st.write(
  filtered_df[["Price", "Rating_Num", "Reviews_Int"]].describe().T
 )
 else:
  st.info("No data to summarize.")


with st.expander("Show Filtered Dataset"):
 st.dataframe(filtered_df.reset_index(drop=True))


@st.cache_data
def build_tfidf_for_subset(titles: pd.Series):
 """Build TF–IDF model for a given subset of titles."""
 vec = TfidfVectorizer(stop_words="english")
 X_sub = vec.fit_transform(titles.astype(str).tolist())
 return vec, X_sub


st.header("NLP Search on Product Titles (Within Current Filters)")
st.write(
 "Use a free-text query to search for products within the **currently "
 "filtered subset**. TF–IDF and cosine similarity are used to rank titles."
)
search_col1, search_col2 = st.columns([3, 1])
with search_col1:
 user_query = st.text_input(
 "Describe what you are looking for "
 "(e.g., 'wireless noise cancelling headphones with mic')"
 )
with search_col2:
 top_k = st.number_input(
 "Top K results",
 min_value=1,
 max_value=20,
 value=5,
 step=1,
 )
run_search = st.button("Find Matches")


if run_search:
 if not user_query:
  st.warning("Please enter a search query before running the NLP search.")
 elif final_count == 0:
  st.warning("No data in the current filtered subset to search within.")
 else:
 # Build TF–IDF on the CURRENT filtered subset
  subset_vec, X_sub = build_tfidf_for_subset(filtered_df["Title"])
  query_vec = subset_vec.transform([user_query])
  sims = cosine_similarity(query_vec, X_sub).flatten()
  k = min(int(top_k), len(filtered_df))
 if k <= 0:
  st.warning("No products available after filtering.")
 else:
  top_idx_local = np.argsort(sims)[-k:][::-1]
  results = filtered_df.iloc[top_idx_local][
  ["Title", "Price", "Rating_Num", "Reviews_Int", "Rating_Category"]
 ].reset_index(drop=True)
 if len(results) == 0:
  st.warning("No matches found in the current filtered subset.")
 else:
  st.subheader(
  f"Top {len(results)} Matching Products (within current filters)"
 )
 st.dataframe(results)



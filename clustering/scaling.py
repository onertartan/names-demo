from sklearn.preprocessing import normalize
import pandas as pd
# Overridden method
import streamlit as st

from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd
import streamlit as st


def scale(scaler_method, df, total_counts):
    """    Row-wise scaling of feature vectors (provinces or names).
    All methods return a DataFrame with the same index/columns. """
    if "L1" in scaler_method:
        # Relative distribution of selected features
        df_scaled = normalize(df, axis=1, norm="l1")

    elif "L2" in scaler_method:
        # Unit-length vectors (angular similarity)
        df_scaled = normalize(df, axis=1, norm="l2")

    elif "Share of Total" in scaler_method:
        # Share of all births (not just top-n)
        total_counts_unique = total_counts.groupby(level=0).first()
        df_scaled = df.div(total_counts_unique, axis=0)

    elif "TF-IDF" in scaler_method:
        tfidf = TfidfTransformer(
            norm="l2",
            use_idf=True,
            smooth_idf=True
        )
        df_scaled = tfidf.fit_transform(df.values).toarray()

    else:
        st.warning("No scaling selected. Using raw counts.")
        df_scaled = df.values
    return pd.DataFrame(
        df_scaled,
        index=df.index,
        columns=df.columns
    )

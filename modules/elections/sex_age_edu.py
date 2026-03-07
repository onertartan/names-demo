from modules.base_page_common import PageCommon
import pandas as pd
import streamlit as st
from utils.checkbox_group import Checkbox_Group


class ElectionSexAgeEdu(PageCommon):
    page_name = "sex_age_edu_elections"
    features = {"nominator":  ["sex", "education", "age"], "denominator": ["sex", "education", "age"]}
    checkbox_group = {"age": Checkbox_Group(page_name, "age", 2,
                      ["all"] + ["18-24"] + [f"{i}-{i + 4}" for i in range(25, 75, 5)] + ["75+"]),
                      "education": Checkbox_Group(page_name,"education",2,
            ["all",'illiterate', 'literate but did not complete any school', 'primary school', 'elementary school', 'secondary school or equivalent',
             'high school or equivalent', 'pre-license or bachelor degree', 'master degree', 'phd', 'unknown']),
                      "sex": Checkbox_Group(page_name, "sex", 1, ["all", "male", "female"])}

    top_row_cols = st.columns([1])  # first col is for geo_scale, others are optional(extras)
    col_weights = [1, 4, 2, .1,1, 4, 2,]


    @staticmethod
    @st.cache_data
    def get_data():
        df = {}
        df["district"]= pd.read_csv("data/preprocessed/elections/df_edu.csv", index_col=[0, 1, 2], header=[0, 1, 2])
        df["province"]  = pd.read_csv("data/preprocessed/elections/df_edu.csv", index_col=[0, 1, 2], header=[0, 1, 2]).droplevel(2).groupby(["year", "province"]).sum()

        df_data = {"denominator": df, "nominator": df}
        return df_data


ElectionSexAgeEdu().run()


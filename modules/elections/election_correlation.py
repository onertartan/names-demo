from matplotlib import pyplot as plt
import numpy as np
from modules.base_page_common import PageCommon
from modules.base_page import BasePage
from utils.checkbox_group import Checkbox_Group
import  streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.colors as mcolors
from viz.gui_helpers.ui_base_page_common import gui_basic_setup


class ElectionCorrelation(PageCommon):
    page_name = "election_correlation"
    features = {"nominator": ["sex", "education", "age"],"denominator":["Party/Alliance"]}

    checkbox_group = {"age": Checkbox_Group(page_name, "age", 2,  ["all"] + ["18-24"] + [f"{i}-{i + 4}" for i in range(25, 75, 5)] + ["75+"]),
                      "sex": Checkbox_Group(page_name, "sex", 1, ["all","male", "female"]),
                      "education": Checkbox_Group(page_name, "education", 2,
                                                  ["all"]+list(pd.read_csv("data/preprocessed/elections/df_edu.csv", index_col=[0, 1, 2], header=[0, 1, 2]).columns.get_level_values(1).unique() ))}
           # ["all",'illiterate', 'literate but did not complete any school', 'primary school', 'elementary school', 'secondary school or equivalent',
          #   'high school or equivalent', 'pre-license or bachelor degree', 'master degree', 'phd', 'unknown'])    }
    col_weights = [1, 4, 2, .1, 6.8, .1, .1]


    @staticmethod  # Add this
    @st.cache_data
    def get_data(geo_scale=None):
        df_sex_age_edu = pd.read_csv("data/preprocessed/elections/df_edu.csv", index_col=[0, 1, 2], header=[0, 1, 2])
        df_election = pd.read_csv("data/preprocessed/elections/df_election.csv", index_col=[0, 1, 2])

        #basic_keys = df_election.loc[2018].dropna(axis=1).columns[5:].tolist()
       # cls.checkbox_group["Party/Alliance"] = Checkbox_Group(cls.page_name,"Party/Alliance",4,basic_keys)
       # df_data = {"denominator": df_edu, "nominator": df_election}
        return df_sex_age_edu, df_election


    def sidebar_controls_basic_setup(self,*args):
        with (st.sidebar):
            st.header('Select options')
            if "selected_election_year" not in st.session_state:
                st.session_state["selected_election_year"] =2018
            year = st.radio("Select year", options=[2018,2023], key="election", index=0)
        return year


    def render(self):
        df_sex_age_edu, df_election = self.get_data()
        st.radio("Options for selection of primary parameters for correlation calculation",
                 ["Aggregate", "Education", "Age"],
                 key="correlation_selection")
        cols_nom_denom = gui_basic_setup(self.col_weights)
        year = self.sidebar_controls_basic_setup()

        basic_keys = ["all"]+df_election.loc[year].dropna(axis=1).columns[5:].tolist() # Parties start from index 5
        self.checkbox_group["Party/Alliance"] = Checkbox_Group(self.page_name, "Party/Alliance", 4, basic_keys)
        selected_features_dict = self.get_selected_features(cols_nom_denom)
        print("AALLL:",selected_features_dict)
        # add checkbox_group with the "Party/Alliance" according to the selected year --> SELECT PARTIES  NOT NAN AT THAT YEAR. hint:  use st.session_state["year_1"],
        #Combine features
        if(st.session_state["correlation_selection"]=="Aggregate"):
            df_primary = df_sex_age_edu.loc[year, selected_features_dict["nominator"]].sum(axis=1 ).to_frame("aggregated selected params")
            selected_features_dict["nominator"] = slice(None)
        # Group by education
        elif(st.session_state["correlation_selection"]) == "Education":
            df_primary = df_sex_age_edu.loc[year, selected_features_dict["nominator"]].groupby(level=1, axis=1).sum()
            selected_features_dict["nominator"] = selected_features_dict["nominator"][1]  # get selected education options only from  (slice(None), ['master',"phd"],slice(None))

        #group by age
        else:
            df_primary = df_sex_age_edu.loc[year, selected_features_dict["nominator"]].groupby(level=2, axis=1).sum( )
            selected_features_dict["nominator"] = selected_features_dict["nominator"][2]

        print("ELEC2023:",df_election.loc[year],"cols:",df_election.loc[year].columns)
        print("SDF:", (selected_features_dict))
        print("len(selected_features_dict[denominator]",len(selected_features_dict["denominator"]))
        selected_parties = selected_features_dict["denominator"][0]
        #if selected_parties==slice(None):
         #   selected_parties =df_election.loc[year].dropna(axis=1).columns[2:].tolist() #select all parties + Number of voters who voted, valid votes,invalid votes
          #  print("CHECK111",selected_parties)
        #else:
          #  print("CHECK2")
        selected_parties = ["Number of voters who voted", "valid votes", "invalid votes"]+ selected_parties
        df_secondary = df_election.loc[year,selected_parties ]#There is on feature group for secondary parameters(tuple of 1 length)

        print("PRIM:",df_primary)
        print("SECOND:",df_secondary)
        df_merged =pd.merge(df_secondary, df_primary, left_index=True, right_index=True)
        print("MERGED:",df_merged)
        # divide by "Number of voters who voted and drop the column
        df_result = df_merged.dropna(axis=1).div(df_merged.loc[:,"Number of voters who voted"] , axis=0).drop(columns=["Number of voters who voted"])
        # drop the column "Number of voters from" the list selected_parties
        selected_parties.remove("Number of voters who voted")
       # st.dataframe(df_result)
        # Find the correlation
        correlations = df_result.corr()
        print("FFEETT:",selected_features_dict["nominator"],"Partiler",selected_parties)
        print("corr",correlations,"SELEF:",correlations.loc[selected_features_dict["nominator"],selected_parties])
        st.dataframe(correlations)

        fig, ax = plt.subplots(figsize=(5,5))
        ax.set_title('Correlation Heatmap', fontsize=20)  # title with fontsize 20

        # Optional: Adjust layout to prevent label cutoff
        # sns.heatmap(correlations, ax=ax, cmap=cmap, norm=norm, annot=False, cbar_kws={"ticks": [-1, -0.38, 0.38, 1]})


        sns.heatmap(correlations.loc[selected_features_dict["nominator"], selected_parties], ax=ax, annot=True)
        sns.set(font_scale=.5)
        # Customize font size for x and y axis labels
        ax.set_xlabel("Party/alliance & valid/invalid votes", fontsize=14)
        y_label = "Aggregated parameters" if(st.session_state["correlation_selection"]=="Aggregate") else st.session_state["correlation_selection"]
        ax.set_ylabel(y_label, fontsize=14)
        # Customize x-axis labels
        x_labels = ax.get_xticklabels()
        y_labels = ax.get_yticklabels()
        ax.set_xticklabels(x_labels,    fontsize=9, fontweight='bold', fontfamily='sans-serif')
        # Customize y-axis labels
        ax.set_yticklabels(y_labels, rotation=0, fontsize=9, fontweight='bold', fontfamily='sans-serif')
        # cls.custom_correlation_heatmap(correlations.to_numpy(), ax=ax, vmin=-0.5, vmax=0.5)
        cols_hm = st.columns([.1, 9.8, .1], gap="small")  #
        cols_hm[1].pyplot(fig)


ElectionCorrelation().run()


def custom_correlation_heatmap(correlations, ax=None, vmin=-0.38, vmax=0.38):
    """
    Create a heatmap with custom coloring where values outside vmin/vmax range are white

    Parameters:
    correlations: correlation matrix
    ax: matplotlib axis
    vmin: minimum value for color scaling
    vmax: maximum value for color scaling
    """
    if ax is None:
        _, ax = plt.subplots()

    # Create mask for values outside the range
    mask = (correlations > vmax)

    # Get the default colormap
    default_cmap = plt.get_cmap('RdBu_r')

    # Create custom colormap with white for out-of-range values
    n_colors = 256
    colors = default_cmap(np.linspace(0, 1, n_colors))
    custom_cmap = mcolors.ListedColormap(colors)

    # Plot the heatmap
    sns.heatmap(correlations,
                ax=ax,
                vmin=vmin,
                vmax=vmax,
                cmap=custom_cmap,
                annot=False)

    # Color the masked values white
    for i in range(correlations.shape[0]):
        for j in range(correlations.shape[1]):
            if mask[i, j]:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=True, color='green'))

    return ax

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import matplotlib.pyplot as plt
from streamlit_shap import st_shap
import mlflow
import shap
import streamlit.components.v1 as components
from matplotlib.image import imread
from sklearn.neighbors import NearestNeighbors


###########################################################
################  Chargement de données   #################
###########################################################

# Titre de la page
st.set_page_config(
    page_title="Implémentez un modèle de scoring",
    page_icon="icon_pret_a_depenser.png",
    layout="wide",
)


components.html(
    """<body style="margin: 0">
                        <h1 style="margin: 0;font-family: Source Sans Pro, sans-serif;
                        font-weight: 700; color: #e41d35;font-size: 2.45rem;text-align: center;">
                        <span>Projet 7 : Implémentez un modèle de scoring</span>
                        </h1>
                    <body>""",
    width=None,
    height=50,
    scrolling=False,
)

# Chargement du modèle avec mlflow
model = mlflow.sklearn.load_model("mlflow_model")
# Récupération du classificateur à partir du pipeline
lgbm = model.named_steps["classifier"]
# Initialisation de l'explainer
explainer = shap.TreeExplainer(lgbm)


data_train = pd.read_csv("donnees_train_essai.csv")
data_train = data_train.reset_index(drop=True)
data_test = pd.read_csv("donnees_test_essai.csv")
data_test_rm = data_test.drop(columns=["SK_ID_CURR"], axis=1)
data_train_rm = data_train.drop(columns=["TARGET"], axis=1)
shap_values = explainer.shap_values(data_test_rm)[1]
exp_value = explainer.expected_value[1]
logo = imread("home_credit_logo.png")

# define constants
APPROVED_COLOR = "#31b002"
REJECTED_COLOR = "#c43145"
THRESHOLD = 0.51

###########################################################
####################   Info Client   ######################
###########################################################


def tab_client(df):
    st.markdown(
        '<p style="background-color:#e41d35;color:#ffffff;font-size:24px;border-radius:2%;">Tableau clientèle</p>',
        unsafe_allow_html=True,
    )

    # Convert "YEARS_BIRTH" and "CNT_CHILDREN" to string for filtering
    df["YEARS_BIRTH"] = (round(abs(df["DAYS_BIRTH"]), 0).astype(int)).astype(str)
    df["CNT_CHILDREN"] = df["CNT_CHILDREN"].astype(str)
    df["SK_ID_CURR"] = df["SK_ID_CURR"].astype(str)

    # Convert "CODE_GENDER" to string for filtering
    df["CODE_GENDER"] = df["CODE_GENDER"].apply(
        lambda x: "Male" if x == 0 else "Female"
    )

    # Define filters
    sex = st.selectbox("Sexe", ["All"] + df["CODE_GENDER"].unique().tolist())
    age = st.selectbox("Age", ["All"] + sorted(df["YEARS_BIRTH"].unique().tolist()))
    child = st.selectbox(
        "Enfants", ["All"] + sorted(df["CNT_CHILDREN"].unique().tolist())
    )

    # Filter DataFrame
    if sex != "All":
        df = df[df["CODE_GENDER"] == sex]
    if age != "All":
        df = df[df["YEARS_BIRTH"] == age]
    if child != "All":
        df = df[df["CNT_CHILDREN"] == child]

    # select only the columns we need
    df = df[
        [
            "SK_ID_CURR",
            "CODE_GENDER",
            "YEARS_BIRTH",
            "CNT_CHILDREN",
            "AMT_INCOME_TOTAL",
            "AMT_CREDIT",
            "AMT_ANNUITY",
            "AMT_GOODS_PRICE",
            "DAYS_EMPLOYED",
        ]
    ]

    # Display DataFrame
    st.markdown(
        f'<h2 style="color:#6ad46a;font-size:14px;">Total clients correspondants: {len(df)}</h2>',
        unsafe_allow_html=True,
    )
    st.dataframe(df)


def get_client():
    """Fetch data from FastAPI and select a client via a selectbox"""
    response = requests.get("http://localhost:8501/credit")
    #response = requests.get(f"fqw7wvgywmudbpkldjueby.streamlit.app")
    #response = requests.get("https://medlionp7.azurewebsites.net/credit")
    json_output = response.json()
    liste_id = json_output["liste_id"]
    client = st.selectbox("**Client**", liste_id)
    # Find the index of the selected client in the list of IDs
    idx_client = liste_id.index(client)
    return client, idx_client


def infos_client(client, col):
    """Fetch data from FastAPI and display the info of the selected client in the sidebar"""
    response = requests.get(f"http://localhost:8501/credit/{client}/data")
    #response = requests.get(f"fqw7wvgywmudbpkldjueby.streamlit.app")
    #response = requests.get(f"https://medlionp7.azurewebsites.net/credit/{client}/data")
    infos_clt = response.json()["data"][0]
    return infos_clt[col]


def get_color(result):
    """Define color based on prediction result"""
    return APPROVED_COLOR if result == "Approved" else REJECTED_COLOR


###########################################################
#######################  Prediction  ######################
###########################################################
def get_proba_for_client(client: int):
    #azure_api_url = f"https://medlionp7.azurewebsites.net/credit/{client}/predict"
    # response = requests.get(azure_api_url)
    local_api_url = f"http://localhost:8501/credit/{client}/predict"
    #local_api_url=f"fqw7wvgywmudbpkldjueby.streamlit.app"
    response = requests.get(local_api_url)
    response.raise_for_status()
    proba_dict = response.json()
    proba = proba_dict.get("probability", 0)
    return proba


def get_prediction(client):
    """Get prediction for a client"""
    proba = get_proba_for_client(client)
    if proba is None:
        return None, "Prediction not available"

    score = np.round(proba / 100, 2)
    result = "Approved" if score > THRESHOLD else "Rejected"
    return score, result


def score_viz(data, client, idx_client):
    """Main function for 'Score visualization' tab"""
    score, result = get_prediction(client)
    if score is None:
        st.write(result)
        return

    color = get_color(result)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=score,
            number={"font": {"size": 38}},
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": result, "font": {"size": 28, "color": color}},
            delta={
                "reference": THRESHOLD,
                "increasing": {"color": APPROVED_COLOR},
                "decreasing": {"color": REJECTED_COLOR},
            },
            gauge={
                "axis": {"range": [0, 1], "tickcolor": color},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, THRESHOLD], "color": APPROVED_COLOR},
                    {"range": [THRESHOLD, 1], "color": REJECTED_COLOR},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 5},
                    "thickness": 1,
                    "value": THRESHOLD,
                },
            },
        )
    )

    st.plotly_chart(fig)

    st_shap(
        shap.force_plot(
            explainer.expected_value[1],
            shap_values[idx_client, :],
            data.iloc[idx_client, :],
        ),
        height=200,
        width=1000,
    )

    shap_values_expl = shap.Explanation(
        values=shap_values[idx_client],
        base_values=explainer.expected_value[1],
        data=data.iloc[idx_client].values,
        feature_names=data.columns.tolist(),
    )

    st_shap(shap.waterfall_plot(shap_values_expl))


def plot_income_distribution(data_train, client):
    """Plot the distribution of client income."""
    # Filter out outliers
    df_revenus = data_train[data_train["AMT_INCOME_TOTAL"] < 500000]

    # Create the plot
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.hist(df_revenus["AMT_INCOME_TOTAL"], bins=20, edgecolor="k")
    ax.axvline(infos_client(client, "AMT_INCOME_TOTAL"), color="red", linestyle=":")
    ax.set_title("Client Income Distribution")
    ax.set_xlabel("Income ($ USD)")
    ax.set_ylabel("Count")

    st.pyplot(fig)


def plot_age_distribution(data_train, client):
    """Plot the distribution of client age."""
    # Convert days to years for easier interpretation
    data_train["YEARS_BIRTH"] = round(abs(data_train["DAYS_BIRTH"]), 0).astype(int)

    # Create the plot
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.hist(data_train["YEARS_BIRTH"], bins=20, edgecolor="k")
    ax.axvline(
        round(abs(infos_client(client, "DAYS_BIRTH")), 0), color="red", linestyle="--"
    )
    ax.set_title("Client Age Distribution")
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Count")

    st.pyplot(fig)


###########################################################
#######################  Comparaison  #####################
###########################################################


def load_neighbors(data_test, idx_client):
    data_client = data_test.copy().loc[idx_client]

    knn = NearestNeighbors(n_neighbors=10, algorithm="auto").fit(data_train_rm)

    distances, indices = knn.kneighbors(data_client.values.reshape(1, -1))

    print("indices")
    print(indices)
    print("distances")
    print(distances)

    df_neighbors = data_train.iloc[indices[0], :]

    return df_neighbors


###########################################################
########################  Main  ###########################
###########################################################


def main():
    """Main function for displaying the sidebar with 3 tabs."""
    # data_train, data_test, shap_values, exp_value, logo = load_data()
    st.sidebar.image(logo)
    PAGES = ["Tableau clientèle", "Comparaison clientèle", "Visualisation score"]
    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.title("Pages")
    selection = st.sidebar.radio("Go to", PAGES)
    if selection == "Tableau clientèle":
        tab_client(data_test)
    if selection == "Comparaison clientèle":
        st.markdown(
            f'<p style="background-color:#e41d35;color:#ffffff;font-size:24px;border-radius:2%;">{"Comparaison clientèle"}</p>',
            unsafe_allow_html=True,
        )
        client, idx_client = get_client()
        gender = "Homme" if infos_client(client, "CODE_GENDER") == 0 else "Femme"
        age = int(round(abs(infos_client(client, "DAYS_BIRTH")), 0))
        st.markdown(
            "Sexe: <span style='color:green'>" + gender + "</span>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "Age: <span style='color:green'>" + str(age) + "</span>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "Enfants: <span style='color:green'>"
            + str(infos_client(client, "CNT_CHILDREN"))
            + "</span>",
            unsafe_allow_html=True,
        )

        chk_infos = st.checkbox(
            "Cochez, si vous voulez des informations sur le client."
        )

        if chk_infos:
            st.write(
                "Total revenus client :",
                str(infos_client(client, "AMT_INCOME_TOTAL")),
                "$",
            )

            # Create two columns for the plots
            col1, col2 = st.columns(2)

            # Plot income distribution in the first column
            with col1:
                plot_income_distribution(data_train, client)

            # Plot age distribution in the second column
            with col2:
                plot_age_distribution(data_train, client)

            st.write(
                "Montant du crédit :", str(infos_client(client, "AMT_CREDIT")), "$"
            )
            st.write("Annuités crédit :", str(infos_client(client, "AMT_ANNUITY")), "$")
            st.write(
                "Montant du bien pour le crédit :",
                str(infos_client(client, "AMT_GOODS_PRICE")),
                "$",
            )

        # Affichage des dossiers similaires
        chk_neighbors = st.checkbox(
            "Cochez, si vous voulez comparer avec des dossiers similaires?"
        )

        if chk_neighbors:
            similar_id = load_neighbors(data_test, idx_client)
            st.markdown(
                "<u>Groupe des 10 clients similaires :</u>", unsafe_allow_html=True
            )
            st.write(similar_id)
            st.markdown("<i>Target 1 = Client en faillite</i>", unsafe_allow_html=True)
            st.markdown("<i>Target 0 = Client solvable</i>", unsafe_allow_html=True)

    if selection == "Visualisation score":
        st.markdown(
            f'<p style="background-color:#e41d35;color:#ffffff;font-size:24px;border-radius:2%;">{"Visualisation score"}</p>',
            unsafe_allow_html=True,
        )
        client2, idx_client2 = get_client()
        gender2 = "Homme" if infos_client(client2, "CODE_GENDER") == 0 else "Femme"
        age2 = int(round(abs(infos_client(client2, "DAYS_BIRTH")), 0))
        st.markdown(
            "Sexe: <span style='color:green'>" + gender2 + "</span>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "Age: <span style='color:green'>" + str(age2) + "</span>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "Enfants: <span style='color:green'>"
            + str(infos_client(client2, "CNT_CHILDREN"))
            + "</span>",
            unsafe_allow_html=True,
        )
        score_viz(data_test_rm, client2, idx_client2)


if __name__ == "__main__":
    main()

# to run the app in local : streamlit run dashboard.py

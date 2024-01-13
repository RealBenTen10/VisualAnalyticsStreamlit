import xgboost as xgb
import pandas as pd
import streamlit as st
import altair as alt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import numpy as np
import shap #Python Version 3.11 benötigt
#import lightgbm as lgb
#from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import warnings
#from pysankey import sankey
#from pySankey.sankey import sankey
import seaborn as sns
from pySankeyBeta import sankey


# Daten laden
def load_data():
    df = pd.read_csv('lucas_organic_carbon_training_and_test_data.csv')
    df_target = pd.read_csv('lucas_organic_carbon_target.csv')
    return df, df_target

# Aufteilung der Daten in Trainings und Testsets
def prepare_data(df, df_target):
    le = LabelEncoder()
    df_target_encoded = le.fit_transform(df_target.values.ravel())
    sc_x = StandardScaler()
    X_standardized = pd.DataFrame(sc_x.fit_transform(df), columns=df.columns)
    data_train, data_test, target_train, target_test = train_test_split(X_standardized, df_target_encoded, test_size=0.25, random_state=42)
    return data_train, data_test, target_train, target_test, le


# -------------------Modeltraning--------------------
# XGBoost-Parameter
def set_params_xgb():
    params_xgb = {
        'device': 'cuda',
        'tree_method': 'hist',
        'booster': 'gbtree',
        'objective': 'multi:softmax',
        'num_class': 5,
        'eval_metric': 'mlogloss'
    }
    return params_xgb

# Modelltraining XGBoost
def train_model(data_train, target_train, params_xgb):
    dtrain = xgb.DMatrix(data_train, label=target_train)
    bst = xgb.train(params_xgb, dtrain)
    return bst

# LightGBM-Parameter
def set_params_lgb():
    params_lgb = {
        'learning_rate': 0.19,
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        'max_depth': 3,
        'num_leaves': 12,
        'num_class': 5
    }
    return params_lgb

# Modelltraining für LightGBM
def train_model_lgb(data_train, target_train, params_lgb):
    d_train = lgb.Dataset(data_train, label=target_train)
    clf = lgb.train(params_lgb, d_train, 100)
    return clf

def calculate_accuracy_and_predictions(model, data_test, target_test):
    dtest = xgb.DMatrix(data_test)
    y_pred = model.predict(dtest)
    accuracy = accuracy_score(target_test, y_pred)
    return accuracy, y_pred

# Berechnung der Genauigkeit und Vorhersagen für LightGBM
def calculate_accuracy_and_predictions_lgb(model, data_test, target_test):
    y_pred_1 = model.predict(data_test)
    lightgbm_pred = [np.argmax(line) for line in y_pred_1]
    accuracy = accuracy_score(target_test, lightgbm_pred)
    return accuracy, lightgbm_pred

#Catboost
# CatBoost-Parameter
def set_params_catboost():
    params_catboost = {
        'iterations': 100,
        'depth': 3,
        'learning_rate': 0.19,
        'loss_function': 'MultiClass',
        'verbose': True,
        'task_type': "GPU", #GPU wenn vorhanden
        'devices': '0:1'
    }
    return params_catboost

# Modelltraining für CatBoost
def train_model_catboost(data_train, target_train, params_catboost):
    cat_model = CatBoostClassifier(**params_catboost)
    cat_model.fit(data_train, target_train)
    return cat_model

# Berechnung der Genauigkeit und Vorhersagen für CatBoost
def calculate_accuracy_and_predictions_catboost(model, data_test, target_test):
    y_pred_cat = model.predict(data_test)
    accuracy = accuracy_score(target_test, y_pred_cat)
    return accuracy, y_pred_cat.flatten()



# -------------------Grafiken plotten--------------------
# Confusion Matrix
def plot_confusion_matrix(y_true, y_pred, class_names):
    # Berechnung der Konfusionsmatrix
    cm = confusion_matrix(y_true, y_pred)
    # Erstellung der Konfusionsmatrix-Anzeige
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    # Erstellung des Plots
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax, cmap=plt.cm.Blues)
    #plt.title('Confusion Matrix')
    plt.show()
    return fig


# Shapley Values
# Funktion zum Generieren der Shapley-Werte
def generate_Shapley(X, model):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    return shap_values, explainer


# Sankey Diagram

def generate_Sankey(predictions, actuals, le):
    # Kategorie-Mapping für Sankey
    category_mapping_sankey = {0: 'high', 1: 'low', 2: 'moderate', 3: 'very high', 4: 'very low'}

    # Erstellen eines DataFrames mit den vorhergesagten und tatsächlichen Werten
    sankey_data = pd.DataFrame({
        'Predicted': [category_mapping_sankey[val] for val in predictions],
        'Actual': [category_mapping_sankey[val] for val in actuals]
    })

    # Gruppieren der Daten und Zählen der Übergänge
    transition_counts = sankey_data.groupby(['Predicted', 'Actual']).size().reset_index(name='count')

    # Erstellen von Listen für die Knoten und Verbindungen des Sankey-Diagramms
    labels = list(set(transition_counts['Predicted']).union(set(transition_counts['Actual'])))
    label_map = {label: i for i, label in enumerate(labels)}
    sources = transition_counts['Predicted'].map(label_map).tolist()
    targets = transition_counts['Actual'].map(label_map).tolist()
    weights = transition_counts['count'].tolist()

    # Erstellen des Sankey-Diagramms
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
        ),
        link=dict(
            source=sources,
            target=targets,
            value=weights
        )
    )])

    # Rückgabe des Sankey-Diagramms
    return fig


_ = """
def generate_Sankey(predictions, actuals, le):
    # Kategorie-Mapping für Sankey
    category_mapping_sankey = {0: 'high', 1: 'low', 2: 'moderate', 3: 'very high', 4: 'very low'}

    # Erstellen eines DataFrames mit den vorhergesagten und tatsächlichen Werten
    sankey_data = pd.DataFrame({
        'Predicted': [category_mapping_sankey[val] for val in predictions],
        'Actual': [category_mapping_sankey[val] for val in actuals]
    })

    # Erstellen des Sankey-Diagramms mit pySankey
    sankey(
        left=sankey_data['Predicted'],
        right=sankey_data['Actual'],
        aspect=20,
        fontsize=12
    )
"""



# -------------------Streamlit Dashboard--------------------
def main():


    st.set_page_config(layout="wide")

    st.title('ModelWise 2023')

    df, df_target = load_data()
    #data_train, data_test, target_train, target_test, le = prepare_data(df, df_target)
    array = np.array([0, 1, 2, 3, 4])
    category_mapping_sankey= {0:'high', 1:'low', 2:'moderate', 3:'very high', 4:'very low'}
    colors = {'high': 'orange', 'low': 'green', 'moderate': 'yellow', 'very high': 'red', 'very low': 'blue'}
    class_names = ['high', 'low', 'moderate', 'very high', 'very low']

    # -------------------Datensatz auswählen--------------------
    if df is not None:
        data = df
        st.write('Datensatz anpassen')
        #st.dataframe(data)

        # Beispiel eines einfachen Filters
        if st.checkbox('Wöhle Features aus zur Optimierung'):
            all_columns = data.columns.tolist()
            if st.checkbox('Select All'):
                selected_columns = all_columns[0:]
            else:
                selected_columns = st.multiselect('Select columns', all_columns, all_columns[0])
                st.dataframe(data[selected_columns])
            df = data[selected_columns]
            #st.dataframe(df)

            # Beispiel eines einfachen Diagramms
            if st.button('Modelle trainieren'):
                #df_new, df_target = load_data()
                data_train, data_test, target_train, target_test, le = prepare_data(df, df_target)
                st.write('Datensatz ausgewählt')


    # -------------------Modell auswählen--------------------
    model_option = st.selectbox('Wähle ein Modell für die Analyse:', ['Wähle das Modell aus','XGBoost', 'LightGBM', 'CatBoost'])

    warnings.filterwarnings('ignore')

    if model_option == 'Wähle das Modell':
        raise NotImplementedError

    # XGBoost
    elif model_option == 'XGBoost':
        try:
            #Modeltraining
            params = set_params_xgb()
            bst = train_model(data_train, target_train, params)
            accuracy, xgboost_pred = calculate_accuracy_and_predictions(bst, data_test, target_test)
            st.header(f'***Genauigkeit von XGBoost: {accuracy:.2f}***')

            if st.checkbox('Sankey Diagram erstellen'):
                st.header('***Sankey Diagram für XGBoost:***')
                # Generiere das Sankey-Diagramm
                fig = generate_Sankey(xgboost_pred, target_test, le)


                st.plotly_chart(fig, use_container_width=True)

            if st.checkbox('Shapley Values berechnen'):
                #with col1:
                st.subheader('***Shapley-Werte für XGBoost:***')
                    # Shapley-Werte generieren
                shap_values, explainer = generate_Shapley(data_train, bst)
                st.set_option('deprecation.showPyplotGlobalUse', False)
                shap.summary_plot(shap_values, data_train, plot_type="bar", show=False)
                st.pyplot(bbox_inches='tight', pad_inches=0)

            if st.checkbox('Confusion Matrix generieren'):
                #col1, col2 = st.columns(2)
                #with col2:
                st.subheader('***Confusionmatrix für XGBoost:***')
                    # Berechnung der Confusion Matrix
                fig = plot_confusion_matrix(target_test, xgboost_pred, class_names)
                st.pyplot(fig)


            st.write('XGBoost-Modell wurde ausgewählt.')

        except:
            st.write('Bitte trainiere dein Modell!')
            st.stop()

    #LightGBM
    elif model_option == 'LightGBM':
        try:
            # LightGBM-Modell
            params_lgb = set_params_lgb()
            clf = train_model_lgb(data_train, target_train, params_lgb)

            accuracy, lightgbm_pred = calculate_accuracy_and_predictions_lgb(clf, data_test, target_test)
            st.header(f'***Genauigkeit von LightGBM: {accuracy:.2f}***')

            if st.checkbox('Sankey Diagram erstellen'):
                # Generiere das Sankey-Diagramm für LightGBM
                st.header('***Sankey Diagram für LightGBM:***')
                generate_Sankey(lightgbm_pred, target_test, le)
                st.plotly_chart(generate_Sankey(lightgbm_pred, target_test, le), use_container_width=True)

            if st.checkbox('Shapley Values berechnen'):

                    # Shapley Values
                shap_values, explainer = generate_Shapley(data_test, clf)
                st.header('***Shapley-Werte für LightGBM:***')
                shap_values, explainer = generate_Shapley(data_train, clf)
                shap.summary_plot(shap_values, data_train, plot_type="bar", show=False)
                st.pyplot(bbox_inches='tight', pad_inches=0)

            if st.checkbox('Confusion Matrix generieren'):

                    # Berechnung der Confusion Matrix
                fig = plot_confusion_matrix(target_test, lightgbm_pred, class_names)
                st.pyplot(fig)

            st.write('LightGBM-Modell wurde ausgewählt.')

        except:
            st.write('Bitte trainiere dein Modell!')
            st.stop()

    # CatBoost
    elif model_option == 'CatBoost':
        try:
            params_catboost = set_params_catboost()
            cat_model = train_model_catboost(data_train, target_train, params_catboost)


            accuracy, catboost_pred = calculate_accuracy_and_predictions_catboost(cat_model, data_test, target_test)
            st.header(f'***Genauigkeit von CatBoost: {accuracy:.2f}***')

            if st.checkbox('Sankey Diagram erstellen'):
                # Generiere das Sankey-Diagramm für CatBoost
                st.header('***Sankey Diagram für CatBoost:***')
                fig = generate_Sankey(catboost_pred, target_test, le)
                st.plotly_chart(fig, use_container_width=True)

            if st.checkbox('Shapley Values berechnen'):

                    # Shapley-Werte generieren
                shap_values, explainer = generate_Shapley(data_train, cat_model)
                st.header('***Shapley-Werte für CatBoost:***')
                shap.summary_plot(shap_values, data_train, plot_type="bar", show=False)
                st.pyplot(bbox_inches='tight', pad_inches=0)

            if st.checkbox('Confusion Matrix generieren'):

                    # Berechnung der Confusion Matrix
                fig = plot_confusion_matrix(target_test, catboost_pred, class_names)
                st.pyplot(fig)

            st.write('CatBoost-Modell wurde ausgewählt.')

        except:
            st.write('Bitte trainiere dein Modell!')
            st.stop()

    # -------------------DataFrame selektieren--------------------
_="""
    if df is not None:
        data = df
        st.write('Optimierung')
        #st.dataframe(data)

        # Beispiel eines einfachen Filters
        if st.checkbox('Wöhle Features zur Optimierung'):
            all_columns = data.columns.tolist()
            selected_columns = st.multiselect('Select columns', all_columns, all_columns[0])
            df_new = data[selected_columns]
            st.dataframe(df_new)

            # Beispiel eines einfachen Diagramms
            if st.button('Feature Update'):
                df_new, df_target = load_data()
                data_train, data_test, target_train, target_test, le = prepare_data(df_new, df_target)
                st.write('Dataset = Feature Update')
"""


if __name__ == "__main__":
    main()
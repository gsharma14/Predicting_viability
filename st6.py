import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
def app():
    #app heading
    st.write("""
    # Target Viability Prediction App
    This app predicts the ***investment viability*** of the target type!
    """)
    #st.sidebar.header("IS YOUR TARGET HOT or NOT?")
    #creating sidebar for user input features
    #st.sidebar.header('User Input Parameters')
   
  
    def user_input_features():
        drug_desc = st.selectbox("Drug Descriptor"  "(0=Adoptive Cell Therapy, 1=Antiinfective Therapy, 2=Antiinflammatory Therapy, 3=Autologous therapy, 4=Chemotherapy, 5=DNA-based Targeted therapy, 6=Dendritic Cell Therapy, 7=Immunotherapy, 8=Multispecific Monoclonal Antibody, 9=Not Given, 10=Opiod/Non-Opiod Painkiller, 11=RNA-based Targeted therapy, 12=Radiopharmaceutical, 13=Recombinant, 14=Targeted Therapy)",["0","1","2","3","4","5","6","7","8","9","10","11","12","13","14"])
        target_type = st.selectbox("Target Type" "(0=CB-1, 1=CD3, 2=DNA, 3=DNA  Methyltransferase, 4=DNA Topoisomerase I, 5=DNA Topoisomerase II, 6=FCGR2B, 7=HSP, 8=IL11, 9=IL12R, 10=IL17RE, 11=IL4, 12=IL6, 13=Unknown, 14=kras, 15=magl, 16=neuropilin, 17=tgfb, 18=tigit, 19=tyk)",["0","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19"])
        company_type = st.selectbox("Company Type" "(0=Institution, 1=Private, 2=Public)",["0","1","2"])
        status = st.selectbox("Status" "(0=Completed, 1=Discontinued, 2=Inactive, 3=Marketed, 4=Ongoing, 5=Phase 2, 6=Planned, 7=Pre-Clinical, 8=Unknown)",["0","1","2","3","4","5","6","7","8"])
        hds = st.selectbox( "Highest Development Stage" "(0=Discontinued, 1=Discovery, 2=IND/CTA filed, 3=Inactive, 4=Marketed, 5=Phase 1, 6=Phase 2, 7=Phase 3, 8=Phase 4, 9=Pre-Clinical)",["0","1","2","3","4","5","6","7","8","9"])
        therapy_area = st.selectbox("therapy area" "(0=CNS, 1=Cardiovascular, 2=Dermatology, 3=Fibrotic Diseases, 4=Immunology, 5=Infectious Disease, 6=Metabolic Disorders, 7=Musculoskeletal Disorders, 8=Oncology, 9=Oncology, 10=Respiratory, 11=Toxicology)",["0","1","2","3","4","5","6","7","8","9","10","11"])
        indication = st.selectbox( "Indication" "(0=Cancer, 1=Cancer, 2=Fibrosis, 3=Obesity, 4=Solid Tumor, 5=AutoImmune Disorders, 6=Cancer, 7=Digestive Disease, 8=Drug Addiction, 9=Fibrosis, 10=Immunology, 11=Lung Disease, 12=Neurodegenerative Diseases, 13=PTSD/Depression, 14=Pain, 15=Psoriasis, 16=Rheumatoid Arthritis, 17=Skin Infections, 18=Solid Tumor, 19=Solid Tumor/Cancer)",["0","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19"])
        mol_type = st.selectbox("Molecular type" "(0=Oligonucleotide, 1=Polymer, 2=Protein, 3=Cell Therapy, 4=Gene Therapy, 5=Inhibitor, 6=Monoclonal Antibody, 7=Oncolytic Virus, 8=Small Molecule, 9=Unknown, 10=Vaccine)",["0","1","2","3","4","5","6","7","8","9","10"])
        trial_count = st.slider("Trial_count", 0, 100,1)
        public_news = st.slider("Count of public news", 1, 100,1)
        deals = st.slider("count of deals", 0, 50,1)
        deal_status = st.selectbox("Deal status" "(0=Announced, 1=Completed, 2=Completed/Announced, 3=Completed/Filing, 4=Completed/Pricing, 5=Terminated, 6=Unknown)",["0","1","2","3","4","5","6"])
        deal_amount = st.slider("Deal amount", 0, 160000,1)
        deal_type = st.selectbox("Deal Type" "(0=A, 1=CSA, 2=DO, 3=EO, 4=G, 5=LA, 6=ND, 7=P, 8=PE, 9=R, 10=VF)",["0","1","2","3","4","5","6","7","8","9","10"])
        data = {'drug_desc': drug_desc,
                'target_type': target_type ,
                'company_type': company_type ,
                'status': status ,
              'hds': hds,
              'therapy_area': therapy_area,
                'indication': indication,
                'mol_type': mol_type ,
                'trial_count': trial_count ,
                'public_news': public_news ,
              'deals': deals,
              'deal_status': deal_status,
                'deal_amount': deal_amount,
                'deal_type': deal_type
                }
        features = pd.DataFrame(data, index=[0])
        return features
    df = user_input_features()

    #st.subheader('User Input parameters')
    st.write(df)
    #reading csv file
    data=pd.read_csv("trial_data.csv")
    X = data.iloc[:,:-1].values
    Y = data.iloc[:,-1].values
    #random forest model
    rfc= RandomForestClassifier()
    rfc.fit(X, Y)
    #st.subheader('Wine quality labels and their corresponding index number')
    #st.write(pd.DataFrame({'wine quality': [3, 4, 5, 6, 7, 8 ]}))

    prediction = rfc.predict(df)
    #prediction_proba = rfc.predict_proba(df)
    st.subheader('Prediction')
    st.write(prediction)

    #st.subheader('Prediction Probability')
    #st.write(prediction_proba)








    


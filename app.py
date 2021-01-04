import streamlit as st
from predict_helper import predictt


def main():
    side_bar = st.sidebar

    # side_bar.title("")
    side_bar.markdown("** Select Model **")

    model_list = ["Logistic Regression", "Support vector classifier", "Random Forest"]

    model = side_bar.selectbox("", model_list)


    text = st.text_area("Enter your text here.")

    if text:

        result = predictt(text)

        if model == model_list[0]:
            r = result["lr"]
        
        elif model == model_list[1]:
            r = result["svc"]

        elif model == model_list[2]:
            r = result["rf"]

        if r == 0:
            st.success("Speech in entered text is not hatefull!")
        else:
            st.error("Speech in entered text is hatefull!")

if __name__=='__main__':
    main()
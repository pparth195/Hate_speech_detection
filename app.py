import streamlit as st
from predict_helper import predictt 


def main():

    st.title("Hate speech detection")

    # st.markdown(" [link] (https://discuss.streamlit.io/t/is-it-possible-to-add-link/1027/3) ")

    description = """I have trained three models on hate speech tweeter dataset.
                    You can check out data and code [here] (https://github.com/pparth195/Hate_speech_detection). 
                    I have find best parameters using gridsearchCV and all three models are deployed here. 
                    You can enter the text below and find out if the text contains hateful tone or not. 
                    Keep in mind that models have been trained on tweets so it would give best result when 
                    entered text is small form ideally a tweet."""

    st.markdown(description)

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
import argparse
import os
from langchain.vectorstores import DeepLake
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import openai
import streamlit as st
from streamlit_chat import message

def get_text():
    """Create a Streamlit input field and return the user's input."""
    input_text = st.text_input("", key="input")
    return input_text


def search_db(db, query):
    """Search for a response to the query in the DeepLake database."""
    # Create a retriever from the DeepLake instance
    retriever = db.as_retriever()
    # Set the search parameters for the retriever
    retriever.search_kwargs["distance_metric"] = "cos"
    retriever.search_kwargs["fetch_k"] = 100
    retriever.search_kwargs["maximal_marginal_relevance"] = True
    retriever.search_kwargs["k"] = 10
    # Create a ChatOpenAI model instance
    model = ChatOpenAI(model="gpt-3.5-turbo")
    # Create a RetrievalQA instance from the model and retriever
    qa = RetrievalQA.from_llm(model, retriever=retriever)
    # Return the result of the query
    return qa.run(query)



def run_chat_app(activeloop_dataset_path):
    """Run the chat application using the Streamlit framework."""
    # Set the title for the Streamlit app
    st.title(f"Mynd Chatbot for Perl")

    # Set the OpenAI API key from the environment variable
    openai.api_key = os.environ.get("OPENAI_API_KEY")

    # Create an instance of OpenAIEmbeddings
    embeddings = OpenAIEmbeddings()

    # Create an instance of DeepLake with the specified dataset path and embeddings
    db = DeepLake(
        dataset_path=activeloop_dataset_path,
        read_only=True,
        embedding_function=embeddings,
    )

    # Initialize the session state for generated responses and past inputs
    if "generated" not in st.session_state:
        st.session_state["generated"] = ['Hi there']

    if "past" not in st.session_state:
        st.session_state["past"] = ['hello']

    # Get the user's input from the text input field
    user_input = get_text()

    # If there is user input, search for a response using the search_db function
    if user_input:
        output = search_db(db, user_input)
        st.session_state.past.append(user_input)
        st.session_state.generated.append(output)

    # If there are generated responses, display the conversation using Streamlit
    # messages
    if st.session_state.get("generated", []):
        for i in range(len(st.session_state["generated"])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
            message(st.session_state["generated"][i], key=str(i))

if __name__ == "__main__":
    run_chat_app('hub://anurag0961/mynd_dataset')
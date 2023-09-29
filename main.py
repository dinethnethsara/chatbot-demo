import base64
import pandas as pd
import openai
import numpy as np
import streamlit as st
import uuid
import requests
import datetime
from openai.embeddings_utils import distances_from_embeddings
from streamlit_chat import message
from streamlit.components.v1 import html

api_url = 'https://connect-dev.schoolinfo.app/api/chat-bot/'

log_credentials = base64.b64encode(f"{st.secrets['log']['username']}:{st.secrets['log']['password']}".encode()).decode()

openai.api_key = st.secrets["api_keys"]["openai"]

df = pd.read_csv('embeddings.csv', index_col=0)
df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)
df.head()

district_name = 'Bakersfield Adult School'

welcome_message = f'Thank you for your interest in {district_name}! What would you like to learn more about?'

messages = [
    {
        'role': 'system',
        'content': f'You are a friendly assistant that answers {district_name} related questions. '
                   'Answer the question as truthfully as possible using the provided context, '
                   'and if the answer is not contained within the text below, say \"I don\'t know.\"'
                   'Be proactive and offer some example question that you can answer.'
    },
    {
        'role': 'assistant',
        'content': welcome_message
    }
]


def create_context(
        question, df, max_len=2500,
):
    """
    Create a context for a question by finding the most similar context from the dataframe
    """

    # Get the embeddings for the question
    q_embeddings = openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding']

    # Get the distances from the embeddings
    df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values, distance_metric='cosine')

    returns = []
    cur_len = 0

    # Sort by distance and add the text to the context until the context is too long
    for i, row in df.sort_values('distances', ascending=True).iterrows():

        # Add the length of the text to the current length
        cur_len += row['n_tokens'] + 4

        # If the context is too long, break
        if cur_len > max_len:
            break

        # Else add it to the text that is being returned
        returns.append(row["text"])

    # Return the context
    return "\n\n###\n\n".join(returns)


def get_completion_from_messages(question='', model="gpt-3.5-turbo", temperature=0):
    question = f'Probably related to the {district_name}. {question}'
    context = create_context(
        question,
        df,
    )

    messages.append({
        'role': 'system',
        'content': f'Here is some background information for the next question: \n\n{context}'
    })
    messages.append({'role': 'user', 'content': question})

    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature,  # this is the degree of randomness of the model's output
        )
    except Exception as e:
        return None

    choices = response.get("choices", [])
    if len(choices) > 0:
        return choices[0]["message"]["content"].strip(" \n")
    else:
        return None


def create_conversation_log(conversation_id):
    if 'conversation_created' not in st.session_state:
        payload = {
            'conversationId': conversation_id,
            'districtName': district_name,
            'createdAt': datetime.datetime.utcnow().isoformat()
        }
        headers = {
            'Authorization': f'Basic {log_credentials}',
            'X-SIA-TENANT': 'DevAlpha'
        }
        response = requests.post(api_url + 'create-conversation-log', json=payload, headers=headers)
        if 200 <= response.status_code < 300:
            st.session_state['conversation_created'] = 'true'

        print(f'Conversation log created. Response status code: {response.status_code}')


def log_message(is_user, message_text):
    payload = {
        'conversationId': st.session_state['conversation_identifier'],
        'isUser': is_user,
        'message': message_text,
        'timeStamp': datetime.datetime.utcnow().isoformat()
    }
    headers = {
        'Authorization': f'Basic {log_credentials}',
        'X-SIA-TENANT': 'DevAlpha'
    }
    response = requests.post(api_url + 'log-message', json=payload, headers=headers)
    print(f'Message logged. Response status code: {response.status_code}')


def user_prompt_submit():
    user_input = st.session_state.input
    st.session_state.prompt = user_input
    st.session_state['input'] = ''
    if 'conversation_identifier' not in st.session_state:
        convo_id = str(uuid.uuid4())
        st.session_state['conversation_identifier'] = convo_id
        create_conversation_log(convo_id)
    log_message(True, user_input)


st.set_page_config(page_title=f"{district_name} ChatBot", page_icon="🤖", layout="wide")


if 'prompt' not in st.session_state:
    st.session_state.prompt = ''

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

html(f"""
<script>
    function scroll(index){{
        setTimeout(() => {{
            const container = parent.document.querySelector('.block-container');
            if (!container) return false;
            container.scrollTop = container.scrollHeight;
            if (index > -1) {{
                scroll(-1);
            }}
        }}, "3000");
    }}
    scroll({len(st.session_state['generated'])});
</script>
""")

message(welcome_message, key='-1')

if st.session_state.prompt:
    assistant_response = get_completion_from_messages(st.session_state.prompt, temperature=0)
    st.session_state.past.append(st.session_state.prompt)
    st.session_state.generated.append(assistant_response)
    messages.append({'role': 'assistant', 'content': assistant_response})
    log_message(False, assistant_response)

if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])):
        message(st.session_state['past'][i], is_user=True, avatar_style="adventurer", key=str(i) + '_user')
        message(st.session_state["generated"][i], key=str(i))

st.text_input(key='input',
              on_change=user_prompt_submit,
              label='Type in your question here',
              label_visibility='hidden',
              placeholder='Type in your question here')

styl = f"""
<style>
    .stTextInput {{
        position: fixed;
        bottom: 10px;
        left: 0;
        right: 0;
        width: 96vw;
        margin: auto;
    }}    
    
    .block-container {{
        position: fixed !important;
        bottom: 1rem !important;
        padding: 0 !important;
        overflow-y: auto !important;
        overflow-x: hidden !important;
        max-height: 90vh !important;
        width: 96vw !important;
    }}
    
    #MainMenu {{
        display: none;
    }}
    
    footer {{
        display: none;
    }}
</style>
"""
st.markdown(styl, unsafe_allow_html=True)

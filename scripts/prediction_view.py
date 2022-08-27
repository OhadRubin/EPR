# To run use `streamlit run myapp_v0.py`
import streamlit as st
import glob
import pandas as pd
import json
import os
from src.utils import eval_datasets
from st_aggrid import AgGrid

st.set_page_config(layout="wide")

#>>>
#Define our options
DATASET_LIST = ["smcalflow", "mtop", "break"]
LM_LIST = ["codex", "gpt3", "gptj", "gptneo"]


file_col, col_dataset, col_lm, col_corr = st.columns([1, 1, 1, 1])
with col_corr:
    acc_filter = st.radio("Filter by correctness:", ["All", "Correct", "Incorrect"])
with col_dataset:
    datasets = st.multiselect("Dataset:", options=DATASET_LIST, default=DATASET_LIST)
with col_lm:
    lms = st.multiselect("Language Model:", options=LM_LIST, default=LM_LIST)
print(datasets)
print(lms)
files = glob.glob("data/*prede.json")
files.sort(key=os.path.getmtime)
files = files[::-1]
# Filter the files based on the selected options
files = [x for x in files if eval_datasets.get_dataset(x) in datasets]
files = [x for x in files if eval_datasets.get_lm(x) in lms]
#<<<

with file_col:
    filename = st.selectbox(
        "Pick a file", files, format_func=lambda x: x.split("data/")[1]
    )



@st.experimental_singleton
def get_data(filename):
    with open(filename) as f:
        data = json.load(f)
    return data
data = get_data(filename)

if acc_filter == "Correct":
    data = [x for x in data if x["acc"]]
if acc_filter == "Incorrect":
    data = [x for x in data if (not x["acc"])]

questions = [[x["question"], i] for i, x in enumerate(data)]
import random
def set_num_in():

	st.session_state.num_in = random.randint(0, len(questions) - 1)
    # ,index=0,on_change=set_num_in
st.sidebar.button('Shuffle', on_click=set_num_in)

# question_id = st.selectbox("Pick a question", questions, format_func=lambda x: x[0])[1]
num_in = st.sidebar.number_input('Question index:', 0, len(data),key="num_in")
curr_el = data[num_in]
st.sidebar.write("## Question")
st.sidebar.write(curr_el['question'])


gold_col, pred_col = st.columns([1, 1])
with gold_col:
    st.write(f"## Gold")
    st.write(f"{eval_datasets.get_answer(curr_el)} ✅")
with pred_col:
    st.write(f"## Generated")
    st.write(
        f"{curr_el['generated'].split('<|endoftext|>')[0]} {'✅'  if curr_el['acc'] else '❌'}"
    )


st.write("## Prompts")
df = pd.DataFrame(curr_el["ctxs"])
df["target"] = df["text"].str.split("\t").str[1]
df["question"] = df["text"].str.split("\t").str[0]

df = df.drop(columns=["title", "id", "has_answer", "text"])


AgGrid(
    df,
    editable=True,
    width="100%",
    enable_enterprise_modules=True,
    fit_columns_on_grid_load=False,
)  # https://www.ag-grid.com/javascript-data-grid/
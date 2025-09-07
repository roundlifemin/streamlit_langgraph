
import streamlit as st
from langgraph.graph import StateGraph
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class MyState(dict):
    sentence: str
    emotion: str
    advice: str

def classify_emotion(state):
    text = state["sentence"]
    prompt = f"다음 영어 문장을 감정 분석해서 '긍정적', '부정적', '중립' 중 하나로 분류하세요:\n{text}"
    result = llm.invoke(prompt)
    return {"emotion": result.content.strip()}

def generate_advice(state):
    emotion = state["emotion"]
    prompt = f"감정이 '{emotion}'일 때 적절한 조언을 한국어로 생성해주세요. 예의 있고 따뜻한 말투로 해주세요."
    result = llm.invoke(prompt)
    return {"advice": result.content.strip()}

graph = StateGraph(MyState)
graph.add_node("감정분석", RunnableLambda(classify_emotion))
graph.add_node("조언생성", RunnableLambda(generate_advice))
graph.set_entry_point("감정분석")
graph.add_edge("감정분석", "조언생성")
graph.set_finish_point("조언생성")

runnable = graph.compile()

st.title("감정 분석 + 조언 생성기")

sentence = st.text_area("영어 문장을 입력하세요:", height=200)

if st.button("분석하기"):
    if not sentence.strip():
        st.warning("문장을 입력해주세요.")
    else:
        with st.spinner("감정 분석 중..."):
            result = runnable.invoke({"sentence": sentence})
        
        st.subheader("감정 분석 결과")
        st.success(f"감정: {result['emotion']}")
        
        st.subheader("조언")
        st.info(result["advice"])

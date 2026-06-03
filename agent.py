from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_openai import AzureChatOpenAI
from langfuse import observe, propagate_attributes, get_client
from dotenv import load_dotenv
import os
from pydantic import BaseModel
load_dotenv()

langfuse = get_client()

class OutputRes(BaseModel):
    is_translated: bool
    reason: str

class Translate(BaseModel):
    translate: str

key1 = os.getenv("AZURE_OPENAI_API_KEY1")
key2 = os.getenv("AZURE_OPENAI_API_KEY2")
endpoint1= os.getenv("AZURE_OPENAI_ENDPOINT1")
endpoint2 = os.getenv("AZURE_OPENAI_ENDPOINT2")

model_name = "gpt-5.4-nano"
deployment_name = "gpt-5.4-nano"

llm1 = AzureChatOpenAI(
    api_key=key1,
    model_name=model_name,
    azure_deployment=deployment_name,
    azure_endpoint=endpoint1, 
    temperature=0,
    api_version="2024-12-01-preview",
).with_structured_output(Translate)

llm2 = AzureChatOpenAI(
    api_key=key1,
    model_name=model_name,
    azure_deployment=deployment_name,
    azure_endpoint=endpoint1, 
    temperature=0,
    api_version="2024-12-01-preview",
).with_structured_output(OutputRes)

class AgentState(TypedDict):
    question: str
    translated_sentence: str
    is_translated: bool

def translate_sentence(state: AgentState) -> AgentState:
    prompt = """
        You are given a sentence, translate it into Hindi
"""

    message = [
        (
            "system",
            prompt
        ),
        (
            "user",
            state["question"]
        )
    ]

    with langfuse.start_as_current_observation(
        as_type="generation",
        name="user-translation-pipeline",
        input=f"{state["question"]}",
    ) as root_gen:
        result = llm1.invoke(message)
        root_gen.update(output=result.translate)
        return {**state, "translated_sentence":result.translate}


def check_translate(state: AgentState) -> AgentState:
    prompt = """
        You are given a hindi sentence, check if it is translated correctly.
"""
    message = [
        (
            "system",
            prompt
        ),
        (
            "user",
            f"Hindi Sentence: {state["translated_sentence"]}\n English Sentence: {state["question"]}"
        )
    ]
    
    with langfuse.start_as_current_observation(
        as_type="evaluator",
        name="user-translation-eval",
        input=f"{state["translated_sentence"]}\n English Sentence: {state["question"]}"
    ) as root_eval:
        result = llm2.invoke(message)
        print(f"\nREASON: {result.reason}\n")
        root_eval.update(output=result)
        return {**state, "is_translated":result.is_translated}


def define_workflow(AgentState):
    workflow = StateGraph(AgentState)
    workflow.add_node("translate", translate_sentence)
    workflow.add_node("check", check_translate)

    workflow.set_entry_point("translate")
    workflow.add_edge("translate", "check")
    workflow.add_edge("check", END)

    return workflow

def compile_workflow(AgentState):
    workflow = define_workflow(AgentState)
    app = workflow.compile()
    return app

app = compile_workflow(AgentState)

#user entry_point
#manually adding the observations to langfuse
@observe(as_type="chain")
def ask_question(question):
    initial_state = {
        "question": "",
        "translated_sentence": "",
        "is_translated": False
    }

    with propagate_attributes(user_id="raaggee08"):
        initial_state["question"] = question
        result = app.invoke(initial_state)
        return result

ques = "Kya yeh sahi hai."
correct_translate = ask_question(ques)
print(correct_translate["is_translated"])


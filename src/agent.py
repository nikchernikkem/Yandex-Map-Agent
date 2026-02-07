import os
import logging
import sys
import json
from pathlib import Path

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, MessagesState
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage

from dotenv import load_dotenv

from src.tools import tools

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

SYSTEM_PROMPT = (Path(__file__).resolve().parent / "prompts" / "system_prompt_base.jinja").read_text(encoding="utf-8")

logger = logging.getLogger("agent_test")
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)

fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
handler.setFormatter(fmt)

logger.handlers.clear()
logger.addHandler(handler)
logger.propagate = False


# ----------------------------
# LLM (OpenAI-compatible)
# ----------------------------
# Настраиваем модель и биндим к ней инструменты
def create_llm(selected_tools=None):
    llm = ChatOpenAI(
        # model="google/gemini-2.0-flash-exp:free",
        model="nvidia/nemotron-3-nano-30b-a3b:free",
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        max_tokens=None,
    )
    if selected_tools:
        llm = llm.bind_tools(selected_tools)
    return llm

# ----------------------------
# LangGraph nodes
# ----------------------------
def build_graph(selected_tools=None, system_prompt: str = SYSTEM_PROMPT):
    selected_tools = list(selected_tools) if selected_tools else []
    has_tools = len(selected_tools) > 0
    llm = create_llm(selected_tools if has_tools else None)

    # функция, которая определяет нужно ли вызывать инструменты 
    # или результат уже получен
    def should_continue(state: MessagesState):
        if not has_tools:
            return END
        last_message = state["messages"][-1]
        # if last_message.tool_calls:
        # If there are tool calls, continue to tools node
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
         # Otherwise, end the graph
        return END

    # функция для ноды взаимодейтсвия с LLM
    def agent_node(state: MessagesState):
        logger.debug("Agent node starts")
        logger.debug(f"Agent node state: {state}")

        messages = [SystemMessage(content=system_prompt)] + list(state["messages"])
        logger.debug(f"Incoming messages count: {len(messages)}")

        response = llm.invoke(messages)

        if getattr(response, "tool_calls", None):
            logger.info("LLM requested tool calls:")
            logger.info(json.dumps(response.tool_calls, indent=2, ensure_ascii=False))
        if getattr(response, "content", None):
            logger.info(response.content)

        return {"messages": [response]}

    workflow = StateGraph(MessagesState)
    # задаём ноды
    workflow.add_node("agent", agent_node)
    # задаём переходы между нодами
    # входная нода - agent
    workflow.set_entry_point("agent")

    if has_tools:
        # встроенная в langgraph нода вызова инструментов
        tool_node = ToolNode(selected_tools)
        # задаём ноды
        workflow.add_node("tools", tool_node)
        # после agent вызываем should_continue, 
        # чтобы определить что делать дальше
        workflow.add_conditional_edges("agent", should_continue, ["tools", END])
        # после вызова инструментов всегда возвращаемся к ноде LLM, 
        # чтобы отдать ей результат вызова инструментов
        workflow.add_edge("tools", "agent")
    else:
        workflow.add_edge("agent", END)

    return workflow.compile()


graph = build_graph(tools)

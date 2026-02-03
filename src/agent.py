import os
import logging, sys
import json
from pathlib import Path

from langchain_openai import ChatOpenAI 
from langgraph.graph import StateGraph, END, MessagesState 
from langgraph.prebuilt import ToolNode 
from langchain_core.messages import HumanMessage, SystemMessage

from dotenv import load_dotenv
load_dotenv()

from src.tools import tools
# try:
#     from src.tools import tools
# except ModuleNotFoundError:
#     from tools import tools

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")


SYSTEM_PROMPT = (Path(__file__).resolve().parent / "system_prompt.jinja").read_text(encoding="utf-8")

# ----------------------------
# LLM (OpenAI-compatible)
# ----------------------------
# Настраиваем модель и биндим к ней инструменты
llm = ChatOpenAI(model="google/gemini-2.0-flash-exp:free", api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL, max_tokens=None).bind_tools(tools)
# llm = ChatOpenAI(model="meta-llama/llama-3.3-70b-instruct:free", 
#                  api_key=OPENAI_API_KEY, 
#                  base_url=OPENAI_BASE_URL, 
#                  max_tokens=None,
#                  streaming=False).bind_tools(tools)


logger = logging.getLogger("agent_test")
# logger.setLevel(logging.INFO)
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)

fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
handler.setFormatter(fmt)

# чтобы не было дублей
logger.handlers.clear()
logger.addHandler(handler)
logger.propagate = False


# ----------------------------
# LangGraph nodes
# ----------------------------
# функция, которая определяет нужно ли вызывать инструменты 
# или результат уже получен
def should_continue(state: MessagesState):
    messages = state["messages"]
    last_message = messages[-1]
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

    # messages = state["messages"]
    # messages.append(SystemMessage(content=SYSTEM_PROMPT))

    messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(state["messages"])
    logger.debug(f"Incoming messages count: {len(messages)}")

    response = llm.invoke(messages)

    # информация для отладки
    # print(json.dumps(response.tool_calls, indent=2,ensure_ascii=False))
    # print(json.dumps(response.content, indent=2,ensure_ascii=False))

    if getattr(response, "tool_calls", None):
        logger.info("LLM requested tool calls:")
        logger.info(json.dumps(response.tool_calls, indent=2, ensure_ascii=False))
    if getattr(response, "content", None):
        # logger.info("LLM content:")
        logger.info(response.content)
   
    return {"messages": [response]}

# встроенная в langgraph нода вызова инструментов
tool_node = ToolNode(tools)

workflow = StateGraph(MessagesState)

# задаём ноды
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)

# задаём переходы между нодами
# входная нода - agent
workflow.set_entry_point("agent")
# после agent вызываем should_continue, 
# чтобы определить что делать дальше
workflow.add_conditional_edges("agent", 
                               should_continue, 
                               ["tools", END])
# после вызова инструментов всегда возвращаемся к ноде LLM, 
# чтобы отдать ей результат вызова инструментов
workflow.add_edge("tools", "agent")

graph = workflow.compile()

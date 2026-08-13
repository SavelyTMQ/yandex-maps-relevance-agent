"""LangGraph для агента."""
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage
from langchain_gigachat import GigaChat

from config import GIGACHAT_CREDENTIALS, GIGACHAT_MODEL, MAX_TOOL_CALLS
from agent.tools import TOOLS


def create_llm():
    """Создаёт LLM с function calling."""
    if not GIGACHAT_CREDENTIALS:
        raise ValueError(
            "GIGACHAT_CREDENTIALS не задан. Создайте .env файл из .env.example"
        )
    
    llm = GigaChat(
        credentials=GIGACHAT_CREDENTIALS,
        scope="GIGACHAT_API_PERS",
        model=GIGACHAT_MODEL,
        verify_ssl_certs=False,
        temperature=0,
        max_tokens=350,
    )
    return llm


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    tool_calls_count: int


def build_graph():
    """Строит и компилирует LangGraph."""
    llm = create_llm()
    llm_with_tools = llm.bind_tools(TOOLS)
    
    def agent_node(state: AgentState):
        if state["tool_calls_count"] >= MAX_TOOL_CALLS:
            forced = llm.invoke(state["messages"] + [HumanMessage(
                content="Лимит инструментов исчерпан. Дай финальный ответ в JSON."
            )])
            return {"messages": [forced]}
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}
    
    def route(state: AgentState):
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return "tools"
        return END
    
    def tools_node_wrapper(state: AgentState):
        result = ToolNode(TOOLS).invoke(state)
        return {
            "messages": result["messages"],
            "tool_calls_count": state["tool_calls_count"] + 1
        }
    
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node_wrapper)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    
    return graph.compile(), llm

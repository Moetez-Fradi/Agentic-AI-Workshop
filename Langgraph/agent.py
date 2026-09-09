import os

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.types import Command, interrupt

from config import MAX_TURNS, SAMPLE_PROMPT, SYSTEM_PROMPT
from llm import active_model_label, get_model
from security import ask_permission
from tools import DANGEROUS_TOOLS, TOOL_FUNCTIONS, TOOLS
from ui import console, print_error, print_thought, print_tool_call, print_tool_result

model_with_tools = get_model(TOOLS)


def call_model(state: MessagesState) -> dict:
    """Ask the model what to do next, given the conversation so far."""
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}


def run_tools(state: MessagesState) -> dict:
    """Execute every tool call from the last model turn, pausing on an
    interrupt() for human approval before any dangerous one runs.

    Split into two passes on purpose: LangGraph replays a node from the top
    each time it resumes from an interrupt() inside it, so all approvals
    are collected first (interrupt() calls are cheap to replay - LangGraph
    caches each one's resolved answer) and the actual tool calls only run
    in the final pass, once every approval is already resolved - so a
    write_file/run_cmd can never get re-executed by a replay.
    """
    tool_calls = state["messages"][-1].tool_calls

    approvals = [
        interrupt({"tool": call["name"], "args": call["args"]})
        if call["name"] in DANGEROUS_TOOLS
        else True
        for call in tool_calls
    ]

    outputs = []
    for call, approved in zip(tool_calls, approvals):
        name, args, call_id = call["name"], call["args"], call["id"]
        if name in DANGEROUS_TOOLS:
            if not approved:
                print_error("[Tool Execution] Denied by human.")
                outputs.append(
                    ToolMessage(content="Tool execution denied by human.", tool_call_id=call_id, name=name)
                )
                continue
        else:
            print_tool_call(f"{name}({args})")

        result = TOOL_FUNCTIONS[name].invoke(args)
        print_tool_result(str(result))
        outputs.append(ToolMessage(content=str(result), tool_call_id=call_id, name=name))

    return {"messages": outputs}


def route_after_model(state: MessagesState) -> str:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END


graph_builder = StateGraph(MessagesState)
graph_builder.add_node("agent", call_model)
graph_builder.add_node("tools", run_tools)
graph_builder.set_entry_point("agent")
graph_builder.add_conditional_edges("agent", route_after_model, {"tools": "tools", END: END})
graph_builder.add_edge("tools", "agent")

graph = graph_builder.compile(checkpointer=InMemorySaver())


def run_agent(user_input: str, config: dict) -> None:
    """Handle one user request: run the graph, resolving every human
    approval interrupt it raises along the way, until it settles on a
    final answer or hits the turn limit."""
    try:
        result = graph.invoke({"messages": [{"role": "user", "content": user_input}]}, config)
        while "__interrupt__" in result:
            payload = result["__interrupt__"][0].value
            allowed = ask_permission(payload["tool"], payload["args"])
            result = graph.invoke(Command(resume=allowed), config)
    except GraphRecursionError:
        print_error("[Agent] Reached max turns without finishing.")
        return

    final = result["messages"][-1]
    if isinstance(final, AIMessage) and final.content:
        print_thought(final.content)


def main():
    tracing = bool(os.environ.get("LANGSMITH_API_KEY")) and os.environ.get(
        "LANGSMITH_TRACING", "true"
    ).lower() not in {"false", "0"}

    console.print(f"[bold]Minimal Coding Agent[/bold] (model: {active_model_label()})")
    console.print(f"Tracing: {'enabled (LangSmith)' if tracing else 'disabled'}")
    console.print("Type your request, or 'exit' to quit.")
    console.print(f"[dim]Try this: {SAMPLE_PROMPT}[/dim]\n")

    config = {"configurable": {"thread_id": "cli-session"}, "recursion_limit": MAX_TURNS * 2 + 2}

    while True:
        user_input = console.input("[bold blue]You: [/bold blue]")
        if user_input.strip().lower() in {"exit", "quit"}:
            break
        run_agent(user_input, config)


if __name__ == "__main__":
    main()

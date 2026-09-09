from rich.console import Console

console = Console()


def print_thought(text: str) -> None:
    """[Agent Thought] - the model's own reasoning/reply text."""
    console.print(f"[cyan][Agent Thought] {text}[/cyan]")


def print_tool_call(text: str) -> None:
    """[Tool Execution] - announcing that a tool is about to run."""
    console.print(f"[yellow][Tool Execution] {text}[/yellow]")


def print_tool_result(text: str) -> None:
    """[Tool Result] - what a tool actually returned."""
    console.print(f"[dim green][Tool Result] {text[:500]}[/dim green]")


def print_error(text: str) -> None:
    """Anything that went wrong or was denied."""
    console.print(f"[bold red]{text}[/bold red]")

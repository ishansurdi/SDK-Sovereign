"""Run one episode with rule-based agents and print a colored transcript."""
from __future__ import annotations

from rich.console import Console
from rich.rule import Rule
from rich.table import Table

from server.environment import SDKSovereignEnvironment
from server.rule_agents import get_rule_agent


ROLE_COLORS = {"auditor": "magenta", "lead": "cyan"}
ACTION_SYMBOLS = {
	"propose_replacement": ">",
	"approve": "OK",
	"reject": "NO",
	"submit_patch": "SUBMIT",
	"pass": ".",
}


def main() -> None:
	"""Run and render a full deterministic demo episode."""
	console = Console()
	console.print(Rule("[bold]SDK-Sovereign - Demo Episode (rule-based agents)"))

	env = SDKSovereignEnvironment(seed=7)
	observation = env.reset()
	console.print(f"\n[dim]Repo:[/] {env.state.repo_id}")
	console.print(f"[dim]Deprecated SDK:[/] {env.state.deprecated_sdk}")
	console.print(f"[dim]Error log:[/] {observation.error_log}\n")

	while not observation.done:
		role = observation.current_role
		action = get_rule_agent(role)(observation)
		symbol = ACTION_SYMBOLS.get(action.action_type, "?")
		color = ROLE_COLORS[role]
		payload = (
			action.proposed_sdk
			or action.rejection_reason
			or ("[patch submitted]" if action.patched_code else "")
		)
		console.print(
			f"[dim]turn {observation.turn_index}[/]  "
			f"[{color}]{role:>7}[/]  {symbol} {action.action_type:<22}  {payload[:60]}"
		)
		observation = env.step(action)

	console.print()
	console.print(Rule("[bold]Outcome"))
	state = env.state
	table = Table()
	table.add_column("Field")
	table.add_column("Value")
	table.add_row("Repo", state.repo_id)
	table.add_row("Approved replacement", str(state.approved_replacement))
	table.add_row("Termination reason", state.terminated_reason)
	table.add_row("Test results", str(state.test_results))
	table.add_row("Auditor reward", f"{state.cumulative_reward_by_role['auditor']:+.2f}")
	table.add_row("Lead reward", f"{state.cumulative_reward_by_role['lead']:+.2f}")
	console.print(table)


if __name__ == "__main__":
	main()

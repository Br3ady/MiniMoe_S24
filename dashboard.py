from rich.live import Live
from rich.table import Table
import time
import random

def generate_table():
    table = Table(title="Client Actions Dashboard")
    table.add_column("Client", justify="right", style="cyan", no_wrap=True)
    table.add_column("Action", style="magenta")
    table.add_column("Time Between Actions", justify="right", style="green")

    for client in range(5):
        action = random.choice(["Login", "Click", "Purchase", "Logout"])
        time_between = f"{random.uniform(0.5, 2.0):.2f} seconds"
        table.add_row(f"Client {client}", action, time_between)
    
    return table

with Live(generate_table(), refresh_per_second=2) as live:
    while True:
        time.sleep(1)
        live.update(generate_table())
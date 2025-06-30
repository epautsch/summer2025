@dataclass
class HistoryManager:
    summarizer: Any
    history: List[str] = field(default_factory=list)
    word_limit: int = 800

    def add(self, entry: str):
        self.history.append(entry)
        
    def get_full(self) -> str:
        full_text = "\n".join(self.history)

        if len(full_text.split()) > self.word_limit:
            console.print("[red][DEBUG SUMMARIZER][/]")
            summary = self.summarizer.generate(full_text)
            console.print("[red][DEBUG SUMMARIZER END][/]\n")
            self.history = [f"History summary: {summary}"]
            return self.history[0]
        return full_text

    def show_history(self):
        table = Table(title="Agent History")
        table.add_column("Step", style="dim", width=6, justify="right")
        table.add_column("Entry")
        for i, entry in enumerate(self.history, 1):
            table.add_row(str(i), entry)
        console.print(table)



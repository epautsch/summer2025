from agents.coder_agent import CoderAgent
from agents.tester_agent import TesterAgent
from agents.analyst_agent import AnalystAgent


class DevPipeline:
    def __init__(self):
        self.analyst = AnalystAgent("Analyst")
        self.coder = CoderAgent("Coder")
        self.tester = TesterAgent("Tester")

    def run(self, task_description: str):
        problem = self.analyst.act(task_description)
        code = self.coder.act(problem)
        test_report = self.tester.act(code)
        return test_report


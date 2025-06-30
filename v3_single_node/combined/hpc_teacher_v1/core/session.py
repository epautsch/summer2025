@dataclass
class Observation:
    result: str

@dataclass
class SessionManager:
    planner: LessonPlanner
    quizzer: Quizzer
    tutor: CodeTutor
    executor: Executor
    history: HistoryManager

    def _call_llm(self, user_prompt: str,
                  schema_hint: str = "Your last response was not valid JSON. Please reply with valid JSON.",
                  max_retries: int = 3) -> dict:

        hist = self.history.get_full()
        prefix = (hist + "\n") if hist else ""

        full_prompt = prefix + user_prompt
        last_err = None

        for attempt in range(1, max_retries + 1):
            raw = self.planner.llm.generate(full_prompt)
            self.history.add(f"UserPrompt (JSON): {user_prompt}")
            self.history.add(f"LLMResponse (raw JSON): {raw}")

            try:
                data = json.loads(strip_markdown_fences(raw))
                return data
            except json.JSONDecodeError as e:
                last_err = e
                console.print(f"[red]Warning:[/] JSON parse failed attempt {attempt}): {e}")
                full_prompt += f"\n{schema_hint}"

        raise last_err

    def _extract_explanation_data(self, raw: str) -> Tuple[str, List[str]]:
        try:
            payload = raw.get("payload", {})
            explanation = payload.get("explanation", raw)
            examples = payload.get("examples", [])
            return explanation, examples
        except Exception:
            return raw, []

    def run(self):
        console.print("[bold green]👋 Welcome to the HPC Tutor![/]")
        
        # topic selection
        topics = self.planner.default_topics
        for idx, topic in enumerate(self.planner.default_topics, start=1):
            console.print(f"{idx}. {topic}")
        choice = Prompt.ask("Choose a topic by number or type a new one")
        try:
            topic = topics[int(choice) - 1]
        except Exception:
            topic = choice
        console.print(f"[red][DEBUG TOPIC CHOICE] You chose {topic}[/].")
        console.print(f"[red][DEBUG TOPIC CHOICE END][/]\n")
        
        # create and set lesson plan
        lesson_plan_prompt = self.planner.create_lesson_plan(topic)
        lesson_plan_json = self._call_llm(lesson_plan_prompt)
            # TODO need better error parsing here for failed conditional
        if lesson_plan_json["action"] == "CREATE_LESSON_PLAN":
            real_topic = lesson_plan_json["payload"]["topic"]
            objectives = lesson_plan_json["payload"]["objectives"]
            self.planner.set_plan(topic, objectives)
            console.print(f"[green] ✅ Saved lesson plan for \"{real_topic}\" with {len(objectives)} objectives.[/]\n")
            action = Action(
                type=ActionType.CREATE_LESSON_PLAN,
                payload={"topic": real_topic, "objectives": objectives}
            )
            obs = self.executor.execute(action)

            self.history.add(f"Observation: {obs.result}")
        else:
            console.print("[red] ✖ Unexpected response—couldn't create lesson plan.[/]")

        # loop through objectives
        for obj in objectives:
            console.print(Rule(f"📖 {obj}"))
            
            # generate explanation
            explanation_prompt = self.planner.explain_concept(obj)
            explanation_raw = self._call_llm(explanation_prompt)
            # parse json for explanation
            explanation, examples = self._extract_explanation_data(explanation_raw)

            action = Action(
                type=ActionType.EXPLAIN_CONCEPT,
                payload={
                    "concept": obj,
                    "explanation": explanation,
                    "examples": examples,
                }
            )
            obs = self.executor.execute(action)
            self.history.add(f"Observation: {obs.result}")

            # ask user for followup explanations before continuing
            while True:
                user_q = Prompt.ask(
                    "\nHave any questions? Type your question, or 'next' to continue"
                ).strip()
                if user_q.lower() in ("next", "n"):
                    break

                answer_prompt = self.planner.answer_question(obj, user_q)
                answer_raw = self._call_llm(answer_prompt)
                answer, examples = self._extract_explanation_data(answer_raw)

                action = Action(
                    type=ActionType.EXPLAIN_CONCEPT,
                    payload={
                        "concept": obj,
                        "explanation": answer,
                        "examples": examples,
                    }
                )
                obs = self.executor.execute(action)
                self.history.add(f"Observation: {obs.result}")
            
            """
            questions = self.quizzer.generate_questions(step)
            for qa in questions:
                user_ans = Prompt.ask(qa['q'])
                if self.quizzer.grade_answer(qa['a'], user_ans):
                    console.print("[green]✔ Correct![/]")
                else:
                    console.print(f"[red]✖ Incorrect.[/] Expected: {qa['a']}")

            skeleton = self.tutor.generate_skeleton(f"{topic}: {step}", backend)
            filename = f"lesson_step_{outline.index(step)+1}.{ext}"
            console.print(Panel(Syntax(skeleton, ext, line_numbers=True), title=f"Code Skeleton → {filename}"))
            save_to_file(skeleton, filename)

            console.print("Paste your completed code below (end with empty line):")
            user_lines = []
            while True:
                line = input()
                if not line.strip(): break
                user_lines.append(line)
            completed = "\n".join(user_lines)
            src_file = f"completed_step_{outline.index(step)+1}.{ext}"
            save_to_file(completed, src_file)

            ok, result = self.tutor.evaluate_submission(src_file, "lesson_exec", expected_output="EXPECTED_OUTPUT")
            if ok:
                console.print("[bold green]🎉 Code ran successfully![/]")
            else:
                console.print(Panel(result, title="Errors / Output"))
            """
        console.print(Rule("🏁 Lesson Complete!"))
        summary = self.planner.llm.generate(f"Summarize the lesson on {topic} using {backend} and key takeaways.")
        console.print(Panel(summary, title="Lesson Summary"))
        hw = self.planner.llm.generate(f"Generate 4 homework exercises for {topic} on {backend}.")
        console.print(Panel(hw, title="Homework"))



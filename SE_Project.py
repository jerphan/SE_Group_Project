
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated

import autogen
from autogen.coding import LocalCommandLineCodeExecutor

OUTPUTS_DIR = Path("outputs")
DIAGRAMS_DIR = OUTPUTS_DIR / "diagrams"
GENERATED_APP_DIR = OUTPUTS_DIR / "generated_app"

for directory in (OUTPUTS_DIR, DIAGRAMS_DIR, GENERATED_APP_DIR):
    directory.mkdir(parents=True, exist_ok=True)


def build_llm_config() -> dict:
    return {
        "config_list": [
            {
                "model": os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
                "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
                "api_key": "ollama"
            }
        ]
    }

# Function to check the replies from the agents for termination keywords
def make_termination_check(*tokens: str):
    """Substring-based termination check (resilient to extra punctuation/words)."""
    upper_tokens = tuple(t.upper() for t in tokens)

    def _is_termination(msg: dict) -> bool:
        content = (msg.get("content") or "").upper()
        return any(token in content for token in upper_tokens)

    return _is_termination

# Summary of the application to build
APP_SUMMARY = """
An application that allows the user to create and save a meal plan that includes what they would like their daily goals for calories, protein (in grams), carbs (in grams), fiber (in grams), and fat (in grams). 
Each meal plan should also have an optional section for dietary restrictions provided by the user. 
The application should also allow the user to create and save a grocery list by asking them to give the list a name, and then adding items to the list until the user selects that they are done. 
The application should be implemented using a command line interface, where the main menu includes options to create a meal plan, view their saved meal plans, create a grocery list, view their saved grocery lists, or quit. 
For each of the options there should be a menu that allows the user to return the main menu.
Keep the application simple.
""".strip()


# Function to ensure thatt we are getting atleast 4-5 use cases from the diagams outputted
def validate_use_case_count(
    use_cases: Annotated[list[str], "List of use case names extracted from the summary"],
) -> str:
    """Validate that the system defines between 4 and 5 use cases."""
    count = len(use_cases)
    valid = 4 <= count <= 5
    return json.dumps(
        {
            "valid": valid,
            "count": count,
            "expected_range": "4-5",
            "message": (
                "Use case count meets required criteria."
                if valid
                else f"Found {count} use cases; adjust to 4 or 5."
            ),
        }
    )

# Function to save the JSON files in the output files
def save_json_output(
    filename: Annotated[str, "Target JSON filename inside OUTPUTS/ (e.g., use_cases.json)"],
    json_content: Annotated[str, "Valid JSON string to persist to the shared blackboard"],
) -> str:
    """Validate JSON syntax and save an intermediate artifact to the blackboard."""
    parsed = json.loads(json_content)
    safe_name = Path(filename).name
    target = OUTPUTS_DIR / safe_name
    target.write_text(json.dumps(parsed, indent=2), encoding="utf-8")
    return f"Saved valid JSON artifact to {target}"


# Function to save the outputed diagrams to the folder for outputs
def save_diagram_source(
    filename: Annotated[str, "Diagram filename (e.g., use_case_diagram.puml)"],
    diagram_code: Annotated[str, "PlantUML source text for the diagram"],
) -> str:
    """Persist diagram source text so it can be rendered to PNG/SVG."""
    safe_name = Path(filename).name
    target = DIAGRAMS_DIR / safe_name
    target.write_text(diagram_code.strip() + "\n", encoding="utf-8")
    return f"Saved diagram source to {target}"

# Creates a use case diagram based on a summary provided
def create_use_case_diagram(llm_config: dict, summary: str) -> None:
    print("\n=== STAGE 1: Use Case Diagram (Tool-Based Function Calling) ===")

    use_case_agent = autogen.AssistantAgent(
        name="UseCaseAgent",
        llm_config=llm_config,
        system_message=(
            "You are a Software Requirements Engineer. Given an application summary:\n"
            "1. Identify 4 to 5 clear use cases and their actors.\n"
            "2. Call `validate_use_case_count` to verify the count is 4-5.\n"
            "3. Call `save_json_output` to save `use_cases.json` containing "
            '{"actors": [...], "use_cases": [{"id": "UC1", "name": "...", "actor": "..."}]}.\n'
            "4. Call `save_diagram_source` to save a PlantUML use case diagram to "
            "`use_case_diagram.puml`.\n"
            "5. After all tool calls succeed, reply with TERMINATE."
        ),
    )

    # Tool executor has code_execution_config=False so it ONLY runs registered tools
    tool_executor = autogen.UserProxyAgent(
        name="ToolExecutor",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=8,
        code_execution_config=False,
        is_termination_msg=make_termination_check("TERMINATE"),
    )

    for fn, desc in (
        (validate_use_case_count, "Check that 4 to 5 use cases are defined."),
        (save_json_output, "Validate and save an intermediate JSON file."),
        (save_diagram_source, "Save PlantUML diagram text to folder."),
    ):
        autogen.register_function(
            fn,
            caller=use_case_agent,
            executor=tool_executor,
            name=fn.__name__,
            description=desc,
        )

    tool_executor.initiate_chat(
        use_case_agent,
        message=f"Analyze this application summary and produce the use case diagram:\n\n{summary}",
    )


# Function to use multiple agents to create specifications for the use case diagram, a class diagram, and sequence diagram
def design_diagrams(llm_config: dict) -> None:
    print("\n=== STAGES 2-4: Specs, Class & Sequence Diagrams (Multi-Agent Collaboration) ===")

    use_cases_text = (
        (OUTPUTS_DIR / "use_cases.json").read_text(encoding="utf-8")
        if (OUTPUTS_DIR / "use_cases.json").exists()
        else "{}"
    )

    spec_agent = autogen.AssistantAgent(
        name="UseCaseSpecAgent",
        llm_config=llm_config,
        system_message=(
            "You are a Requirements Specifier. For EVERY use case in `use_cases.json`, "
            "write a complete specification (ID, name, actor, preconditions, main flow, "
            "alternate flows, postconditions). Save the list to `use_case_specs.json` "
            "using `save_json_output`. Ensure spec count >= use case count."
        ),
    )

    class_architect = autogen.AssistantAgent(
        name="ClassArchitectAgent",
        llm_config=llm_config,
        system_message=(
            "You are an Object-Oriented System Architect. Read the use case "
            "specifications from the conversation and design a clean Class Diagram. "
            "Save structured classes (names, attributes, methods, relationships) to "
            "`class_diagram.json` via `save_json_output`, and save PlantUML to "
            "`class_diagram.puml` via `save_diagram_source`."
        ),
    )

    sequence_designer = autogen.AssistantAgent(
        name="SequenceDesignerAgent",
        llm_config=llm_config,
        system_message=(
            "You are a UML Sequence Diagram Designer. For EVERY use case specification, "
            "design a sequence diagram using ONLY classes and method names defined by "
            "ClassArchitectAgent. Save structured flows to `sequence_diagrams.json` via "
            "`save_json_output` and save each PlantUML diagram (`seq_UC1.puml`, etc.) "
            "via `save_diagram_source`. End with READY_FOR_REVIEW when done."
        ),
    )

    coordinator = autogen.UserProxyAgent(
        name="DesignCoordinator",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=12,
        code_execution_config=False,
        is_termination_msg=make_termination_check("READY_FOR_REVIEW"),
    )

    # Give each collaborating agent access to the blackboard save tools
    for agent in (spec_agent, class_architect, sequence_designer):
        autogen.register_function(
            save_json_output,
            caller=agent,
            executor=coordinator,
            name="save_json_output",
            description="Validate and save an intermediate JSON file.",
        )
        autogen.register_function(
            save_diagram_source,
            caller=agent,
            executor=coordinator,
            name="save_diagram_source",
            description="Save PlantUML diagram text to disk.",
        )

    group_chat = autogen.GroupChat(
        agents=[coordinator, spec_agent, class_architect, sequence_designer],
        messages=[],
        max_round=16,
        speaker_selection_method="round_robin",
    )
    manager = autogen.GroupChatManager(groupchat=group_chat, llm_config=llm_config)

    coordinator.initiate_chat(
        manager,
        message=(
            "Collaborate to produce: (1) use case specifications for every use case, "
            "(2) a consistent class diagram, and (3) one sequence diagram per use case.\n\n"
            f"Current use_cases.json:\n{use_cases_text}"
        ),
    )


# Function to run a reflection loop, and ensure that all out the outputs meet the required specifications

def run_reflection_loop(llm_config: dict) -> None:
    """Observer/Reflector evaluates design OUTPUTS and triggers fixes."""
    print("\n=== REFLECTION LOOP: Observer / Reflect-Refine Quality Gate ===")

    snapshot = {}
    for fname in ("use_cases.json", "use_case_specs.json", "class_diagram.json", "sequence_diagrams.json"):
        path = OUTPUTS_DIR / fname
        snapshot[fname] = path.read_text(encoding="utf-8") if path.exists() else "MISSING"

    reflector_agent = autogen.AssistantAgent(
        name="ReflectorAgent",
        llm_config=llm_config,
        system_message=(
            "You are an Observer and Quality Reflector Agent. Inspect the design "
            "OUTPUTS against this rubric:\n"
            "1. Number of use case specs >= number of use cases (4-5).\n"
            "2. Every spec includes preconditions, main flow, and alternate flows.\n"
            "3. Number of sequence diagrams >= number of use cases.\n"
            "4. Every method called in a sequence diagram exists in class_diagram.json.\n"
            "If ANY check fails, list exact defects for RefinerAgent to fix. "
            "If ALL checks pass, reply with APPROVED."
        ),
    )

    refiner_agent = autogen.AssistantAgent(
        name="RefinerAgent",
        llm_config=llm_config,
        system_message=(
            "You are a Design Refiner Agent. When ReflectorAgent reports defects, "
            "fix the affected JSON or PlantUML files using `save_json_output` or "
            "`save_diagram_source` and summarize what you corrected."
        ),
    )

    reflection_runner = autogen.UserProxyAgent(
        name="ReflectionRunner",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=6,
        code_execution_config=False,
        is_termination_msg=make_termination_check("APPROVED"),
    )

    autogen.register_function(
        save_json_output,
        caller=refiner_agent,
        executor=reflection_runner,
        name="save_json_output",
        description="Save corrected JSON OUTPUTS to the blackboard.",
    )
    autogen.register_function(
        save_diagram_source,
        caller=refiner_agent,
        executor=reflection_runner,
        name="save_diagram_source",
        description="Save corrected PlantUML diagrams.",
    )

    review_chat = autogen.GroupChat(
        agents=[reflection_runner, reflector_agent, refiner_agent],
        messages=[],
        max_round=8,
        speaker_selection_method="round_robin",
    )
    review_manager = autogen.GroupChatManager(groupchat=review_chat, llm_config=llm_config)

    reflection_runner.initiate_chat(
        review_manager,
        message=(
            "Review these OUTPUTS for completeness and cross-diagram consistency. "
            "Critique any issues so RefinerAgent can fix them, or reply APPROVED.\n\n"
            f"{json.dumps(snapshot, indent=2)}"
        ),
    )


# Function to then generate code and then attempt to execute it to ensure that it is executable

def generate_code(llm_config: dict) -> None:
    print("\n=== STAGE 5: Code Generation & Verification (Coding Agent) ===")

    class_json = (
        (OUTPUTS_DIR / "class_diagram.json").read_text(encoding="utf-8")
        if (OUTPUTS_DIR / "class_diagram.json").exists()
        else "{}"
    )
    seq_json = (
        (OUTPUTS_DIR / "sequence_diagrams.json").read_text(encoding="utf-8")
        if (OUTPUTS_DIR / "sequence_diagrams.json").exists()
        else "{}"
    )

    python_coder = autogen.AssistantAgent(
        name="PythonCoderAgent",
        llm_config=llm_config,
        system_message=(
            "You are a Senior Python Developer. Using `class_diagram.json` and "
            "`sequence_diagrams.json`, write:\n"
            "1. `app.py` implementing all classes and sequence diagram workflows, "
            "plus a CLI menu guarded by `if __name__ == '__main__':`.\n"
            "2. `test_scenarios.py` that imports `app` and runs non-interactive "
            "assertions verifying every use case scenario (do NOT call `input()` "
            "or `serve_forever()` during tests).\n"
            "Output executable ```python blocks using `# filename: app.py` and "
            "`# filename: test_scenarios.py` so the executor saves and runs them. "
            "Once all tests pass with exit code 0, reply with TERMINATE."
            "You must name your files app.py and test_scenarios.py."
        ),
    )

    # Use LocalCommandLineCodeExecutor with an explicit timeout
    code_executor = autogen.UserProxyAgent(
        name="CodeExecutorAgent",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=6,
        code_execution_config={
            "executor": LocalCommandLineCodeExecutor(
                timeout=120,
                work_dir=str(GENERATED_APP_DIR),
            )
        },
        is_termination_msg=make_termination_check("TERMINATE"),
    )

    code_executor.initiate_chat(
        python_coder,
        message=(
            "Generate the executable Python application (`app.py`) and automated "
            "scenario tests (`test_scenarios.py`) from these verified designs.\n\n"
            f"class_diagram.json:\n{class_json}\n\n"
            f"sequence_diagrams.json:\n{seq_json}"
        ),
    )


def main() -> None:
    llm_config = build_llm_config()

    create_use_case_diagram(llm_config, APP_SUMMARY)
    design_diagrams(llm_config)
    run_reflection_loop(llm_config)
    generate_code(llm_config)

    print(
        "\nPipeline complete."
    )


if __name__ == "__main__":
    main()
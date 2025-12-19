import io
import random
from typing import List, Dict
import csv
import requests
import streamlit as st
from dotenv import load_dotenv
import os, re, shlex, subprocess, sys, tempfile, zipfile, json, pathlib
import re, html, requests
from bs4 import BeautifulSoup
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_experimental.utilities import PythonREPL
from langchain.agents import Tool, AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

st.set_page_config(page_title="Agentic Chat + Code Runner")

GROQ_API_KEY='APIKEY'

TIMEOUT_SECS = 150
ENTRY_CANDIDATES = ["main.py", "app.py", "__main__.py", "run.py"]
ERROR_HINTS = [
(r"ModuleNotFoundError: No module named '([^']+)'",
lambda m: f"Install missing package `{m.group(1)}` (add to requirements.txt) or fix import path."),
(r"ImportError: cannot import name '([^']+)' from '([^']+)'",
lambda m: f"Export `{m.group(1)}` from `{m.group(2)}` or correct the import statement."),
(r"SyntaxError: (.*)\n.*File \".*\", line (\d+)",
lambda m: f"Syntax error on line {m.group(2)}: {m.group(1)}. Correct the syntax."),
(r"IndentationError: (.+)",
lambda m: f"Indentation error: {m.group(1)}. Fix inconsistent spaces/tabs."),
(r"NameError: name '([^']+)' is not defined",
lambda m: f"Define `{m.group(1)}` before use or correct the identifier."),
(r"TypeError: (.+) takes (\d+) positional arguments but (\d+) were given",
lambda m: "Called a function with the wrong number of arguments. Check its signature and calls."),
(r"FileNotFoundError: \[Errno 2\] No such file or directory: '([^']+)'",
lambda m: f"Ensure file `{m.group(1)}` exists at runtime or use a correct path."),
]
# Load environment

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error("🚨 Please set your GROQ_API_KEY as an environment variable before running.")
    st.stop()


# Initialize LLM
if not api_key:
    st.error("Groq API key is missing.")
    st.stop()


llm = ChatGroq(
    groq_api_key=api_key,
    model="llama3-70b-8192",  
    temperature=0.7,
    max_tokens=1024,
)

# Define Tools

search_tool = DuckDuckGoSearchResults(
    name="web_search",
    description="Search the web for information."
)

repl = PythonREPL()

def safe_python(code: str) -> str:

    try:
        return repl.run(code)
    except Exception as e:
        return f"Python execution error: {e}"

repl_tool = Tool(
    name="python_repl",
    description="Execute Python code and return the result. Input must be valid Python.",
    func=safe_python,
)


tools = [search_tool, repl_tool]

def run_cmd(cmd, cwd: pathlib.Path, timeout: int = TIMEOUT_SECS):
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        return 124, "", f"Timeout: process exceeded {timeout}s"
    except Exception as e:
        return 125, "", f"Runner error: {e}"


def maybe_install_requirements(base: pathlib.Path):
    req = base / "requirements.txt"
    if req.exists():
        return run_cmd([sys.executable, "-m", "pip", "install", "-r", str(req), "--no-input"], base)
    return 0, "", ""


def find_entrypoint(base: pathlib.Path):

    for cand in ENTRY_CANDIDATES:
        p = base / cand
        if p.exists():
            return ["python", str(p)]
 
    for p in base.rglob("*.py"):
        try:
            if "__name__ == '__main__'" in p.read_text(errors="ignore"):
                return ["python", str(p)]
        except Exception:
            pass
    roots = [p for p in base.iterdir() if p.suffix == ".py"]
    if len(roots) == 1:
        return ["python", str(roots[0])]
    return None


def parse_hints(stderr: str):
    hints = []
    for pat, make in ERROR_HINTS:
        m = re.search(pat, stderr, re.MULTILINE)
        if m:
            hints.append(make(m))
    if not hints and stderr.strip():
        last = stderr.strip().splitlines()[-1]
        hints.append(f"Last error: {last}. Inspect the stack trace for the failing line.")
    return hints
#Quiz generation 
def _quiz_schema_instruction() -> str:
    return (
        "Return ONLY valid JSON with this exact schema:\n"
        "{\n"
        '  "questions": [\n'
        "    {\n"
        '      "id": "string",\n'
        '      "type": "MCQ" | "TRUE_FALSE" | "SHORT",\n'
        '      "question": "string",\n'
        '      "options": ["string",], // required for MCQ (3-6 options)\n'
        '      "correct_answer": "string" | true | false,\n'
        '      "explanation": "string"\n'
        "    }\n"
        "  ]\n"
        "}\n"
    )

def build_quiz_prompt(
    topic: str,
    count: int = 5,
    difficulty: str = "medium",
    level: str = "general adult",
    qtype: str = "MCQ",
    include_explanations: bool = True,
) -> List[Dict]:
    system = (
        "You are a meticulous exam item writer. "
        "Questions must be unambiguous, self-contained, and aligned to the requested difficulty. "
        "No copyrighted or private test content. "
        "Use plausible distractors that test the concept (not trick wording). "
        "For MCQ, exactly 4 options unless specified otherwise. "
        + _quiz_schema_instruction()
    )
    human = (
        f"Create {count} {difficulty} {qtype} questions about '{topic}' "
        f"for learners at the '{level}' level. "
        f"Include concise explanations: {include_explanations}. "
        "Avoid requiring external images. Keep each question under 280 characters; explanations under 300."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": human},
    ]

def generate_questions_with_llm(
    llm,
    topic: str,
    count: int,
    difficulty: str,
    level: str,
    qtype: str,
    include_explanations: bool,
    seed: int | None = None,
    retries: int = 2,
) -> Dict:
    if seed is not None:
        random.seed(seed)
    messages = build_quiz_prompt(topic, count, difficulty, level, qtype, include_explanations)

    last_err = None
    for _ in range(retries + 1):
        try:
            resp = llm.invoke(messages) 
            raw = resp.content if hasattr(resp, "content") else str(resp)
            raw = raw.strip()
            if raw.startswith("```"):
                parts = raw.split("```")
                if len(parts) >= 3:
                    raw = parts[1]
                    raw = raw.split("\n", 1)[1] if "\n" in raw else raw

            data = json.loads(raw)

            if "questions" not in data or not isinstance(data["questions"], list):
                raise ValueError("Missing 'questions' array.")

            cleaned = []
            for i, q in enumerate(data["questions"], start=1):
                qt = str(q.get("type", qtype)).upper()
                qid = q.get("id") or f"q{i:03d}"
                question = (q.get("question") or "").strip()
                expl = (q.get("explanation") or "").strip()

                if qt == "MCQ":
                    options = q.get("options") or []
                    if len(options) < 4:
                        needed = 4 - len(options)
                        options += [f"Option {len(options)+k+1}" for k in range(needed)]
                    options = options[:6]
                    correct = q.get("correct_answer")
                    if isinstance(correct, str) and correct not in options:
                        options.insert(random.randint(0, min(3, len(options))), correct)
                    cleaned.append({
                        "id": qid, "type": "MCQ", "question": question,
                        "options": options[:4], "correct_answer": correct, "explanation": expl
                    })

                elif qt in ("TRUE_FALSE", "TRUEFALSE", "TF"):
                    correct = q.get("correct_answer")
                    if isinstance(correct, str):
                        correct = correct.strip().lower() in ("true", "t", "yes")
                    cleaned.append({
                        "id": qid, "type": "TRUE_FALSE", "question": question,
                        "options": ["True", "False"], "correct_answer": bool(correct), "explanation": expl
                    })

                else:
                    cleaned.append({
                        "id": qid, "type": "SHORT", "question": question,
                        "options": [], "correct_answer": q.get("correct_answer", ""), "explanation": expl
                    })

            return {"questions": cleaned[:count]}
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(f"Failed to parse LLM output as JSON: {last_err}")

def questions_to_csv(questions: List[Dict]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["id","type","question","option_a","option_b","option_c","option_d","correct_answer","explanation"])
    for q in questions:
        opts = q.get("options", [])
        row = [
            q.get("id",""),
            q.get("type",""),
            q.get("question",""),
            opts[0] if len(opts)>0 else "",
            opts[1] if len(opts)>1 else "",
            opts[2] if len(opts)>2 else "",
            opts[3] if len(opts)>3 else "",
            q.get("correct_answer",""),
            q.get("explanation",""),
        ]
        writer.writerow(row)
    return buf.getvalue()

# Define Prompt

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful AI agent. "
     "Your name is Helper"
     "Use tools ONLY if necessary. "
     "When calling a tool, provide valid and minimal input. "
     "If unsure, answer directly instead of forcing tool use."
     "In the beginning of the response, begin with Tool (if duckduckgo was utilized) or Groq (if the LLM was utilized)"
     ),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

tools = [search_tool, repl_tool]
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools,prompt=prompt, verbose=True, handle_parsing_errors=True)

# Create Agent

agent = create_tool_calling_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)


st.set_page_config(page_title="Agentic Chat + Code Runner", page_icon="🤖")
st.sidebar.write("Groq key: **OK** ")

tab_chat, tab_run, tab_quiz = st.tabs([" Agentic Chat", "Run Code", "Quiz"])

#Chat tab
with tab_chat:
    st.title("Agentic Chat")

 
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [] 


    for role, msg in st.session_state.chat_history:
        with st.chat_message("user" if role == "user" else "assistant"):
            st.markdown(msg)

    if user_msg := st.chat_input("Type your message…"):
        # add user msg
        st.session_state.chat_history.append(("user", user_msg))
        with st.chat_message("user"):
            st.markdown(user_msg)

        try:
            resp = agent_executor.invoke({
                "input": user_msg,
                "chat_history": st.session_state.chat_history
            })
            reply = resp.get("output", "(no response)")
        except Exception as e:
            reply = f"Error: {e}"
        st.session_state.chat_history.append(("assistant", reply))
        with st.chat_message("assistant"):
            st.markdown(reply)


# Runner tab 
with tab_run:
    st.title("Check code with the LLM")
    st.caption("Upload a .py or .zip project. I’ll read it, run it, and show errors & outputs. LLM tips optional.")
    uploaded_file = st.file_uploader("Upload (.py or .zip)", type=["py", "zip"])

    if uploaded_file:
        with tempfile.TemporaryDirectory() as td:
            base = pathlib.Path(td)

            if uploaded_file.name.endswith(".py"):
                target = base / uploaded_file.name
                target.write_bytes(uploaded_file.read())
                (base / "main.py").write_bytes(target.read_bytes())
            else:
                zpath = base / uploaded_file.name
                zpath.write_bytes(uploaded_file.read())
                with zipfile.ZipFile(zpath, "r") as zf:
                    zf.extractall(base)

            with st.expander("Project tree"):
                for p in sorted(base.rglob("*")):
                    st.write(str(p.relative_to(base)))

            root_pys = sorted([p for p in base.iterdir() if p.suffix == ".py"])
            if root_pys:
                choice = st.selectbox("Choose entry file (used if auto-detect fails):",
                                      [p.name for p in root_pys], index=0)
                with st.expander(f"Preview: {choice}"):
                    st.code((base / choice).read_text(errors="ignore"), language="python")

            with st.status("Installing requirements (if any)…", expanded=False):
                rc, out, err = maybe_install_requirements(base)
                if rc != 0:
                    st.warning("pip install had errors (continuing).")
                    if err:
                        st.code(err, language="bash")

            entry = find_entrypoint(base)
            if not entry and root_pys:
                entry = ["python", str(base / root_pys[0].name)]

            if not entry:
                st.error("No entrypoint found. Add main.py or __main__.py, or place a single .py at root.")
            else:
                st.info(f"Running: `{' '.join(shlex.quote(c) for c in entry)}` (timeout {TIMEOUT_SECS}s)")
                rc, out, err = run_cmd(entry, base)
                ok = (rc == 0)

                c1, c2 = st.columns(2)
                with c1:
                    st.subheader("stdout"); st.code(out or "(empty)")
                with c2:
                    st.subheader("stderr"); st.code(err or "(empty)")

                comments = []
                if rc == 124:
                    comments.append({"severity": "error",
                                     "message": f"Program timed out after {TIMEOUT_SECS}s. Check for infinite loops/blocking I/O."})
                elif not ok:
                    comments += [{"severity": "error", "message": h} for h in parse_hints(err)]
                else:
                    comments.append({"severity": "info",
                                     "message": "Program executed successfully. Add tests to lock behavior."})

                st.subheader("Comments")
                for c in comments:
                    badge = "BAD" if c["severity"] == "error" else "GOOD"
                    st.markdown(f"{badge} **{c['severity'].upper()}** — {c['message']}")

                # Always offer the run artifact download
                artifact = {
                    "ok": ok,
                    "command": " ".join(shlex.quote(c) for c in entry),
                    "exit_code": rc,
                    "comments": comments,
                }
                st.download_button(
                    "Download result.json",
                    data=json.dumps(artifact, indent=2).encode("utf-8"),
                    file_name="result.json",
                    mime="application/json",
                    key="runner_result_json",
                )

                # === LLM CODE REVIEW (properly scoped) ===
                st.subheader("LLM review & suggestions")

                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    include_stdout = st.checkbox("Include stdout", value=False, key="llm_inc_stdout")
                with col_b:
                    include_stderr = st.checkbox("Include stderr", value=True, key="llm_inc_stderr")
                with col_c:
                    max_files = st.number_input("Max files", 1, 25, 8, key="llm_max_files")

                def _read_file(p: pathlib.Path, limit_bytes: int = 60_000) -> str:
                    try:
                        data = p.read_text(errors="ignore")
                        if len(data.encode("utf-8")) > limit_bytes:
                            data = data[:limit_bytes] + "\n# [truncated]\n"
                        return data
                    except Exception as e:
                        return f"# [error reading file: {e}]"

                # Build a small project tree
                tree_lines = []
                for p in sorted(base.rglob("*")):
                    rel = str(p.relative_to(base))
                    if p.is_dir():
                        continue
                    tree_lines.append(rel)
                tree_text = "\n".join(tree_lines[:200])  # cap tree length

                # Choose files to review
                py_files = sorted([p for p in base.rglob("*.py")], key=lambda x: len(str(x)))
                meta_files = [p for p in [base/"requirements.txt", base/"README.md"] if p.exists()]
                review_files = (meta_files + py_files)[: int(max_files)]

                if st.button("Ask LLM for code review", key="btn_llm_review"):
                    # Build review payload
                    parts = []
                    for fp in review_files:
                        lang = "python" if fp.suffix == ".py" else ""
                        parts.append(f"### {fp.relative_to(base)}\n```{lang}\n{_read_file(fp)}\n```")

                    run_info = [
                        f"Command: {' '.join(shlex.quote(c) for c in entry)}",
                        f"Exit code: {rc}",
                    ]
                    if include_stdout:
                        run_info.append(f"\nSTDOUT:\n{out or '(empty)'}")
                    if include_stderr:
                        run_info.append(f"\nSTDERR:\n{err or '(empty)'}")
                    run_info_text = "\n".join(run_info)

                    messages = [
                        {
                            "role": "system",
                            "content": (
                                "You are a senior Python reviewer. Provide clear, actionable suggestions.\n"
                                "Focus on correctness, exceptions, security, performance, readability, and Streamlit UX.\n"
                                "If you propose code, keep patches small and self-contained."
                            )
                        },
                        {
                            "role": "user",
                            "content": (
                                f"Project tree (truncated):\n{tree_text}\n\n"
                                f"Run info:\n{run_info_text}\n\n"
                                "Review these files (truncated as needed). "
                                "Return markdown with:\n"
                                "1) Top issues (bullet list)\n"
                                "2) Specific fixes with code snippets\n"
                                "3) Any dependency or packaging problems you notice\n\n"
                                + "\n\n".join(parts)
                            )
                        },
                    ]

                    with st.spinner("LLM is reviewing your code…"):
                        try:
                            resp = llm.invoke(messages)
                            review_md = getattr(resp, "content", str(resp))
                            st.markdown(review_md)
                            st.download_button(
                                "⬇️ Download LLM review",
                                data=review_md.encode("utf-8"),
                                file_name="llm_code_review.md",
                                mime="text/markdown",
                                key="dl_llm_review",
                            )
                        except Exception as e:
                            st.error(f"LLM review failed: {e}")

#WEATHER TOOL 
def get_weather(city: str) -> str:
    """Fetch current weather using Open-Meteo API (free, no key required)."""
    try:
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1}
        ).json()

        if "results" not in geo or len(geo["results"]) == 0:
            return f"❌ Could not find location: {city}"

        lat = geo["results"][0]["latitude"]
        lon = geo["results"][0]["longitude"]

        weather = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={"latitude": lat, "longitude": lon, "current_weather": True}
        ).json()

        if "current_weather" not in weather:
            return f"Weather data not available for {city}"

        temp = weather["current_weather"]["temperature"]
        wind = weather["current_weather"]["windspeed"]

        return f"{temp}°C, 🌬️ {wind} km/h wind in {city}"

    except Exception as e:
        return f"Weather API error: {e}"
#Quiz questions
def _quiz_schema_instruction() -> str:
    return (
        "Return ONLY valid JSON with this exact schema:\n"
        "{\n"
        '  "questions": [\n'
        "    {\n"
        '      "id": "string",\n'
        '      "type": "MCQ" | "TRUE_FALSE" | "SHORT",\n'
        '      "question": "string",\n'
        '      "options": ["string", ...],         // required for MCQ (3-6 options)\n'
        '      "correct_answer": "string" | true | false,\n'
        '      "explanation": "string"\n'
        "    }\n"
        "  ]\n"
        "}\n"
    )

def build_quiz_prompt(
    topic: str,
    count: int = 5,
    difficulty: str = "medium",
    level: str = "general adult",
    qtype: str = "MCQ",
    include_explanations: bool = True,  
    ) -> List[Dict]:
    
    system = (
        "You are a meticulous exam item writer. "
        "Questions must be unambiguous, self-contained, and aligned to the requested difficulty. "
        "No copyrighted or private test content. "
        "Use plausible distractors that test the concept (not trick wording). "
        "For MCQ, exactly 4 options unless specified otherwise. "
        "“Return ONLY valid JSON. No code fences, no prose, no comments.” "
        + _quiz_schema_instruction()
    )
    human = (
        f"Create {count} {difficulty} {qtype} questions about '{topic}' "
        f"for learners at the '{level}' level. "
        f"Include concise explanations: {include_explanations}. "
        "Avoid requiring external images. Keep each question under 280 characters; explanations under 300."
    )
    # LangChain ChatGroq expects a list of messages (role/content)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": human},
    ]

def generate_questions_with_llm(
    llm,
    topic: str,
    count: int,
    difficulty: str,
    level: str,
    qtype: str,
    include_explanations: bool,
    seed: int | None = None,
    retries: int = 2,
) -> Dict:
    if seed is not None:
        random.seed(seed)
    messages = build_quiz_prompt(topic, count, difficulty, level, qtype, include_explanations)

    last_err = None
    for _ in range(retries + 1):
        try:
            # ChatGroq via LangChain: .invoke takes a list of dicts with role/content
            resp = llm.invoke(messages)  # returns a BaseMessage-like obj with .content
            raw = resp.content if hasattr(resp, "content") else str(resp)
            # Some models wrap JSON in fences; strip code fences if present
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("```", 2)[1]
                # if it contains a language tag like ```json
                raw = raw.split("\n", 1)[1] if "\n" in raw else raw
            data = json.loads(raw)
            # basic validation
            if "questions" not in data or not isinstance(data["questions"], list):
                raise ValueError("Missing 'questions' array.")
            cleaned = []
            for i, q in enumerate(data["questions"], start=1):
                qt = q.get("type", qtype).upper()
                qid = q.get("id") or f"q{i:03d}"
                question = q.get("question", "").strip()
                expl = (q.get("explanation") or "").strip()
                if qt == "MCQ":
                    options = q.get("options") or []
                    # enforce 4 options
                    if len(options) < 4:
                        # pad unique placeholders if needed
                        needed = 4 - len(options)
                        options += [f"Option {len(options)+k+1}" for k in range(needed)]
                    options = options[:6]
                    correct = q.get("correct_answer")
                    # if correct not in options, try to insert at random
                    if isinstance(correct, str) and correct not in options:
                        options.insert(random.randint(0, min(3, len(options))), correct)
                    cleaned.append({
                        "id": qid, "type": "MCQ", "question": question,
                        "options": options[:4], "correct_answer": correct, "explanation": expl
                    })
                elif qt in ("TRUE_FALSE", "TRUEFALSE", "TF"):
                    correct = q.get("correct_answer")
                    if isinstance(correct, str):
                        correct = correct.strip().lower() in ("true", "t", "yes")
                    cleaned.append({
                        "id": qid, "type": "TRUE_FALSE", "question": question,
                        "options": ["True", "False"], "correct_answer": bool(correct), "explanation": expl
                    })
                else:
                    cleaned.append({
                        "id": qid, "type": "SHORT", "question": question,
                        "options": [], "correct_answer": q.get("correct_answer", ""), "explanation": expl
                    })
            return {"questions": cleaned[:count]}
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Failed to parse LLM output as JSON: {last_err}")

def questions_to_csv(questions: List[Dict]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["id","type","question","option_a","option_b","option_c","option_d","correct_answer","explanation"])
    for q in questions:
        opts = q.get("options", [])
        row = [
            q.get("id",""),
            q.get("type",""),
            q.get("question",""),
            opts[0] if len(opts)>0 else "",
            opts[1] if len(opts)>1 else "",
            opts[2] if len(opts)>2 else "",
            opts[3] if len(opts)>3 else "",
            q.get("correct_answer",""),
            q.get("explanation",""),
        ]
        writer.writerow(row)
    return buf.getvalue()


with tab_quiz:
    st.title("Generate Quiz ")

    #init state
    if "quiz_qs" not in st.session_state:
        st.session_state.quiz_qs = None
    if "quiz_result" not in st.session_state:
        st.session_state.quiz_result = None
    if "quiz_checked" not in st.session_state:
        st.session_state.quiz_checked = False

    # Controls to generate the quiz
    c1, c2 = st.columns(2)
    with c1:
        topic = st.text_input("Topic", value="Python conditionals")
        count = st.number_input("Number of questions", 1, 50, 6)
        difficulty = st.selectbox("Difficulty", ["easy", "medium", "hard"], index=1)
    with c2:
        level = st.text_input("Learner level", value="general adult")
        qtype = st.selectbox("Question type", ["MCQ", "TRUE_FALSE", "SHORT"], index=0)
        include_expl = st.checkbox("Include explanations", value=True)

    colS1, colS2 = st.columns([1, 1])
    with colS1:
        seeded = st.checkbox("Use seed (reproducible)", value=False)
    with colS2:
        seed = st.number_input("Seed", min_value=0, max_value=999999, value=42, step=1)

    gen_col, reset_col = st.columns([1, 1])
    with gen_col:
        if st.button("Generate", key="quiz_generate"):
            if not topic.strip():
                st.warning("Please enter a topic.")
            else:
                # clear old answers & flags
                for k in list(st.session_state.keys()):
                    if k.startswith("q") and k.endswith("_ans"):
                        del st.session_state[k]
                st.session_state.quiz_checked = False

                with st.spinner("Asking the LLM to create questions…"):
                    try:
                        result = generate_questions_with_llm(
                            llm=llm,
                            topic=topic.strip(),
                            count=int(count),
                            difficulty=difficulty,
                            level=level.strip(),
                            qtype=qtype,
                            include_explanations=include_expl,
                            seed=int(seed) if seeded else None,
                        )
                        qs = result.get("questions", [])
                        if not qs:
                            st.warning("No questions were generated.")
                            st.session_state.quiz_qs = None
                            st.session_state.quiz_result = None
                        else:
                            st.session_state.quiz_qs = qs
                            st.session_state.quiz_result = result
                    except Exception as e:
                        st.error(f"Generation failed: {e}")

    with reset_col:
        if st.button("New quiz", key="quiz_reset"):
            st.session_state.quiz_qs = None
            st.session_state.quiz_result = None
            st.session_state.quiz_checked = False
            for k in list(st.session_state.keys()):
                if k.startswith("q") and k.endswith("_ans"):
                    del st.session_state[k]

    #Quiz format
    qs = st.session_state.quiz_qs
    if qs:
        for i, q in enumerate(qs, start=1):
            st.markdown(f"**Q{i}. {q['question']}**")
            qtype_here = (q.get("type") or "MCQ").upper()
            key = f"q{i}_ans"

            if qtype_here == "MCQ":
                opts = q.get("options", [])
                labels = [f"{chr(65+j)}. {opt}" for j, opt in enumerate(opts)]
                st.radio(" ", labels, index=None, key=key, label_visibility="collapsed")

            elif qtype_here in ("TRUE_FALSE", "TRUEFALSE", "TF"):
                st.radio(" ", ["True", "False"], index=None, key=key, label_visibility="collapsed")

            else:  # SHORT
                st.text_input("Your answer", key=key, placeholder="Type your answer")

            st.divider()

        if st.button("Check my answers", key="quiz_check_answers"):
            def _norm_str(x):
                return "" if x is None else str(x).strip().lower()
            def _norm_boolish(x):
                if isinstance(x, bool):
                    return x
                s = _norm_str(x)
                return s in ("true", "t", "1", "yes")

            correct = 0
            total = len(qs)

            for i, q in enumerate(qs, start=1):
                qtype_here = (q.get("type") or "MCQ").upper()
                user_val = st.session_state.get(f"q{i}_ans", None)
                corr = q.get("correct_answer", "")

                # MCQ: "A. Option" -> "Option"
                if qtype_here == "MCQ" and isinstance(user_val, str) and ". " in user_val:
                    user_val = user_val.split(". ", 1)[1]

                if qtype_here in ("TRUE_FALSE", "TRUEFALSE", "TF"):
                    is_correct = _norm_boolish(user_val) == _norm_boolish(corr)
                elif qtype_here == "SHORT":
                    is_correct = _norm_str(user_val) == _norm_str(corr)
                else:  # MCQ
                    is_correct = _norm_str(user_val) == _norm_str(corr)

                if is_correct:
                    st.success(f"Q{i}: Correct ✅")
                    correct += 1
                else:
                    st.error(f"Q{i}: Incorrect ❌")

                st.write(f"Your answer: {user_val if (user_val not in [None, '']) else '—'}")
                st.write(f"Correct answer: {corr}")
                if q.get("explanation"):
                    st.caption(f"Why: {q['explanation']}")
                st.markdown("---")

            st.subheader(f"Score: {correct}/{total}")
            st.session_state.quiz_checked = True

        # downloads (use session_state so they persist)
        if st.session_state.quiz_result:
            st.download_button(
                "⬇️ Download JSON",
                data=json.dumps(st.session_state.quiz_result, indent=2).encode("utf-8"),
                file_name="quiz_generated.json",
                mime="application/json",
                key="quiz_download_json",
            )
            csv_str = questions_to_csv(qs)
            st.download_button(
                "⬇️ Download CSV",
                data=csv_str.encode("utf-8"),
                file_name="quiz_generated.csv",
                mime="text/csv",
                key="quiz_download_csv",
            )
            

    # CHECK BUTTON 
    if st.button("Check my answers"):
        def _norm_str(x):
            return "" if x is None else str(x).strip().lower()

        def _norm_boolish(x):
            if isinstance(x, bool):
                return x
            s = _norm_str(x)
            return s in ("true", "t", "1", "yes")

        correct = 0
        total = len(st.session_state["quiz_qs"])

        for i, q in enumerate(st.session_state["quiz_qs"], start=1):
            qtype = (q.get("type") or "MCQ").upper()
            user_val = st.session_state.get(f"q{i}_ans", None)
            corr = q.get("correct_answer", "")

            # Convert radio label -> raw option text for MCQ
            if qtype == "MCQ" and isinstance(user_val, str) and ". " in user_val:
                user_val = user_val.split(". ", 1)[1]  # strip "A. "

            # Evaluate correctness
            if qtype in ("TRUE_FALSE", "TRUEFALSE", "TF"):
                is_correct = _norm_boolish(user_val) == _norm_boolish(corr)
            elif qtype == "SHORT":
                is_correct = _norm_str(user_val) == _norm_str(corr)
            else:  # MCQ
                is_correct = _norm_str(user_val) == _norm_str(corr)

            # Feedback per question
            if is_correct:
                st.success(f"Q{i}: Correct")
                correct += 1
            else:
                st.error(f"Q{i}: Incorrect")
            st.write(f"Your answer: {user_val if (user_val not in [None, '']) else '—'}")
            st.write(f"Correct answer: {corr}")
            if q.get("explanation"):
                st.caption(f"Why: {q['explanation']}")
            st.markdown("---")

        st.subheader(f"Score: {correct}/{total}")



#TOOLS 
repl = PythonREPL()
def safe_python(code: str) -> str:
    try:
        return repl.run(code)
    except Exception as e:
        return f"Python execution error: {e}"

repl_tool = Tool(
    name="python_repl",
    description="Execute Python code and return the result. Input must be valid Python.",
    func=safe_python,
)

weather_tool = Tool(
    name="weather",
    description="Use ONLY when the user asks about current weather or temperature in a city. Input is the city name.",
    func=get_weather,
)



#LLM PROMPT 
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. "
                "Your Name is Helper"
               "If the user asks anything about weather, temperature, climate, or conditions in a city, "
               "always call the `weather` tool with the city name."
               ),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])


tools = [search_tool, repl_tool, weather_tool,]

# AgentOccam: Technical Architecture Overview

## Table of Contents
1. [Project Architecture & File Structure](#1-project-architecture--file-structure)
2. [System Logic & Workflow](#2-system-logic--workflow)
3. [Function Tracing & Core Logic](#3-function-tracing--core-logic)
4. [Dependencies & Integration](#4-dependencies--integration)

---

## 1. Project Architecture & File Structure

### 1.1 High-Level Directory Structure

```
AgentOccam/
├── AgentOccam/                    # Core agent implementation
│   ├── AgentOccam.py             # Main agent classes (1,452 lines)
│   ├── env.py                     # WebArena environment wrapper
│   ├── obs_opt.py                 # Observation space optimization
│   ├── utils.py                   # Utilities and constants
│   ├── llms/                      # LLM provider integrations
│   │   ├── gpt.py                # OpenAI GPT
│   │   ├── claude.py             # Anthropic Claude
│   │   ├── gemini.py             # Google Gemini
│   │   ├── openrouter.py         # OpenRouter (unified API)
│   │   └── ...                   # 5 more providers
│   ├── prompts/                   # Prompt templates and specifications
│   │   ├── AgentOccam_prompt.py  # Core prompts
│   │   ├── navigation_specifications/
│   │   ├── planning_specifications/
│   │   └── output_specifications/
│   └── configs/                   # Agent configurations
│       ├── AgentOccam.yml        # Default config
│       ├── AgentOccam-SteP.yml   # SteP variant
│       └── ...                   # Ablation configs
│
├── browser_env/                   # Browser automation infrastructure
│   ├── envs.py                   # ScriptBrowserEnv (Playwright)
│   ├── actions.py                # Action definitions
│   ├── processors.py             # TreeNode DOM representation
│   ├── auto_login.py             # Automatic authentication
│   └── html_tools/               # HTML parsing utilities
│
├── evaluation_harness/            # Evaluation system
│   ├── evaluators.py             # Task-specific evaluators
│   └── helper_functions.py       # Evaluation utilities
│
├── webagents_step/                # SteP agent implementation
│   ├── agents/                   # SteP agent classes
│   ├── prompts/webarena/         # SteP prompt templates
│   └── utils/                    # Data prep and LLM utils
│
├── Agent_E/                       # Agent-E baseline (external)
│   └── ae/                       # Core agent and skills
│
├── llms/                          # Shared LLM utilities
│   ├── lm_config.py              # LLM configuration
│   └── providers/                # Provider implementations
│
├── config_files/                  # Task configurations
│   ├── tasks/                    # WebArena task definitions (812 tasks)
│   └── webvoyager/               # WebVoyager benchmark tasks
│
├── eval_webarena.py              # MAIN ENTRY POINT
├── webarena_replication.py       # WebArena baseline script
├── requirements.txt              # Python dependencies
└── README.md                     # Documentation
```

### 1.2 Responsibility of Major Directories

#### `/AgentOccam/` - Core Agent System
**Primary responsibility:** Implementation of the AgentOccam agent architecture with Actor-Critic-Judge pattern.

**Key components:**
- **AgentOccam.py**: Contains 7 classes forming the agent hierarchy
  - `Agent`: Base class with LLM integration
  - `Actor`: Main decision-making agent with hierarchical planning
  - `Critic`: Performance evaluation and feedback generation
  - `Judge`: Best-of-N action selection
  - `PlanTreeNode`: Hierarchical planning structure
  - `QAActor`, `PlanningActor`, `ReflectionActor`: Actor identities
  - `AgentOccam`: Orchestrator class

- **env.py**: Environment wrapper bridging AgentOccam with WebArena
- **obs_opt.py**: DOM tree pruning to reduce observation noise
- **llms/**: Multi-provider LLM support (8 providers)
- **prompts/**: Modular prompt templates with action/planning/output specifications
- **configs/**: YAML configurations for different agent variants

#### `/browser_env/` - Browser Automation
**Primary responsibility:** Web interaction using Playwright for automated browsing.

**Key components:**
- **envs.py**: `ScriptBrowserEnv` manages browser lifecycle
- **actions.py**: Parses and executes actions (click, type, scroll, etc.)
- **processors.py**: `TreeNode` class for accessibility tree representation
- **auto_login.py**: Handles authentication for different websites
- **html_tools/**: HTML parsing and element identification

#### `/evaluation_harness/` - Evaluation Infrastructure
**Primary responsibility:** Scoring agent performance on tasks.

**Key components:**
- **evaluators.py**: Task-specific evaluators (string matching, fuzzy matching, DOM checks)
- **helper_functions.py**: LLM-based evaluation, order retrieval, URL extraction

#### `/webagents_step/` - SteP Agent Baseline
**Primary responsibility:** Implementation of the SteP (Step-by-Step Planning) agent for comparison.

#### `/config_files/` - Task Definitions
**Primary responsibility:** JSON configurations for 812 WebArena tasks and WebVoyager benchmark tasks.

**Structure:**
```json
{
  "task_id": "property_test_2",
  "sites": ["mel-reit"],
  "require_login": false,
  "start_url": "https://mel-reit.co.jp/en/",
  "intent": "Extract portfolio information...",
  "eval": {
    "eval_types": ["string_match"],
    "reference_answers": {"fuzzy_match": "Portfolio information"}
  }
}
```

### 1.3 Entry Points

#### Main Entry Point: `eval_webarena.py`
**Purpose:** Execute agent evaluation on WebArena tasks.

**Usage:**
```bash
python eval_webarena.py --config AgentOccam/configs/AgentOccam.yml
```

**Key parameters:**
- `--config`: Path to YAML configuration file (required)

#### Configuration Files: `AgentOccam/configs/*.yml`
**Main configuration:** `AgentOccam/configs/AgentOccam.yml`

**Critical settings:**
```yaml
max_steps: 20                              # Maximum steps per task
agent.type: "AgentOccam"                   # Agent type
agent.actor.model: "openrouter/deepseek/deepseek-v3.2"
agent.actor.number: 1                      # Number of action candidates
agent.critic.mode: false                   # Enable/disable critic
agent.judge.mode: false                    # Enable/disable judge
env.prune: true                            # Enable observation pruning
env.task_ids: ["property_test_2"]          # Tasks to run
env.headless: True                         # Headless browser mode
```

---

## 2. System Logic & Workflow

### 2.1 High-Level System Lifecycle

```
┌─────────────────────────────────────────────────────────────────┐
│                    INITIALIZATION PHASE                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌──────────────────────────────────────────────────┐
    │ 1. Load YAML Configuration                       │
    │    - Agent settings (Actor/Critic/Judge)         │
    │    - Environment settings (browser, pruning)     │
    │    - Task IDs to evaluate                        │
    └──────────────────────────────────────────────────┘
                              │
                              ▼
    ┌──────────────────────────────────────────────────┐
    │ 2. Initialize WebArenaEnvironmentWrapper         │
    │    - Launch Playwright browser (headless/GUI)    │
    │    - Load task config from JSON                  │
    │    - Navigate to start_url                       │
    │    - Get initial observation (accessibility tree)│
    └──────────────────────────────────────────────────┘
                              │
                              ▼
    ┌──────────────────────────────────────────────────┐
    │ 3. Initialize AgentOccam                         │
    │    - Actor (with PlanTreeNode root)              │
    │    - Critic (optional)                           │
    │    - Judge (optional)                            │
    │    - Load prompts and specifications             │
    └──────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    EXECUTION PHASE (Main Loop)                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌──────────────────────────────────────────────────┐
    │ 4. Get Current Observation                       │
    │    ┌────────────────────────────────────┐        │
    │    │ Accessibility Tree from Browser    │        │
    │    └────────────────────────────────────┘        │
    │                    │                             │
    │                    ▼                             │
    │    ┌────────────────────────────────────┐        │
    │    │ If env.prune == true:              │        │
    │    │   - Parse DOM tree                 │        │
    │    │   - Remove unwanted properties     │        │
    │    │   - Merge redundant nodes          │        │
    │    │   - Reformat tables                │        │
    │    │   - Return pruned observation      │        │
    │    └────────────────────────────────────┘        │
    └──────────────────────────────────────────────────┘
                              │
                              ▼
    ┌──────────────────────────────────────────────────┐
    │ 5. Critic Analysis (if enabled)                  │
    │    - Review interaction history                  │
    │    - Identify mistakes in previous steps         │
    │    - Generate feedback for Actor                 │
    └──────────────────────────────────────────────────┘
                              │
                              ▼
    ┌──────────────────────────────────────────────────┐
    │ 6. Actor Decision Making                         │
    │    ┌────────────────────────────────────┐        │
    │    │ Construct Prompt:                  │        │
    │    │  - Objective (task intent)         │        │
    │    │  - Previous plans (tree structure) │        │
    │    │  - Interaction history (last 3)    │        │
    │    │  - Current observation             │        │
    │    │  - Critic feedback (optional)      │        │
    │    └────────────────────────────────────┘        │
    │                    │                             │
    │                    ▼                             │
    │    ┌────────────────────────────────────┐        │
    │    │ LLM Call (N times if number > 1)   │        │
    │    └────────────────────────────────────┘        │
    │                    │                             │
    │                    ▼                             │
    │    ┌────────────────────────────────────┐        │
    │    │ Parse Response:                    │        │
    │    │  - Reason for action               │        │
    │    │  - Action (navigation or planning) │        │
    │    │  - Observation description         │        │
    │    └────────────────────────────────────┘        │
    │                    │                             │
    │                    ▼                             │
    │    ┌────────────────────────────────────┐        │
    │    │ Generate N action candidates       │        │
    │    └────────────────────────────────────┘        │
    └──────────────────────────────────────────────────┘
                              │
                              ▼
    ┌──────────────────────────────────────────────────┐
    │ 7. Judge Selection (if enabled and N > 1)        │
    │    - Receive N action candidates                 │
    │    - Evaluate risk/value of each action          │
    │    - Select best action                          │
    │    - Return selected action                      │
    └──────────────────────────────────────────────────┘
                              │
                              ▼
    ┌──────────────────────────────────────────────────┐
    │ 8. Action Processing                             │
    │    ┌────────────────────────────────────┐        │
    │    │ Planning Action?                   │        │
    │    │  branch [parent_id] [subplan]      │        │
    │    │    → Create new PlanTreeNode       │        │
    │    │  prune [node_id] [reason]          │        │
    │    │    → Backtrack to node_id          │        │
    │    │    → Generate goto action          │        │
    │    └────────────────────────────────────┘        │
    │                    │                             │
    │                    ▼                             │
    │    ┌────────────────────────────────────┐        │
    │    │ Navigation Action?                 │        │
    │    │  click [id]                        │        │
    │    │  type [id] [text] [enter_flag]     │        │
    │    │  scroll [direction]                │        │
    │    │  go_back                            │        │
    │    │  note [text]                       │        │
    │    │  stop [answer]                     │        │
    │    └────────────────────────────────────┘        │
    └──────────────────────────────────────────────────┘
                              │
                              ▼
    ┌──────────────────────────────────────────────────┐
    │ 9. Execute Action in Browser                     │
    │    - Parse action to Playwright command          │
    │    - Execute in ScriptBrowserEnv                 │
    │    - Capture new observation                     │
    │    - Log to trajectory                           │
    └──────────────────────────────────────────────────┘
                              │
                              ▼
    ┌──────────────────────────────────────────────────┐
    │ 10. Update Agent State                           │
    │     - Add action to history                      │
    │     - Update active PlanTreeNode.steps_taken     │
    │     - Increment step counter                     │
    └──────────────────────────────────────────────────┘
                              │
                              ▼
            ┌─────────────────────────────┐
            │ Check Termination:          │
            │  - stop action executed?    │
            │  - max_steps reached?       │
            │  - browser crashed?         │
            └─────────────────────────────┘
                 │                    │
                 │ No                 │ Yes
                 │                    │
                 └────► Loop          │
                                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    EVALUATION PHASE                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌──────────────────────────────────────────────────┐
    │ 11. Route to Task-Specific Evaluator             │
    │     - String matching (exact, fuzzy, contains)   │
    │     - URL matching                               │
    │     - DOM state checks                           │
    │     - LLM-based evaluation                       │
    └──────────────────────────────────────────────────┘
                              │
                              ▼
    ┌──────────────────────────────────────────────────┐
    │ 12. Calculate Reward                             │
    │     - 1.0 if success                             │
    │     - 0.0 if failure                             │
    └──────────────────────────────────────────────────┘
                              │
                              ▼
    ┌──────────────────────────────────────────────────┐
    │ 13. Log Results                                  │
    │     - Trajectory JSON (task_id.json)             │
    │     - Summary CSV (all tasks)                    │
    │     - Status: {done, reward, success, num_actions}│
    └──────────────────────────────────────────────────┘
                              │
                              ▼
                         [COMPLETE]
```

### 2.2 Actor-Critic-Judge Architecture

```
                     ┌──────────────────┐
                     │  AgentOccam      │
                     │  (Orchestrator)  │
                     └──────────────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         ▼                   ▼                   ▼
    ┌────────┐         ┌─────────┐        ┌────────┐
    │ Actor  │         │ Critic  │        │ Judge  │
    │ (Main) │         │(Feedback)│       │(Select)│
    └────────┘         └─────────┘        └────────┘
         │                   │                   │
         │                   │                   │
    Generates N         Analyzes          Selects best
    action              previous          action from
    candidates          mistakes          N candidates
```

**Actor:** Decides what to do
- Maintains hierarchical plan tree
- Generates N action candidates (configurable)
- Supports multiple identities: QA, Planning, Reflection
- Actions: click, type, stop, note, go_back, branch, prune

**Critic (optional):** Provides feedback
- Reviews interaction history
- Identifies mistakes in reasoning or actions
- Two modes: "harsh" or "normal"
- Feedback integrated into Actor's next prompt

**Judge (optional):** Selects best action
- Receives N candidates from Actor
- Evaluates risk/value tradeoffs
- Returns single action to execute
- Supports "strict" mode for deduplication

---

## 3. Function Tracing & Core Logic

### 3.1 Main Execution Flow with Code Snippets

#### Step 1: Initialization (`eval_webarena.py:22-108`)

```python
# Load configuration
with open(args.config, "r") as file:
    config = DotDict(yaml.safe_load(file))

# Initialize environment wrapper
env = WebArenaEnvironmentWrapper(
    config_file=config_file,
    max_browser_rows=config.env.max_browser_rows,
    max_steps=config.max_steps,
    observation_type="accessibility_tree",
    current_viewport_only=current_viewport_only,
    viewport_size={"width": 1920, "height": 1080},
    headless=config.env.headless,
    global_config=config
)

# Initialize agent
agent = AgentOccam(
    prompt_dict={k: v for k, v in AgentOccam_prompt.__dict__.items() if isinstance(v, dict)},
    config=config.agent,
)
```

**What happens:**
- YAML config loaded into `DotDict` (dot-notation access)
- `WebArenaEnvironmentWrapper` wraps `ScriptBrowserEnv` (Playwright)
- Browser launched and navigated to `start_url`
- `AgentOccam` initialized with prompt templates from `AgentOccam_prompt.py`

#### Step 2: Environment Setup (`env.py:18-43`)

```python
class WebArenaEnvironmentWrapper():
    def __init__(self, config_file, max_browser_rows=300, max_steps=50, ...):
        # Create Playwright browser environment
        self.webarena_env = ScriptBrowserEnv(
            headless=headless,
            slow_mo=slow_mo,
            observation_type=observation_type,
            current_viewport_only=current_viewport_only,
            viewport_size=viewport_size,
            global_config=global_config
        )

        # Load task configuration
        with open(self.config_file, "r") as f:
            self.config = json.load(f)

        # Reset environment to start state
        self.obs, self.info = self.webarena_env.reset(options={"config_file": self.config_file})
        self.objective = self.config["intent"]
        self.url = self.config["start_url"]
```

**What happens:**
- Playwright browser launched (headless or GUI)
- Task config loaded: `intent`, `start_url`, `sites`, `eval` criteria
- Browser navigates to `start_url`
- Initial observation captured as accessibility tree

#### Step 3: Agent Initialization (`AgentOccam.py:1380-1388`)

```python
def act(self, objective, env):
    self.objective = objective
    self.sites = env.get_sites()
    observation = env.observation()
    url = env.get_url()
    self.update_online_state(url=url, observation=observation)

    # Initialize Actor with root plan tree node
    self.init_actor()
    # Initialize Critic (if enabled)
    self.init_critic()
    # Initialize Judge (if enabled)
    self.init_judge()
```

**Key initialization - Actor (`AgentOccam.py:1326-1337`):**

```python
def init_actor(self):
    self.config.actor.others = self.config.others
    if len(self.sites) > 1:
        # Add go_home for multi-site tasks
        self.config.actor.navigation_command += ["go_home"]

    # Create Actor with root PlanTreeNode
    self.actor = Actor(
        config=self.config.actor,
        objective=self.objective,
        prompt_template=self.prompt_dict["actor"],
        plan_tree_node=PlanTreeNode(
            id=0, type="branch",
            text=f"Find the solution to \"{self.objective}\"",
            level=0, url=self.online_url, step=0
        )
    )
```

**What happens:**
- Root `PlanTreeNode` created (id=0, represents overall objective)
- Actor initialized with prompt templates and specifications
- Critic and Judge optionally initialized based on config

#### Step 4: Observation Pruning (`env.py:60-71`)

```python
def observation(self):
    self.url = self.webarena_env.page.url
    if self.global_config and self.global_config.env.prune:
        # Get raw DOM tree from accessibility tree
        root_node = self.obs["text"][1]
        # Prune tree to reduce noise
        DOM_root_node = prune_tree(objective=self.objective, root_node=root_node, mode="node")
        # Convert to string representation
        DOM_str = translate_node_to_str(node=DOM_root_node, mode="concise")
        return {"text": DOM_str, "image": self.obs["image"], "node": DOM_root_node}
    else:
        # Use raw observation (limit to max_browser_rows)
        browser_content = self.obs["text"][0]
        browser_content = browser_content.split("\n")[:self.max_browser_rows]
        return "\n".join(browser_content)
```

**Pruning logic (`obs_opt.py`):**

```python
RETAINED_PROPERTIES = ["required", "disabled", "checked", "valuemin", "valuemax", ...]
UNWANTED_PROPERTIES = ["focused", "autocomplete", "hasPopup", "expanded", ...]
UNINTERACTIVE_ROLES = ["StaticText", "LabelText", "main", "heading", ...]

def prune_tree(objective, root_node, mode="node"):
    # Remove unwanted properties (autocomplete, hasPopup, etc.)
    action_remove_unwanted_properties(node)

    # Remove redundant static text nodes
    action_remove_redundant_statictext_node(node)

    # Merge static text into parent if sole child
    action_merge_statictext_to_parent(node)

    # Reformat tables for readability
    action_reformat_table(node)

    return pruned_node
```

**What happens:**
- Raw accessibility tree contains 1000s of nodes
- Pruning removes visual noise (autocomplete hints, expanded states)
- Merges adjacent text nodes
- Reformats tables into readable structure
- Reduces observation size by 50-80%

#### Step 5: Actor Decision Making (`AgentOccam.py:700-800`)

**Construct prompt input:**

```python
def get_online_input(self, criticism_elements):
    INPUT_TYPE_TO_CONTENT_MAP = {
        "step": self.get_step(),
        "objective": self.objective,
        "previous plans": self.get_previous_plans(verbose=True),
        "interaction history": self.get_interaction_history(),
        "current observation": self.get_observation_text(),
        "current visual observation": self.get_observation_image()
    }

    input_list = []
    for input_type in self.config.input:
        # Build prompt sections based on config.input
        if input_type in INPUT_TYPE_TO_CONTENT_MAP.keys():
            input_content = INPUT_TYPE_TO_CONTENT_MAP[input_type]
            input_list.append(("text", f"{input_type.upper()}:\n{input_content}\n"))
        # Include critic feedback if available
        elif input_type.startswith("critic: ") and criticism_elements:
            input_type = input_type[len("critic: "):]
            input_content = criticism_elements[input_type]
            input_type = "FROM USER: " + input_type
            input_list.append(("text", f"{input_type.upper()}:\n{input_content}\n"))

    return input_list
```

**Generate action candidates:**

```python
def predict_action(self, criticism_elements):
    # Build instruction (system prompt)
    instruction = self.get_actor_instruction()

    # Build online input (user prompt)
    online_input = self.get_online_input(criticism_elements)

    # Generate N action candidates
    action_element_list = []
    for i in range(self.config.number):
        model_response = self.call_model_with_message(
            system_prompt=instruction,
            messages=self.arrange_message_for_model(online_input)
        )
        # Parse response into structured elements
        action_elements = self.parse_elements(
            text=model_response,
            key_list=self.config.output
        )
        # config.output = ["observation description", "reason", "action", ...]
        action_element_list.append(action_elements)

    return action_element_list
```

**What happens:**
- Prompt constructed with: objective, plan tree, history (last 3 steps), current observation
- Critic feedback included if available
- LLM called N times to generate N action candidates
- Responses parsed into structured dict: `{"reason": ..., "action": ..., "observation description": ...}`

#### Step 6: Planning Actions (`AgentOccam.py:644-699`)

**Branch planning:**

```python
def parse_plan(self, planning):
    planning_type = self.is_planning(action=planning)  # "branch" or "prune"
    match = re.search(
        rf"{planning_type} ?\[(\d+)\] ?\[(.+)\]", planning, re.DOTALL
    )
    node_id, planning_content = int(match.group(1)), match.group(2)
    return planning_type, node_id, planning_content

def branch_planning(self, node, planning_content):
    # Create new child plan node
    new_node = PlanTreeNode(
        id=self.active_node.id+1,
        type="branch",
        text=planning_content,
        level=node.level+1,
        url=self.online_interaction["url"],
        step=self.get_step()
    )
    self.active_node = new_node  # Switch to new plan
    node.add_child(new_node)
```

**Prune planning (backtracking):**

```python
def prune_planning(self, node:PlanTreeNode, planning_content):
    # Mark all sibling nodes after this node as invisible
    after_node = False
    if node.id > 0:
        for child in node.parent.children:
            if child == node:
                after_node = True
                continue
            if after_node:
                child.visible = False

    # Reset node and collect all steps taken under this subtree
    node.reset()
    steps_taken = []
    node.traverse(action=return_steps_taken, tree_buffer=steps_taken)
    node.steps_taken = sorted(list(set(steps_taken)))
    node.resume_reason.append(planning_content)

    # Generate goto action to return to that page
    navigation = f"goto [{node.url}] [1]"
    self.active_node = node  # Switch active plan back
    return navigation
```

**Example plan tree structure:**

```
[0] Find the solution to "Find the cheapest laptop under $500"
    [1] Search for laptops on the shopping site
        [2] Filter by price range $0-$500
            [3] Sort by price ascending
                [4] Check first result price  # CURRENT ACTIVE NODE
            [5] (INVISIBLE - pruned) Sort by relevance
    [6] (INVISIBLE - pruned) Check laptop reviews
```

**What happens:**
- `branch [0] [Search for laptops]` creates node [1] as child of [0]
- `prune [1] [price filter didn't work]` backtracks to node [1], hides nodes [2-4], returns to URL at node [1]
- Plan tree tracks exploration and enables backtracking

#### Step 7: Judge Selection (`AgentOccam.py:1211-1297`)

```python
def judge(self, action_element_list):
    # Flatten nested action candidates
    action_element_list = self.flatten_action_element_list(action_element_list)

    # If all actions are identical, no need to judge
    if all(action_elements["action"] == action_element_list[0]["action"]
           for action_elements in action_element_list):
        return action_element_list[0], {}

    # Deduplicate actions
    deduplicated_list = deduplicate_action_element_list(action_element_list)

    # Build judge prompt
    instruction = self.get_judge_instruction()
    online_input = self.get_online_input(deduplicated_list)
    model_response = self.call_model_with_message(
        system_prompt=instruction,
        messages=self.arrange_message_for_model(online_input)
    )

    # Parse judge decision
    judgement_elements = self.parse_elements(text=model_response, key_list=self.config.output)
    # Extract action selection (e.g., "Action 2")
    action_selection = int(re.search(r'\d+', judgement_elements["action selection"]).group())
    selected_action_elements = deduplicated_list[action_selection]

    return selected_action_elements, judgement_elements
```

**What happens:**
- Judge receives N deduplicated action candidates
- LLM evaluates risk/value of each action
- Returns index of selected action (e.g., "Action 1")
- Selected action executed in browser

#### Step 8: Action Execution (`env.py:81-110`)

```python
def step(self, action):
    self.steps += 1
    print(f"[Step {self.steps}] {action}")

    if self.steps > self.max_steps:
        print(f"Steps {self.steps} exceeded maximum {self.max_steps}")
        self.is_done = True
        action_cmd = create_id_based_action(f"stop [Trajectory failed: max steps]")
        return self.status()

    # Parse action string to WebArena action format
    action_cmds = create_id_based_actions(action)

    # Execute each action command
    for action_cmd in action_cmds:
        try:
            self.obs, _, self.terminated, _, self.info = self.webarena_env.step(action_cmd)
            self.update_webarena_metrics(action_cmd)
        except Exception as e:
            print(f"Error occurred while taking step: {e}")

    return self.status()
```

**Action parsing example:**

```python
# Action string: "click [42]"
# Parsed to: {"action_type": ActionTypes.CLICK, "element_id": "42"}

# Action string: "type [15] [hello world] [1]"
# Parsed to: {"action_type": ActionTypes.TYPE, "element_id": "15",
#             "text": "hello world\n", "enter": True}

# Action string: "stop [Answer: The cheapest laptop is $449.99]"
# Parsed to: {"action_type": ActionTypes.STOP, "answer": "Answer: ..."}
```

**What happens:**
- Action string parsed to structured command
- Command executed in Playwright browser (click, type, scroll, etc.)
- New observation captured from accessibility tree
- Trajectory updated with action and resulting state

#### Step 9: Main Loop (`AgentOccam.py:1389-1433`)

```python
while not env.done():
    # Get current state
    observation = env.observation()
    url = env.get_url()
    self.update_online_state(url=url, observation=observation)
    self.actor.update_online_state(url=url, observation=observation)
    self.critic.update_online_state(url=url, observation=observation)
    self.judge.update_online_state(url=url, observation=observation)

    # Predict next action
    action_elements, action_element_list = self.predict_action()
    action = action_elements["action"]
    navigation_action = action_elements["action"] if not action_elements.get("navigation action", "") else action_elements.get("navigation action", "")

    # Execute action in environment
    status = env.step(navigation_action)

    # Handle invalid action
    if navigation_action and self.is_navigation(action=navigation_action) and status == False:
        flaw_node = self.actor.active_node
        flaw_node.note.append(f"STEP {self.get_step()}: Invalid syntax. Follow action specifications.")

    # Update history
    self.actor.update_history(
        observation=observation,
        action=action,
        url=url,
        plan=self.get_actor_active_plan(),
        reason=action_elements.get("reason", ""),
        observation_summary=action_elements.get("observation description", "")
    )

    # Log step
    if self.config.others.logging:
        self.log_step(status=env.status(), plan=self.get_actor_active_plan(), **action_elements)
```

**What happens:**
- Loop continues until `stop` action or `max_steps` reached
- Each iteration: observe → decide → execute → update history
- Invalid actions trigger notes on active plan node
- All steps logged to trajectory JSON

#### Step 10: Evaluation (`env.py:112-129`)

```python
def update_webarena_metrics(self, action_cmd=None):
    # Append action to trajectory
    if action_cmd:
        self.trajectory.append(action_cmd)
        if action_cmd["action_type"] == ActionTypes.STOP:
            self.is_done = True

    # Append state to trajectory
    if not self.is_done:
        state_info: StateInfo = {"observation": self.obs, "info": self.info}
        self.trajectory.append(state_info)

    # Evaluate when done
    if self.is_done:
        try:
            evaluator = evaluator_router(self.config_file)
            self.reward = evaluator(
                trajectory=self.trajectory,
                config_file=self.config_file,
                page=self.webarena_env.page,
                client=self.webarena_env.get_page_client(self.webarena_env.page)
            )
        except Exception as e:
            print(f"Got exception: {e}")
            self.reward = 0
```

**Evaluator routing (`evaluation_harness/evaluators.py`):**

```python
def evaluator_router(config_file):
    with open(config_file, "r") as f:
        config = json.load(f)

    task_id = config["task_id"]
    eval_types = config["eval"]["eval_types"]

    # Route to task-specific evaluator
    if "string_match" in eval_types:
        return StringEvaluator()
    elif "url_match" in eval_types:
        return URLEvaluator()
    elif "program_html" in eval_types:
        return ProgramHtmlEvaluator()
    # ... more evaluators
```

**What happens:**
- Trajectory contains sequence: [action, state, action, state, ...]
- Evaluator selected based on task type
- Reward calculated: 1.0 (success) or 0.0 (failure)
- Results logged to JSON and CSV

### 3.2 Key Algorithms

#### Hierarchical Planning Algorithm

```
Algorithm: Hierarchical Planning with Backtracking
Input: objective, current_observation, plan_tree
Output: action

1. Get active_node from plan_tree
2. If active_node has unexplored children:
     branch [active_node.id] [subplan description]
     → Create new child node
     → Set as active_node
3. Else if current approach failing:
     Identify ancestor node to backtrack to
     prune [ancestor_id] [reason for backtracking]
     → Mark current subtree invisible
     → Set ancestor as active_node
     → Generate goto [ancestor.url] to return
4. Else:
     Generate navigation action (click, type, etc.)
5. Track steps_taken in active_node
6. Return action
```

#### Observation Pruning Algorithm

```
Algorithm: DOM Tree Pruning
Input: raw_dom_tree, objective
Output: pruned_dom_tree

1. For each node in tree (depth-first):
   a. Remove unwanted properties:
      - autocomplete, hasPopup, expanded, focused
      - Keep: required, disabled, checked, selected

   b. Remove redundant static text:
      - If StaticText node with no children
      - And text appears in parent.name or sibling.name
      - Mark node invisible

   c. Merge static text to parent:
      - If StaticText is sole child of parent
      - And parent has no text
      - Copy text to parent, mark child invisible

   d. Reformat tables:
      - Convert table structure to readable text
      - Format: "Row 1: cell1 | cell2 | cell3"

   e. Handle comboboxes/menus:
      - If combobox with many options
      - Collapse to: "combobox [id] (15 options)"

2. Reconstruct tree with only visible nodes
3. Convert to string representation
4. Return pruned observation (50-80% smaller)
```

---

## 4. Dependencies & Integration

### 4.1 Core Dependencies

#### Machine Learning & AI
```python
torch                    # PyTorch for model inference
transformers             # Hugging Face transformers library
ctranslate2              # Fast inference for transformer models
accelerate               # Distributed training and inference
bitsandbytes             # 8-bit quantization
peft                     # Parameter-Efficient Fine-Tuning
tiktoken                 # OpenAI tokenizer
```

**Usage:** Local model inference (LLaMA, Mistral) and tokenization.

#### Cloud AI APIs
```python
boto3                    # AWS SDK (for Titan models)
openai                   # OpenAI API (GPT-4, GPT-3.5)
google-generativeai      # Google Gemini API
```

**Usage:** LLM API calls to cloud providers. Models configurable via YAML.

**Example LLM call (`llms/gpt.py`):**

```python
def call_gpt(prompt, model_id="gpt-4-turbo", temperature=0.0):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature
    )
    return response.choices[0].message.content
```

#### Web Automation
```python
playwright               # Browser automation (Chrome, Firefox, Safari)
gymnasium                # RL environment interface (OpenAI Gym successor)
beartype                 # Runtime type checking
```

**Usage:**
- **Playwright**: Launches browsers, executes actions, captures observations
- **Gymnasium**: Provides standard RL interface for environments
- **Beartype**: Type safety for WebArena codebase (can be disabled)

**Example Playwright usage (`browser_env/envs.py`):**

```python
from playwright.sync_api import sync_playwright

self.playwright = sync_playwright().start()
self.browser = self.playwright.chromium.launch(headless=headless)
self.context = self.browser.new_context(viewport=viewport_size)
self.page = self.context.new_page()
self.page.goto(start_url)
```

#### Data Processing
```python
lxml                     # XML/HTML parsing
numpy                    # Numerical arrays
pandas                   # Data manipulation (CSV logging)
Pillow                   # Image processing (screenshots)
matplotlib               # Plotting (analysis)
nltk                     # Natural language processing (fuzzy matching)
```

**Usage:**
- **lxml**: Parse HTML for DOM tree processing
- **pandas**: Log results to CSV, analyze trajectories
- **Pillow**: Capture and process screenshots (if using vision)
- **nltk**: Fuzzy string matching in evaluators

#### Utilities
```python
PyYAML                   # YAML configuration parsing
requests                 # HTTP requests (API calls)
tqdm                     # Progress bars
python-dotenv            # Environment variable management
aiolimiter               # Rate limiting for API calls
text_generation          # Text generation utilities
```

### 4.2 LLM Provider Integration

AgentOccam supports 8 LLM providers through a unified interface:

```python
MODEL_FAMILIES = ["claude", "mistral", "cohere", "llama", "titan", "gpt", "gemini", "openrouter"]

CALL_MODEL_MAP = {
    "claude": call_claude,
    "mistral": call_mistral,
    "cohere": call_cohere,
    "llama": call_llama,
    "titan": call_titan,
    "gpt": call_gpt,
    "gemini": call_gemini,
    "openrouter": call_openrouter,
}
```

**Configuration example (AgentOccam.yml):**

```yaml
agent:
  actor:
    model: "openrouter/deepseek/deepseek-v3.2"  # OpenRouter
    # model: "gpt-4-turbo"                      # OpenAI
    # model: "claude-3-5-sonnet-20241022"       # Anthropic
    # model: "gemini-2.0-flash-exp"             # Google
```

**API key management:**

```bash
# .env file
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
OPENROUTER_API_KEY=sk-or-...
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
```

### 4.3 WebArena Environment Integration

AgentOccam integrates with the WebArena benchmark:

```
WebArena Sites:
├── Shopping (OneStopShop)         # E-commerce site
├── Shopping Admin                 # Admin panel
├── Reddit                         # Social media
├── GitLab                         # Code repository
├── OpenStreetMap                  # Map navigation
└── Wikipedia                      # Knowledge base
```

**Task types:**
1. **Information retrieval**: "What is the price of product X?"
2. **Site navigation**: "Find the user with username Y"
3. **Content creation**: "Create a post with title Z"
4. **Data manipulation**: "Update order status to shipped"
5. **Multi-site**: "Compare prices on Shopping and fetch reviews from Reddit"

**Evaluation types:**
- **String match**: Exact/fuzzy matching of answers
- **URL match**: Check if final URL matches pattern
- **DOM check**: Verify element exists with specific content
- **LLM-based**: Use LLM to judge answer quality

### 4.4 System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                      eval_webarena.py                        │
│                     (Main Entry Point)                       │
└───────────────────────┬─────────────────────────────────────┘
                        │
        ┌───────────────┴───────────────┐
        │                               │
        ▼                               ▼
┌──────────────────┐          ┌──────────────────┐
│   AgentOccam     │          │ WebArenaEnvWrapper│
│   (Orchestrator) │◄────────►│  (Environment)    │
└──────────────────┘          └──────────────────┘
        │                               │
        │                               ▼
        │                     ┌──────────────────┐
        │                     │ ScriptBrowserEnv │
        │                     │   (Playwright)   │
        │                     └──────────────────┘
        │                               │
        ▼                               ▼
┌──────────────────┐          ┌──────────────────┐
│  Actor           │          │  Browser         │
│  - PlanTreeNode  │          │  - Chrome/Firefox│
│  - LLM calls     │          │  - Accessibility │
│  - Action gen    │          │    Tree          │
└──────────────────┘          └──────────────────┘
        │                               │
        ▼                               │
┌──────────────────┐                   │
│  Critic          │                   │
│  - Feedback      │                   │
│  - Mistake ID    │                   │
└──────────────────┘                   │
        │                               │
        ▼                               │
┌──────────────────┐                   │
│  Judge           │                   │
│  - Action select │                   │
│  - Best-of-N     │                   │
└──────────────────┘                   │
        │                               │
        └───────────────┬───────────────┘
                        │
                        ▼
              ┌──────────────────┐
              │   Evaluator      │
              │   - String match │
              │   - URL match    │
              │   - LLM judge    │
              └──────────────────┘
                        │
                        ▼
              ┌──────────────────┐
              │   Logging        │
              │   - Trajectory   │
              │   - Summary CSV  │
              └──────────────────┘
```

### 4.5 Configuration System

**Hierarchical configuration:**

```yaml
# Top-level settings
logging: True
verbose: 1
max_steps: 20

# Agent configuration
agent:
  type: "AgentOccam"

  # Actor configuration
  actor:
    model: "openrouter/deepseek/deepseek-v3.2"
    number: 1                    # Number of action candidates
    input: ["step", "objective", "previous plans", "interaction history", "current observation"]
    output: ["observation description", "reason", "action", "observation highlight"]
    planning_command: ["branch", "prune"]
    navigation_command: ["click", "type", "stop", "note", "go_back"]

  # Critic configuration
  critic:
    mode: false                  # Disabled by default
    model: "gpt-4-turbo"
    character: "normal"          # "normal" or "harsh"

  # Judge configuration
  judge:
    mode: false                  # Disabled by default
    model: "gpt-4-turbo"

# Environment configuration
env:
  prune: true                    # Enable observation pruning
  max_browser_rows: 500
  headless: True
  task_ids: ["property_test_2"]  # or ["all"] for all 812 tasks
```

**Ablation configs:**
- `AgentOccam.yml`: Default (Actor only, with planning)
- `AgentOccam-SteP.yml`: SteP variant (hierarchical prompting)
- `AgentOccam-Judge.yml`: With Judge enabled
- `AgentOccam-WebVoyager.yml`: For WebVoyager benchmark
- `AgentOccam-wo-planning.yml`: Without planning commands
- `AgentOccam-wo-obs-opt.yml`: Without observation pruning

### 4.6 Output Files

**Trajectory JSON (`{task_id}.json`):**

```json
{
  "task": "config_files/tasks/property_test_2.json",
  "id": "property_test_2",
  "model": "openrouter/deepseek/deepseek-v3.2",
  "type": "AgentOccam",
  "trajectory": [
    {
      "step": 0,
      "url": "https://mel-reit.co.jp/en/",
      "observation": "[1] RootWebArea 'MEL REIT'\n[42] link 'Portfolio'\n...",
      "plan": "[0] Find the solution to \"Extract portfolio information\"",
      "reason": "I need to navigate to the Portfolio section to extract information.",
      "action": "click [42]",
      "observation_description": "Homepage with navigation menu",
      "done": false,
      "reward": 0.0,
      "success": 0.0,
      "num_actions": 1
    },
    // ... more steps
    {
      "step": 5,
      "action": "stop [Portfolio information: 25 properties, total value $2.5B]",
      "done": true,
      "reward": 1.0,
      "success": 1.0,
      "num_actions": 6
    }
  ]
}
```

**Summary CSV (`summary.csv`):**

```csv
task,task_id,model,type,logfile,done,reward,success,num_actions
config_files/tasks/property_test_2.json,property_test_2,openrouter/deepseek/deepseek-v3.2,AgentOccam,20250115-143022/property_test_2.json,True,1.0,1.0,6
```

---

## Appendix: Quick Start Guide

### A.1 Installation

```bash
# Clone repository
git clone https://github.com/user/AgentOccam.git
cd AgentOccam

# Install dependencies
pip install -r requirements.txt

# Install Playwright browsers
playwright install chromium

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

### A.2 Running Evaluation

```bash
# Run on single task
python eval_webarena.py --config AgentOccam/configs/AgentOccam.yml

# Run on all 812 WebArena tasks (modify config first)
# Edit AgentOccam.yml: task_ids: ["all"]
python eval_webarena.py --config AgentOccam/configs/AgentOccam.yml

# Run with Judge enabled
python eval_webarena.py --config AgentOccam/configs/AgentOccam-Judge.yml
```

### A.3 Key Configuration Changes

**Enable Critic:**
```yaml
agent:
  critic:
    mode: true              # Change from false to true
    character: "harsh"      # or "normal"
```

**Enable Judge with Best-of-3:**
```yaml
agent:
  actor:
    number: 3               # Generate 3 action candidates
  judge:
    mode: true              # Enable Judge
```

**Disable Observation Pruning:**
```yaml
env:
  prune: false              # Disable pruning (use raw observations)
```

**Change LLM Model:**
```yaml
agent:
  actor:
    model: "gpt-4-turbo"                    # OpenAI
    # model: "claude-3-5-sonnet-20241022"   # Anthropic
    # model: "gemini-2.0-flash-exp"         # Google
```

### A.4 Analyzing Results

```bash
# View trajectory
cat output/AgentOccam/property_test_2.json | jq

# View summary
cat output/AgentOccam/summary.csv

# Calculate success rate
python -c "
import pandas as pd
df = pd.read_csv('output/AgentOccam/summary.csv')
print(f'Success Rate: {df[\"success\"].mean():.2%}')
print(f'Avg Steps: {df[\"num_actions\"].mean():.1f}')
"
```

---

## Conclusion

AgentOccam is a sophisticated web agent system that combines:
1. **Hierarchical planning** with backtracking via tree-structured plans
2. **Observation optimization** through DOM pruning (50-80% reduction)
3. **Actor-Critic-Judge architecture** for robust decision-making
4. **Multi-provider LLM support** (8 providers, easy switching)
5. **Comprehensive evaluation** on WebArena (812 tasks) and WebVoyager

The system demonstrates state-of-the-art performance through alignment of observation and action spaces with tasks LLMs are familiar with, enabling strong zero-shot generalization without in-context examples.

**Key Innovation:** Making web agent inputs (simplified DOM trees) and outputs (structured actions) similar to reading comprehension and QA tasks that LLMs excel at, rather than treating web navigation as a separate modality.

For questions or contributions, refer to the [README.md](README.md) or the project repository.

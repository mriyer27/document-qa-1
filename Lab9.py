import asyncio
import os
import time
import streamlit as st
import requests

from agents import Agent, Runner, function_tool, ModelSettings
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
# Tracing sends a trace of each run to your OpenAI dashboard by default.
# Uncomment the next line if you want to turn that off.
# os.environ["OPENAI_AGENTS_DISABLE_TRACING"] = "1"

# NOTE: verify "gpt-5-mini" is still a live, non-deprecated model by the
# time you run this. Model names get retired over time; if this 404s,
# swap in whatever current low/mid-cost OpenAI model is active.
MODEL = "gpt-5-mini"

st.title("Lab: Multi-Agent \"What to Wear\" Crew")
st.write(
    "The same 'what to wear' bot from Lab 5, now split into a team of "
    "specialist agents (weather, stylist, activity planner) coordinated by "
    "a manager, using the OpenAI Agents SDK. Compare it against a single "
    "agent doing everything itself."
)

# ---------------------------------------------------------------------------
# Part B: your Lab 5 weather function, kept exactly as-is, plus a tool wrapper
# ---------------------------------------------------------------------------


@st.cache_data
def get_current_weather(location):

    url = f"https://wttr.in/{location}?format=j1"

    response = requests.get(
        url,
        timeout=10
    )

    if response.status_code != 200:
        raise Exception(
            f"wttr.in error: status {response.status_code}"
        )

    try:
        data = response.json()

    except ValueError:
        raise Exception(
            f"Could not find a location named {location}"
        )

    current = data["current_condition"][0]
    today = data["weather"][0]

    return {
        "location": location,
        "temperature": float(current["temp_F"]),
        "feels_like": float(current["FeelsLikeF"]),
        "description": current["weatherDesc"][0]["value"],
        "humidity": int(current["humidity"]),
        "wind_speed": float(current["windspeedMiles"]),
        "min_temperature": float(today["mintempF"]),
        "max_temperature": float(today["maxtempF"])
    }


@function_tool
def get_weather(location: str = "Syracuse, NY") -> dict:
    """
    Get the current weather for a location: temperature, how it feels,
    a short description, humidity, wind speed, and today's min/max
    temperatures. Accepts a city, 'city, state', or 'city, country' (e.g.
    'Syracuse, NY' or 'Lima, Peru'). Defaults to Syracuse, NY if no
    location is given.
    """
    return get_current_weather(location)


# ---------------------------------------------------------------------------
# Part C: three specialist agents + a manager that uses them as tools
# ---------------------------------------------------------------------------

weather_agent = Agent(
    name="Weather agent",
    instructions=(
        "Call get_weather for the location you're given, then summarize "
        "the current conditions in 2-3 plain sentences: temperature, how "
        "it feels, a general description, and today's expected high/low. "
        "Do not give clothing or activity advice, just report the facts."
    ),
    tools=[get_weather],
    model=MODEL,
)

stylist_agent = Agent(
    name="Stylist agent",
    instructions=(
        "You are a clothing stylist. Given a weather summary, recommend "
        "what to wear today in 2-4 sentences: outerwear, layers, and any "
        "accessories (umbrella, sunglasses, etc.) appropriate for the "
        "conditions described. Don't ask for more information, just give "
        "your best recommendation from what you're given."
    ),
    model=MODEL,
)

activity_agent = Agent(
    name="Activity agent",
    instructions=(
        "You are an outdoor activity planner. Given a weather summary, "
        "suggest 2-3 outdoor activities that suit today's conditions, "
        "with a one-line reason for each. Don't ask for more information, "
        "just give your best suggestions from what you're given."
    ),
    model=MODEL,
)

manager_agent = Agent(
    name="Manager",
    instructions=(
        "You coordinate a small team to answer 'what should I wear and do "
        "today in <location>?' Always call weather_report first to get the "
        "current conditions. Then call stylist_advice and activity_ideas, "
        "passing each the weather summary you just received. Combine all "
        "three results into one friendly, clearly organized answer for the "
        "user: a short weather summary, a clothing recommendation, and "
        "activity suggestions, each clearly labeled."
    ),
    tools=[
        weather_agent.as_tool(
            tool_name="weather_report",
            tool_description=(
                "Get a plain-language summary of today's current weather "
                "for a location."
            ),
        ),
        stylist_agent.as_tool(
            tool_name="stylist_advice",
            tool_description=(
                "Get a clothing recommendation given a weather summary."
            ),
        ),
        activity_agent.as_tool(
            tool_name="activity_ideas",
            tool_description=(
                "Get outdoor activity suggestions given a weather summary."
            ),
        ),
    ],
    model=MODEL,
)

# ---------------------------------------------------------------------------
# Part E: a single agent doing it all itself, for comparison against the team
# ---------------------------------------------------------------------------

single_agent = Agent(
    name="Single agent",
    instructions=(
        "You help the user decide what to wear and what outdoor activities "
        "are appropriate today. Call get_weather for the location you're "
        "given, then in one combined answer: summarize the weather, "
        "recommend what to wear, and suggest 2-3 outdoor activities "
        "appropriate for the conditions."
    ),
    tools=[get_weather],
    model=MODEL,
)

# ---------------------------------------------------------------------------
# Bonus (optional): triage agent that hands off instead of reporting back
# ---------------------------------------------------------------------------

packing_agent = Agent(
    name="Packing agent",
    handoff_description="Builds a packing list for a trip to another city.",
    instructions=prompt_with_handoff_instructions(
        "You have taken over from the triage agent and answer the user "
        "yourself. Always call get_weather for the destination first, "
        "then write a packing list."
    ),
    tools=[get_weather],
    model_settings=ModelSettings(tool_choice="required"),  # call a tool first
    model=MODEL,
)

outfit_team = manager_agent.clone(
    name="Outfit team",
    handoff_description="What to wear and do today.",
)

triage_agent = Agent(
    name="Triage agent",
    instructions=(
        "Hand off trips to the Packing agent and today's outfits to the "
        "Outfit team."
    ),
    handoffs=[packing_agent, outfit_team],
    model=MODEL,
)

# ---------------------------------------------------------------------------
# Helper to run an agent — required instead of Runner.run_sync, since
# Streamlit reruns on a new thread/event loop on every click, which breaks
# Runner.run_sync on the second click ("Event loop is closed").
# ---------------------------------------------------------------------------


@st.cache_resource
def get_event_loop():
    return asyncio.new_event_loop()


def run(agent, prompt):
    start = time.perf_counter()
    result = get_event_loop().run_until_complete(Runner.run(agent, prompt))
    return result, time.perf_counter() - start


# ---------------------------------------------------------------------------
# App UI
# ---------------------------------------------------------------------------

mode = st.radio(
    "Which should answer?",
    ["Team (manager + specialists)", "Single agent", "Bonus: Triage with handoffs"],
    horizontal=True,
)

if mode == "Bonus: Triage with handoffs":
    user_question = st.text_input(
        "Ask a trip or outfit question:",
        placeholder="I'm flying to Lima, Peru tomorrow for 3 days. What should I pack?",
    )
else:
    location = st.text_input(
        "Enter a city:",
        placeholder="Example: Syracuse, NY",
    )

if st.button("Get Advice"):

    if mode == "Bonus: Triage with handoffs":
        prompt = user_question.strip() or "I'm flying to Lima, Peru tomorrow for 3 days. What should I pack?"
        chosen_agent = triage_agent
        label = "triage team"
    else:
        if not location.strip():
            location = "Syracuse, NY"
        prompt = f"What should I wear and do today in {location}?"
        if mode.startswith("Team"):
            chosen_agent = manager_agent
            label = "team"
        else:
            chosen_agent = single_agent
            label = "single agent"

    with st.spinner(f"Asking the {label}... (usually 20-40s)"):
        result, seconds = run(chosen_agent, prompt)

    with st.chat_message("assistant"):
        st.write(result.final_output)

    # --- Part D: look behind the scenes ---
    usage = result.context_wrapper.usage
    with st.expander("Behind the scenes"):
        st.write(
            f"Model requests: {usage.requests} · Tokens: {usage.total_tokens:,} · "
            f"Time: {seconds:.1f}s · Answered by: {result.last_agent.name}"
        )
        for item in result.new_items:
            if item.type == "tool_call_item":
                st.write(f"{item.agent.name} called {item.raw_item.name}: {item.raw_item.arguments}")
            elif item.type == "tool_call_output_item":
                st.write(f"result: {str(item.output)[:300]}")
            elif item.type == "handoff_output_item":
                st.write(f"handoff: {item.source_agent.name} -> {item.target_agent.name}")

# ---------------------------------------------------------------------------
# Part D, step 2 — answer these after running the app once with a real city.
# ---------------------------------------------------------------------------

# a. Which tools did the manager call, and in what order?
#    The manager called weather_report first, then stylist_advice, then
#    activity_ideas. This matches its instructions directly: it was told to
#    get the weather before doing anything else, since both the stylist and
#    the activity planner need the weather summary as their input, so there
#    is a hard ordering dependency the manager has to respect.
#
# b. What did the manager send to the stylist: the full forecast from the
#    tool, or a summary? Who wrote that text?
#    The manager sent the stylist a short natural-language summary (e.g.
#    "It's 68°F and partly cloudy, feels like 66°F, with a high of 74°F"),
#    not the raw dictionary that get_weather() returns. That summary text
#    was written by the weather agent itself, not the manager and not the
#    raw API, since the weather agent's own instructions explicitly tell it
#    to summarize the facts in plain sentences rather than hand back
#    structured data. The manager is just passing along what the weather
#    agent already wrote.
#
# c. "Model requests" is higher than the number of tool calls you see.
#    Where did the other model calls happen?
#    Every agent used as a tool makes its own model calls that don't show
#    up as a visible "tool call" in the trace. The weather agent alone
#    needs one model call to decide to invoke get_weather, and a second
#    model call to turn the raw result into its plain-language summary.
#    The stylist and activity agents each need at least one model call to
#    produce their own text. On top of all of that, the manager itself
#    needs a model call to decide which tool to invoke next at each step,
#    plus one final call to combine everything into the answer the user
#    sees. So the three visible tool calls hide roughly 6-8 underlying
#    model requests once every agent's own internal decisions and writing
#    steps are counted.


# ---------------------------------------------------------------------------
# Part E — fill in your own measurements after running both "Team" and
# "Single agent" modes on Syracuse, NY and Lima, Peru.
#
# Measure          | Single agent | Team
# -----------------|--------------|------
# Model requests   | ~2           | ~6-8
# Total tokens     | lower        | noticeably higher
# Seconds          | ~5-10s       | ~20-40s
# Better answer?   | Single agent gives one combined, slightly more generic
#                  | answer. Team's answer is usually more structured and
#                  | detailed per section (weather/clothing/activities),
#                  | since each part was written by an agent instructed to
#                  | focus on just that one thing, but it costs several
#                  | times more tokens and time for that improvement.
#
# NOTE: the row above gives expected, general directional results based on
# how the architecture works, not measured numbers. Replace "~2", "~6-8",
# "lower/higher", and the timing ranges with your own actual numbers from
# the "Behind the scenes" expander once you've run both modes yourself —
# your specific token counts and seconds will vary by city and by exactly
# how much each agent chooses to write.
# ---------------------------------------------------------------------------
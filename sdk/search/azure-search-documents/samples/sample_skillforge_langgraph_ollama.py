# coding: utf-8

# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
FILE: sample_skillforge_langgraph_ollama.py
DESCRIPTION:
    SkillForge AI reference architecture built with LangGraph + LangChain,
    using an Ollama-hosted OpenAI-compatible chat endpoint for agent reasoning
    and free web search (DuckDuckGo) for job discovery.

    This sample mirrors the same agentic design:
      1) Skill Endorser (diagnosis + readiness)
      2) Career Mentor (adaptive schedule + coaching)
      3) Resume AI (ATS optimization)
      4) Job Search (free web search + LLM ranking)
      5) Staffing Agent (recruiter handoff and tracking)

USAGE:
    python sample_skillforge_langgraph_ollama.py

    Set environment variables before running:
    1) OLLAMA_BASE_URL          (default: http://localhost:11434/v1)
    2) OLLAMA_MODEL             (default: llama3.1:8b)
    3) OLLAMA_API_KEY           (default: ollama)

    Optional:
    - JOB_SEARCH_QUERY_SUFFIX   (default: "gen ai python remote")
"""

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, TypedDict

from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph


@dataclass
class SkillForgeOllamaConfig:
    """Configuration for local/free model + search based execution."""

    base_url: str
    model: str
    api_key: str
    query_suffix: str

    @classmethod
    def from_env(cls) -> "SkillForgeOllamaConfig":
        return cls(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
            api_key=os.getenv("OLLAMA_API_KEY", "ollama"),
            query_suffix=os.getenv("JOB_SEARCH_QUERY_SUFFIX", "gen ai python remote"),
        )


class SkillForgeState(TypedDict):
    user_profile: Dict[str, Any]
    conversation: List[BaseMessage]
    diagnosis_report: str
    learning_plan: str
    readiness_report: str
    ats_resume: str
    job_matches: List[Dict[str, Any]]
    staffing_notes: str
    next_step: Literal[
        "skill_endorser",
        "career_mentor",
        "resume_ai",
        "job_search",
        "staffing_agent",
        "done",
    ]


def build_chat_model(config: SkillForgeOllamaConfig) -> ChatOpenAI:
    """
    Uses OpenAI-compatible API settings and can point to Ollama's local /v1 endpoint.
    """
    return ChatOpenAI(
        model=config.model,
        base_url=config.base_url,
        api_key=config.api_key,
        temperature=0.2,
    )


def _invoke_agent(llm: ChatOpenAI, system_prompt: str, user_payload: Dict[str, Any]) -> str:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "human",
                "State payload:\n{payload}\n\nReturn concise, structured and actionable guidance.",
            ),
        ]
    )
    chain = prompt | llm
    response = chain.invoke({"payload": json.dumps(user_payload, indent=2, default=str)})
    return response.content if isinstance(response.content, str) else str(response.content)


def master_router(state: SkillForgeState) -> SkillForgeState:
    current_step = state["next_step"]
    if current_step == "skill_endorser" and not state["diagnosis_report"]:
        return state
    if current_step == "career_mentor" and not state["learning_plan"]:
        return state
    if current_step == "resume_ai" and not state["ats_resume"]:
        return state
    if current_step == "job_search" and not state["job_matches"]:
        return state
    if current_step == "staffing_agent" and not state["staffing_notes"]:
        return state

    flow = ["skill_endorser", "career_mentor", "resume_ai", "job_search", "staffing_agent", "done"]
    state["next_step"] = flow[flow.index(current_step) + 1]
    return state


def skill_endorser_agent(state: SkillForgeState, llm: ChatOpenAI) -> SkillForgeState:
    report = _invoke_agent(
        llm,
        (
            "You are Skill Endorser, an execution agent. Evaluate skill strengths, gaps, mock interview readiness, "
            "capstone criteria, and produce a certification recommendation."
        ),
        {
            "user_profile": state["user_profile"],
            "conversation": [m.content for m in state["conversation"]],
        },
    )
    state["diagnosis_report"] = report
    state["readiness_report"] = report
    state["conversation"].append(AIMessage(content=f"Skill Endorser report:\n{report}"))
    return state


def career_mentor_agent(state: SkillForgeState, llm: ChatOpenAI) -> SkillForgeState:
    plan = _invoke_agent(
        llm,
        (
            "You are Career Mentor, a planning agent. Build a 4-week dynamic learning plan with daily practice, "
            "coding tasks, quizzes, and weekly milestones aligned to the target role."
        ),
        {
            "user_profile": state["user_profile"],
            "diagnosis_report": state["diagnosis_report"],
        },
    )
    state["learning_plan"] = plan
    state["conversation"].append(AIMessage(content=f"Career Mentor plan:\n{plan}"))
    return state


def resume_ai_agent(state: SkillForgeState, llm: ChatOpenAI) -> SkillForgeState:
    optimized_resume = _invoke_agent(
        llm,
        (
            "You are Resume AI, an analytical agent. Perform ATS alignment, rewrite bullet points with measurable impact, "
            "and provide an improved resume summary tailored to the target role."
        ),
        {
            "resume": state["user_profile"].get("resume_text", ""),
            "target_skill": state["user_profile"].get("primary_skill", ""),
            "readiness_report": state["readiness_report"],
        },
    )
    state["ats_resume"] = optimized_resume
    state["conversation"].append(AIMessage(content=f"Resume AI output:\n{optimized_resume}"))
    return state


def job_search_agent(
    state: SkillForgeState,
    llm: ChatOpenAI,
    search_tool: DuckDuckGoSearchResults,
    query_suffix: str,
) -> SkillForgeState:
    role = state["user_profile"].get("primary_skill", "genai engineer")
    query = f"{role} jobs {query_suffix}"
    raw_search = search_tool.invoke(query)

    ranking = _invoke_agent(
        llm,
        (
            "You are Job Search, an analytical matchmaker. Parse the raw job search snippets and produce the top 5 "
            "job matches with title, company, location, reason_for_fit, and apply_url if available. "
            "Return valid JSON list only."
        ),
        {
            "user_profile": state["user_profile"],
            "ats_resume": state["ats_resume"],
            "raw_search": raw_search,
        },
    )

    try:
        jobs = json.loads(ranking)
        if not isinstance(jobs, list):
            jobs = [{"raw": ranking}]
    except json.JSONDecodeError:
        jobs = [{"raw": ranking}]

    state["job_matches"] = jobs
    state["conversation"].append(AIMessage(content=f"Job Search output:\n{ranking}"))
    return state


def staffing_agent(state: SkillForgeState, llm: ChatOpenAI) -> SkillForgeState:
    handoff = _invoke_agent(
        llm,
        (
            "You are Staffing Agent, a recruiter proxy execution agent. Produce recruiter handoff notes, candidate pitch, "
            "and an application tracking plan by job_id/title."
        ),
        {
            "user_profile": state["user_profile"],
            "job_matches": state["job_matches"],
            "readiness_report": state["readiness_report"],
        },
    )
    state["staffing_notes"] = handoff
    state["conversation"].append(AIMessage(content=f"Staffing Agent handoff:\n{handoff}"))
    return state


def next_node(state: SkillForgeState) -> str:
    return state["next_step"]


def build_skillforge_graph(config: SkillForgeOllamaConfig):
    llm = build_chat_model(config)
    search_tool = DuckDuckGoSearchResults(num_results=8)

    graph = StateGraph(SkillForgeState)
    graph.add_node("master", master_router)
    graph.add_node("skill_endorser", lambda s: skill_endorser_agent(s, llm))
    graph.add_node("career_mentor", lambda s: career_mentor_agent(s, llm))
    graph.add_node("resume_ai", lambda s: resume_ai_agent(s, llm))
    graph.add_node("job_search", lambda s: job_search_agent(s, llm, search_tool, config.query_suffix))
    graph.add_node("staffing_agent", lambda s: staffing_agent(s, llm))

    graph.set_entry_point("master")
    graph.add_conditional_edges(
        "master",
        next_node,
        {
            "skill_endorser": "skill_endorser",
            "career_mentor": "career_mentor",
            "resume_ai": "resume_ai",
            "job_search": "job_search",
            "staffing_agent": "staffing_agent",
            "done": END,
        },
    )
    graph.add_edge("skill_endorser", "master")
    graph.add_edge("career_mentor", "master")
    graph.add_edge("resume_ai", "master")
    graph.add_edge("job_search", "master")
    graph.add_edge("staffing_agent", "master")
    return graph.compile()


def run_skillforge_workflow() -> SkillForgeState:
    config = SkillForgeOllamaConfig.from_env()
    app = build_skillforge_graph(config)

    initial_state: SkillForgeState = {
        "user_profile": {
            "name": "Ada Johnson",
            "company_id": "COMP-1007",
            "primary_skill": "Python GenAI Engineer",
            "resume_text": (
                "Python developer with API integrations, prompt engineering basics, and practical "
                "experience in LLM app development."
            ),
        },
        "conversation": [
            SystemMessage(content="You are SkillForge AI master orchestrator."),
            HumanMessage(content="Guide me from assessment to placement."),
        ],
        "diagnosis_report": "",
        "learning_plan": "",
        "readiness_report": "",
        "ats_resume": "",
        "job_matches": [],
        "staffing_notes": "",
        "next_step": "skill_endorser",
    }

    final_state = app.invoke(initial_state)

    print("\n=== SkillForge (Ollama + Free Search) Outputs ===")
    print("\n[Diagnosis Report]\n", final_state["diagnosis_report"])
    print("\n[Learning Plan]\n", final_state["learning_plan"])
    print("\n[ATS Resume]\n", final_state["ats_resume"])
    print("\n[Job Matches]\n", json.dumps(final_state["job_matches"], indent=2, default=str))
    print("\n[Staffing Notes]\n", final_state["staffing_notes"])

    return final_state


if __name__ == "__main__":
    run_skillforge_workflow()

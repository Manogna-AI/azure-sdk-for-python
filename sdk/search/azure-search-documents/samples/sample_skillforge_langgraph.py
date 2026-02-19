# coding: utf-8

# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
FILE: sample_skillforge_langgraph.py
DESCRIPTION:
    SkillForge AI reference architecture built with LangGraph + LangChain,
    using Azure OpenAI for reasoning and Azure AI Search for job retrieval.

    The sample replaces Dialogflow-style flows with a multi-agent graph made of:
      1) Skill Endorser agent (diagnostic + readiness certification)
      2) Career Mentor agent (learning-plan planning and coaching)
      3) Resume AI agent (ATS-oriented resume optimization)
      4) Job Search agent (Azure AI Search retrieval + ranking prompt)
      5) Staffing agent (recruiter handoff + application tracking notes)

USAGE:
    python sample_skillforge_langgraph.py

    Set the environment variables with your own values before running the sample:
    1) AZURE_OPENAI_ENDPOINT
    2) AZURE_OPENAI_API_KEY
    3) AZURE_OPENAI_CHAT_DEPLOYMENT
    4) AZURE_OPENAI_API_VERSION
    5) AZURE_SEARCH_SERVICE_ENDPOINT
    6) AZURE_SEARCH_INDEX_NAME
    7) AZURE_SEARCH_API_KEY
"""

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, TypedDict

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from langgraph.graph import END, StateGraph


@dataclass
class SkillForgeConfig:
    """Central configuration for model and Azure service clients."""

    azure_openai_endpoint: str
    azure_openai_api_key: str
    azure_openai_chat_deployment: str
    azure_openai_api_version: str
    search_endpoint: str
    search_index_name: str
    search_api_key: str

    @classmethod
    def from_env(cls) -> "SkillForgeConfig":
        return cls(
            azure_openai_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            azure_openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_openai_chat_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"],
            azure_openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
            search_endpoint=os.environ["AZURE_SEARCH_SERVICE_ENDPOINT"],
            search_index_name=os.environ["AZURE_SEARCH_INDEX_NAME"],
            search_api_key=os.environ["AZURE_SEARCH_API_KEY"],
        )


class SkillForgeState(TypedDict):
    """Shared context passed between autonomous agents."""

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


def build_chat_model(config: SkillForgeConfig) -> AzureChatOpenAI:
    return AzureChatOpenAI(
        azure_endpoint=config.azure_openai_endpoint,
        api_key=config.azure_openai_api_key,
        azure_deployment=config.azure_openai_chat_deployment,
        api_version=config.azure_openai_api_version,
        temperature=0.2,
    )


def build_search_client(config: SkillForgeConfig) -> SearchClient:
    return SearchClient(
        endpoint=config.search_endpoint,
        index_name=config.search_index_name,
        credential=AzureKeyCredential(config.search_api_key),
    )


def _invoke_agent(
    llm: AzureChatOpenAI,
    system_prompt: str,
    user_payload: Dict[str, Any],
) -> str:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "human",
                "User profile and state payload:\n{payload}\n\nReturn concise but actionable output.",
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
    next_index = flow.index(current_step) + 1
    state["next_step"] = flow[next_index]
    return state


def skill_endorser_agent(state: SkillForgeState, llm: AzureChatOpenAI) -> SkillForgeState:
    report = _invoke_agent(
        llm,
        system_prompt=(
            "You are Skill Endorser, an execution agent. "
            "Evaluate strengths, identify skill gaps, propose mock interview and capstone checkpoints, "
            "and issue readiness level with objective criteria."
        ),
        user_payload={
            "user_profile": state["user_profile"],
            "conversation": [m.content for m in state["conversation"]],
        },
    )
    state["diagnosis_report"] = report
    state["readiness_report"] = report
    state["conversation"].append(AIMessage(content=f"Skill Endorser report:\n{report}"))
    return state


def career_mentor_agent(state: SkillForgeState, llm: AzureChatOpenAI) -> SkillForgeState:
    plan = _invoke_agent(
        llm,
        system_prompt=(
            "You are Career Mentor, a planning agent. "
            "Create adaptive weekly and daily learning sprints with measurable milestones, "
            "including coding labs, theory quizzes, and interview preparation tasks."
        ),
        user_payload={
            "user_profile": state["user_profile"],
            "diagnosis_report": state["diagnosis_report"],
        },
    )
    state["learning_plan"] = plan
    state["conversation"].append(AIMessage(content=f"Career Mentor plan:\n{plan}"))
    return state


def resume_ai_agent(state: SkillForgeState, llm: AzureChatOpenAI) -> SkillForgeState:
    updated_resume = _invoke_agent(
        llm,
        system_prompt=(
            "You are Resume AI, an analytical agent. "
            "Optimize resume content for ATS relevance, skill keyword density, and impact-based bullet points. "
            "Return a revised resume section plus improvement checklist."
        ),
        user_payload={
            "resume": state["user_profile"].get("resume_text", ""),
            "target_skill": state["user_profile"].get("primary_skill", ""),
            "readiness_report": state["readiness_report"],
        },
    )
    state["ats_resume"] = updated_resume
    state["conversation"].append(AIMessage(content=f"Resume AI output:\n{updated_resume}"))
    return state


def job_search_agent(
    state: SkillForgeState,
    llm: AzureChatOpenAI,
    search_client: SearchClient,
) -> SkillForgeState:
    skill = state["user_profile"].get("primary_skill", "Generative AI")
    search_results = search_client.search(
        search_text=skill,
        top=5,
        select=["job_id", "title", "location", "skills", "description"],
    )

    jobs: List[Dict[str, Any]] = []
    for item in search_results:
        jobs.append(
            {
                "job_id": item.get("job_id"),
                "title": item.get("title"),
                "location": item.get("location"),
                "skills": item.get("skills"),
                "description": item.get("description"),
            }
        )

    ranking = _invoke_agent(
        llm,
        system_prompt=(
            "You are Job Search, an analytical matchmaker agent. "
            "Rank retrieved jobs by alignment with candidate profile, certification readiness, "
            "and ATS-optimized resume strengths."
        ),
        user_payload={
            "user_profile": state["user_profile"],
            "ats_resume": state["ats_resume"],
            "jobs": jobs,
        },
    )

    state["job_matches"] = jobs
    state["conversation"].append(AIMessage(content=f"Job Search ranking:\n{ranking}"))
    return state


def staffing_agent(state: SkillForgeState, llm: AzureChatOpenAI) -> SkillForgeState:
    handoff = _invoke_agent(
        llm,
        system_prompt=(
            "You are Staffing Agent, an execution agent acting as recruiter proxy. "
            "Generate a recruiter handoff summary, outreach message draft, and application tracking checklist."
        ),
        user_payload={
            "user_profile": state["user_profile"],
            "readiness_report": state["readiness_report"],
            "job_matches": state["job_matches"],
        },
    )
    state["staffing_notes"] = handoff
    state["conversation"].append(AIMessage(content=f"Staffing Agent handoff:\n{handoff}"))
    return state


def next_node(state: SkillForgeState) -> str:
    return state["next_step"]


def build_skillforge_graph(config: SkillForgeConfig):
    llm = build_chat_model(config)
    search_client = build_search_client(config)

    graph = StateGraph(SkillForgeState)
    graph.add_node("master", master_router)
    graph.add_node("skill_endorser", lambda state: skill_endorser_agent(state, llm))
    graph.add_node("career_mentor", lambda state: career_mentor_agent(state, llm))
    graph.add_node("resume_ai", lambda state: resume_ai_agent(state, llm))
    graph.add_node("job_search", lambda state: job_search_agent(state, llm, search_client))
    graph.add_node("staffing_agent", lambda state: staffing_agent(state, llm))

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
    config = SkillForgeConfig.from_env()
    app = build_skillforge_graph(config)

    initial_state: SkillForgeState = {
        "user_profile": {
            "name": "Ada Johnson",
            "company_id": "COMP-1007",
            "primary_skill": "Python GenAI Engineer",
            "resume_text": (
                "Python developer with API integration background, basic prompt engineering, "
                "and interest in Azure AI services."
            ),
        },
        "conversation": [
            SystemMessage(content="You are SkillForge AI master orchestrator."),
            HumanMessage(content="I want a complete skill-to-job pipeline with certification and recruiter routing."),
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

    print("\n=== SkillForge AI Outputs ===")
    print("\n[Diagnosis Report]\n", final_state["diagnosis_report"])
    print("\n[Learning Plan]\n", final_state["learning_plan"])
    print("\n[ATS Resume]\n", final_state["ats_resume"])
    print("\n[Job Matches]\n", json.dumps(final_state["job_matches"], indent=2, default=str))
    print("\n[Staffing Notes]\n", final_state["staffing_notes"])

    return final_state


if __name__ == "__main__":
    run_skillforge_workflow()

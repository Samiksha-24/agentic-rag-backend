"""
crew_builder.py
================
Framework-agnostic CrewAI crew construction.

Pipeline (matches the diagram in the platform brief):

    User Query + conversation history
        -> Query Rewriter Agent   (resolves follow-ups: "when was it released?")
        -> Retriever Agent        (hybrid search tool: dense + BM25 + rerank)
        -> [Verifier Agent]       (research_mode only)
        -> Response Synthesizer Agent

This used to be Streamlit-only logic inside app_runtime.py. It's extracted
here so app_runtime.py (Streamlit UI) and agentic_rag.api (FastAPI backend)
build the exact same agents/tasks instead of maintaining two copies that
could drift apart.
"""

from __future__ import annotations

from typing import Callable, Optional

from crewai import Agent, Crew, Process, Task


def build_crew(
    pdf_tool,
    web_search_tool,
    llm,
    research_mode: bool = False,
    step_callback: Optional[Callable] = None,
    enable_query_rewriter: bool = True,
) -> Crew:
    """
    Build the AgenticRag crew.

    pdf_tool: a DocumentSearchTool / MultiDocumentSearchTool / HybridDocumentSearchTool instance (or None)
    web_search_tool: a web search tool instance (or None)
    llm: a configured LLM instance, or None to let crewai resolve the
         default LLM from environment variables (unchanged from original
         behavior)
    research_mode: if True, inserts a verification agent between retrieval
         and response synthesis
    step_callback: optional callable invoked on every agent step, used by
         callers to surface a live "agent reasoning" trace
    enable_query_rewriter: if True (default), inserts a query-rewriting
         agent before retrieval that uses conversation history — passed via
         the "history" key in kickoff(inputs=...) — to turn follow-up
         questions ("when was it released?") into standalone search
         queries. Callers using this must pass a "history" input
         (empty string is fine for a fresh conversation).
    """
    tools = [t for t in [pdf_tool, web_search_tool] if t]

    agents: list = []
    tasks: list = []

    if enable_query_rewriter:
        query_rewriter_agent = Agent(
            role="Rewrite the user query: {query} into a standalone search query",
            goal=(
                "Given the user's latest message: {query} and the recent conversation "
                "history: {history}, produce a single, precise, standalone search query "
                "that resolves pronouns and implicit references (e.g. 'it', 'that', 'the "
                "previous one', 'when was it released'). If the message is already "
                "standalone and needs no context, return it unchanged. Output ONLY the "
                "rewritten query text, nothing else."
            ),
            backstory=(
                "You're an expert at conversational context. You never answer the "
                "question yourself — you only reformulate it into the clearest possible "
                "standalone search query for a retrieval system."
            ),
            verbose=True,
            llm=llm,
            step_callback=step_callback,
        )
        query_rewriter_task = Task(
            description=(
                "Rewrite the user's latest message: {query} into a standalone search "
                "query, using this conversation history for context: {history}"
            ),
            expected_output="A single rewritten, standalone search query string. Nothing else.",
            agent=query_rewriter_agent,
        )
        agents.append(query_rewriter_agent)
        tasks.append(query_rewriter_task)
        retrieval_context = [query_rewriter_task]
        retrieval_description = (
            "Using the standalone query produced by the previous step (fall back to "
            "the original user message: {query} if no rewrite was needed), retrieve "
            "the most relevant information from the available sources."
        )
    else:
        retrieval_context = []
        retrieval_description = (
            "Retrieve the most relevant information from the available sources for "
            "the user query: {query}"
        )

    retriever_agent = Agent(
        role="Retrieve relevant information to answer the user query: {query}",
        goal=(
            "Retrieve the most relevant information from the available sources "
            "for the user query: {query}. Always try to use the document search tool first. "
            "If you are not able to retrieve the information from the documents, "
            "then try to use the web search tool."
        ),
        backstory=(
            "You're a meticulous analyst with a keen eye for detail. "
            "You're known for your ability to understand user queries: {query} "
            "and retrieve knowledge from the most suitable knowledge base."
        ),
        verbose=True,
        tools=tools,
        llm=llm,
        step_callback=step_callback,
    )

    retrieval_task = Task(
        description=retrieval_description,
        expected_output="The most relevant information in the form of text as retrieved from the sources.",
        agent=retriever_agent,
        context=retrieval_context,
    )
    agents.append(retriever_agent)
    tasks.append(retrieval_task)

    if research_mode:
        verification_agent = Agent(
            role="Verify and cross-check retrieved information for the query: {query}",
            goal=(
                "Critically review the retrieved information for the query: {query}. "
                "Flag any claims that are unsupported, contradictory, or low-confidence. "
                "Note explicitly which claims came from documents vs. the web."
            ),
            backstory="You're a rigorous fact-checker who never lets an unverified claim pass without a note of caution.",
            verbose=True,
            llm=llm,
            step_callback=step_callback,
        )
        verification_task = Task(
            description="Verify the retrieved information for the query: {query} and flag any low-confidence claims.",
            expected_output="A short verification note listing confidence and source type for each key claim.",
            agent=verification_agent,
            context=[retrieval_task],
        )
        tasks.append(verification_task)
        agents.append(verification_agent)

    response_synthesizer_agent = Agent(
        role="Response synthesizer agent for the user query: {query}",
        goal=(
            "Synthesize the retrieved (and, if available, verified) information into a concise "
            "and coherent response based on the user query: {query}. If you are not able to retrieve the "
            'information then respond with "I\'m sorry, I couldn\'t find the information you\'re looking for."'
        ),
        backstory="You're a skilled communicator with a knack for turning complex information into clear and concise responses.",
        verbose=True,
        llm=llm,
        step_callback=step_callback,
    )
    response_task = Task(
        description="Synthesize the final response for the user query: {query}",
        expected_output=(
            "A concise and coherent response based on the retrieved information "
            "from the right source for the user query: {query}. If you are not "
            "able to retrieve the information, then respond with: "
            '"I\'m sorry, I couldn\'t find the information you\'re looking for."'
        ),
        agent=response_synthesizer_agent,
        context=tasks,
    )
    tasks.append(response_task)
    agents.append(response_synthesizer_agent)

    return Crew(agents=agents, tasks=tasks, process=Process.sequential, verbose=True)

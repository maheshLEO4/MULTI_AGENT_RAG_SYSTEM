from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any
from langchain_core.documents import Document
import logging

from .research_agent import ResearchAgent
from .verification_agent import VerificationAgent
from .relevance_checker import RelevanceChecker

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ---------------------------
# Agent State Definition
# ---------------------------
class AgentState(TypedDict):
    question: str
    documents: List[Document]
    draft_answer: str
    verification_report: str
    is_relevant: bool
    retriever: Any  # Custom hybrid retriever
    iteration_count: int  # Track iterations to prevent infinite loops
    enable_verification: bool  # Toggle verification for speed


# ---------------------------
# Workflow Class
# ---------------------------
class AgentWorkflow:
    def __init__(self, enable_verification: bool = False):
        """
        Initialize workflow.
        
        Args:
            enable_verification: If True, runs verification (slower but more accurate).
                               If False, skips verification (faster).
        """
        self.researcher = ResearchAgent()
        self.enable_verification = enable_verification
        
        # Only initialize verification components if needed
        if enable_verification:
            self.verifier = VerificationAgent()
            self.relevance_checker = RelevanceChecker()
        else:
            self.verifier = None
            self.relevance_checker = None
            
        self.compiled_workflow = self._build_workflow()

    # ---------------------------
    # Build LangGraph Workflow
    # ---------------------------
    def _build_workflow(self):
        workflow = StateGraph(AgentState)

        if self.enable_verification:
            # Full workflow with verification
            workflow.add_node("check_relevance", self._check_relevance_step)
            workflow.add_node("research", self._research_step)
            workflow.add_node("verify", self._verification_step)

            workflow.set_entry_point("check_relevance")

            workflow.add_conditional_edges(
                "check_relevance",
                self._decide_after_relevance_check,
                {
                    "relevant": "research",
                    "irrelevant": END
                }
            )

            workflow.add_edge("research", "verify")

            workflow.add_conditional_edges(
                "verify",
                self._decide_next_step,
                {
                    "re_research": "research",
                    "end": END
                }
            )
        else:
            # Fast workflow - skip verification
            workflow.add_node("research", self._research_step)
            workflow.set_entry_point("research")
            workflow.add_edge("research", END)

        return workflow.compile()

    # ---------------------------
    # Relevance Check Step
    # ---------------------------
    def _check_relevance_step(self, state: AgentState) -> Dict:
        retriever = state["retriever"]
        question = state["question"]

        logger.info("Checking question relevance")

        try:
            classification = self.relevance_checker.check(
                question=question,
                retriever=retriever,
                k=20
            )
        except Exception as e:
            logger.error(f"Error in relevance checking: {e}")
            return {
                "is_relevant": False,
                "draft_answer": "❌ An error occurred while checking relevance. Please try again."
            }

        if classification in ("CAN_ANSWER", "PARTIAL"):
            return {"is_relevant": True}

        return {
            "is_relevant": False,
            "draft_answer": (
                "❌ The question is not related to the uploaded documents "
                "or there is insufficient information to answer it."
            )
        }

    def _decide_after_relevance_check(self, state: AgentState) -> str:
        decision = "relevant" if state["is_relevant"] else "irrelevant"
        logger.info(f"Relevance decision: {decision}")
        return decision

    # ---------------------------
    # Public Pipeline Entry
    # ---------------------------
    def full_pipeline(self, question: str, retriever: Any) -> Dict[str, str]:
        try:
            logger.info(f"Starting workflow for question: {question}")

            # Use invoke() to retrieve documents
            try:
                documents = retriever.invoke(question)
            except Exception as e:
                logger.error(f"Error retrieving documents: {e}")
                return {
                    "draft_answer": "❌ An error occurred while retrieving documents. Please ensure PDFs are properly indexed.",
                    "verification_report": ""
                }

            logger.info(f"Retrieved {len(documents)} documents")

            initial_state: AgentState = {
                "question": question,
                "documents": documents,
                "draft_answer": "",
                "verification_report": "⚡ Verification disabled for faster responses" if not self.enable_verification else "",
                "is_relevant": True,  # Skip check if verification disabled
                "retriever": retriever,
                "iteration_count": 0,
                "enable_verification": self.enable_verification
            }

            try:
                final_state = self.compiled_workflow.invoke(initial_state)
            except Exception as e:
                logger.error(f"Error in workflow execution: {e}")
                return {
                    "draft_answer": "❌ An error occurred during the workflow execution. Please try again.",
                    "verification_report": ""
                }

            return {
                "draft_answer": final_state.get("draft_answer", ""),
                "verification_report": final_state.get("verification_report", "")
            }

        except Exception as e:
            logger.exception("❌ Workflow execution failed")
            return {
                "draft_answer": f"❌ An unexpected error occurred: {str(e)}",
                "verification_report": ""
            }

    # ---------------------------
    # Research Step
    # ---------------------------
    def _research_step(self, state: AgentState) -> Dict:
        logger.info("Running research agent")

        try:
            result = self.researcher.generate(
                question=state["question"],
                documents=state["documents"]
            )
        except Exception as e:
            logger.error(f"Error in research step: {e}")
            return {
                "draft_answer": "❌ An error occurred while generating the answer."
            }

        # Increment iteration count
        iteration_count = state.get("iteration_count", 0) + 1

        return {
            "draft_answer": result.get("draft_answer", ""),
            "iteration_count": iteration_count
        }

    # ---------------------------
    # Verification Step
    # ---------------------------
    def _verification_step(self, state: AgentState) -> Dict:
        logger.info("Running verification agent")

        try:
            result = self.verifier.check(
                answer=state["draft_answer"],
                documents=state["documents"]
            )
        except Exception as e:
            logger.error(f"Error in verification step: {e}")
            return {
                "verification_report": "❌ An error occurred during verification."
            }

        return {
            "verification_report": result.get("verification_report", "")
        }

    # ---------------------------
    # Decide Loop or End
    # ---------------------------
    def _decide_next_step(self, state: AgentState) -> str:
        report = state.get("verification_report", "")
        iteration_count = state.get("iteration_count", 0)
        
        # Prevent infinite loops - max 2 iterations
        if iteration_count >= 2:
            logger.info("Maximum iterations reached → ending workflow")
            return "end"

        # Check if verification failed
        if "Supported: NO" in report or "Relevant: NO" in report:
            logger.info("Verification failed → re-running research")
            return "re_research"

        logger.info("Verification successful → ending workflow")
        return "end"
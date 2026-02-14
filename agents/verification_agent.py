from langchain_groq import ChatGroq
from typing import Dict, List
from langchain_core.documents import Document
import os

class VerificationAgent:
    def __init__(self):
        """
        Initialize the verification agent with Groq LLM.
        """
        print("Initializing VerificationAgent with Groq LLM...")
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set")
        
        self.llm = ChatGroq(
            model_name="llama-3.1-8b-instant",
            temperature=0.0,
            max_tokens=200,
            groq_api_key=api_key,
        )
        print("LLM initialized successfully.")

    def sanitize_response(self, response_text: str) -> str:
        """
        Sanitize the LLM's response by stripping unnecessary whitespace.
        """
        return response_text.strip()

    def generate_prompt(self, answer: str, context: str) -> str:
        """
        Generate a structured prompt for the LLM to verify the answer against the context.
        """
        prompt = f"""
        You are an AI assistant designed to verify the accuracy and relevance of answers based on provided context.

        **Instructions:**
        - Verify the following answer against the provided context.
        - Check for:
        1. Direct/indirect factual support (YES/NO)
        2. Unsupported claims (list any if present)
        3. Contradictions (list any if present)
        4. Relevance to the question (YES/NO)
        - Provide additional details or explanations where relevant.
        - Respond in the exact format specified below without adding any unrelated information.

        **Format:**
        Supported: YES/NO
        Unsupported Claims: [item1, item2, ...]
        Contradictions: [item1, item2, ...]
        Relevant: YES/NO
        Additional Details: [Any extra information or explanations]

        **Answer:** {answer}
        **Context:**
        {context}

        **Respond ONLY with the above format.**
        """
        return prompt

    def parse_verification_response(self, response_text: str) -> Dict:
        """
        Parse the LLM's verification response into a structured dictionary.
        """
        try:
            lines = response_text.split('\n')
            verification = {}
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Normalize key names
                    key_lower = key.lower()
                    if "supported" in key_lower and "unsupported" not in key_lower:
                        verification["Supported"] = value.upper()
                    elif "unsupported" in key_lower:
                        if value.startswith('[') and value.endswith(']'):
                            items = value[1:-1].split(',')
                            items = [item.strip().strip('"').strip("'") for item in items if item.strip()]
                            verification["Unsupported Claims"] = items
                        else:
                            verification["Unsupported Claims"] = []
                    elif "contradiction" in key_lower:
                        if value.startswith('[') and value.endswith(']'):
                            items = value[1:-1].split(',')
                            items = [item.strip().strip('"').strip("'") for item in items if item.strip()]
                            verification["Contradictions"] = items
                        else:
                            verification["Contradictions"] = []
                    elif "relevant" in key_lower:
                        verification["Relevant"] = value.upper()
                    elif "additional" in key_lower or "details" in key_lower:
                        verification["Additional Details"] = value
            
            # Ensure all keys are present
            if "Supported" not in verification:
                verification["Supported"] = "NO"
            if "Unsupported Claims" not in verification:
                verification["Unsupported Claims"] = []
            if "Contradictions" not in verification:
                verification["Contradictions"] = []
            if "Relevant" not in verification:
                verification["Relevant"] = "NO"
            if "Additional Details" not in verification:
                verification["Additional Details"] = ""

            return verification
        except Exception as e:
            print(f"Error parsing verification response: {e}")
            return None

    def format_verification_report(self, verification: Dict) -> str:
        """
        Format the verification report dictionary into a readable paragraph.
        """
        supported = verification.get("Supported", "NO")
        unsupported_claims = verification.get("Unsupported Claims", [])
        contradictions = verification.get("Contradictions", [])
        relevant = verification.get("Relevant", "NO")
        additional_details = verification.get("Additional Details", "")

        report = f"**Supported:** {supported}\n"
        if unsupported_claims:
            report += f"**Unsupported Claims:** {', '.join(unsupported_claims)}\n"
        else:
            report += f"**Unsupported Claims:** None\n"

        if contradictions:
            report += f"**Contradictions:** {', '.join(contradictions)}\n"
        else:
            report += f"**Contradictions:** None\n"

        report += f"**Relevant:** {relevant}\n"

        if additional_details:
            report += f"**Additional Details:** {additional_details}\n"
        else:
            report += f"**Additional Details:** None\n"

        return report

    def check(self, answer: str, documents: List[Document]) -> Dict:
        """
        Verify the answer against the provided documents.
        """
        print(f"VerificationAgent.check called with answer='{answer}' and {len(documents)} documents.")

        if not documents:
            print("No documents provided for verification.")
            verification_report = {
                "Supported": "NO",
                "Unsupported Claims": [],
                "Contradictions": [],
                "Relevant": "NO",
                "Additional Details": "No documents available for verification."
            }
            verification_report_formatted = self.format_verification_report(verification_report)
            return {
                "verification_report": verification_report_formatted,
                "context_used": ""
            }

        # Combine all document contents into one string without truncation
        context = "\n\n".join([doc.page_content for doc in documents])
        print(f"Combined context length: {len(context)} characters.")

        # Create a prompt for the LLM to verify the answer
        prompt = self.generate_prompt(answer, context)
        print("Prompt created for the LLM.")

        # Call the LLM to generate the verification report
        try:
            print("Sending prompt to the model...")
            response = self.llm.invoke(prompt)
            print("LLM response received.")
        except Exception as e:
            print(f"Error during model inference: {e}")
            verification_report = {
                "Supported": "NO",
                "Unsupported Claims": [],
                "Contradictions": [],
                "Relevant": "NO",
                "Additional Details": f"Model error: {str(e)}"
            }
            verification_report_formatted = self.format_verification_report(verification_report)
            return {
                "verification_report": verification_report_formatted,
                "context_used": context
            }

        # Extract and process the LLM's response
        try:
            llm_response = response.content.strip()
            print(f"Raw LLM response:\n{llm_response}")
        except Exception as e:
            print(f"Unexpected response structure: {e}")
            verification_report = {
                "Supported": "NO",
                "Unsupported Claims": [],
                "Contradictions": [],
                "Relevant": "NO",
                "Additional Details": "Invalid response structure from the model."
            }
            verification_report_formatted = self.format_verification_report(verification_report)
            print(f"Verification report:\n{verification_report_formatted}")
            print(f"Context used: {context}")
            return {
                "verification_report": verification_report_formatted,
                "context_used": context
            }

        # Sanitize the response
        sanitized_response = self.sanitize_response(llm_response) if llm_response else ""
        if not sanitized_response:
            print("LLM returned an empty response.")
            verification_report = {
                "Supported": "NO",
                "Unsupported Claims": [],
                "Contradictions": [],
                "Relevant": "NO",
                "Additional Details": "Empty response from the model."
            }
        else:
            # Parse the response into the expected format
            verification_report = self.parse_verification_response(sanitized_response)
            if verification_report is None:
                print("LLM did not respond with the expected format. Using default verification report.")
                verification_report = {
                    "Supported": "NO",
                    "Unsupported Claims": [],
                    "Contradictions": [],
                    "Relevant": "NO",
                    "Additional Details": "Failed to parse the model's response."
                }

        # Format the verification report into a paragraph
        verification_report_formatted = self.format_verification_report(verification_report)
        print(f"Verification report:\n{verification_report_formatted}")
        print(f"Context used: {context}")

        return {
            "verification_report": verification_report_formatted,
            "context_used": context
        }
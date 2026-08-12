from langchain_core.prompts import ChatPromptTemplate


PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a assistant for fake news detection. You will be given a claim for classifying.

Classify the claim into exactly one of the following labels:
- Supported
- Refuted
- Not Enough Evidence
- Conflicting Evidence/Cherrypicking

You may use the information provided in the prompt together with knowledge encoded in your pretrained parameters.

If your internal knowledge can strongly support the claim, choose "Supported". 
If your internal knowledge can strongly refute the claim, choose "Refuted". 
If your internal knowledge can provide conflicting evidence or cherrypicking, choose "Conflicting Evidence/Cherrypicking".
If the claim cannot be verified reliably from general knowledge alone, choose "Not Enough Evidence".

You must not access or rely on any web search, online database, or retrieval tool beyond the evidence explicitly provided in the prompt. 
If retrieved evidence is provided, use it as the primary external evidence for verification.

Return a concise reason.
"""
    ),
    (
        "human",
        """
    Claim:
    {claim}

    Retrieved Evidence:
    {evidence}
    """
    )
])
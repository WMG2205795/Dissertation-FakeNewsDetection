from langchain_ollama import ChatOllama

from src.schemas import VerificationOutput


def load_open_llm(
    model_name,
    temperature,
):
    llm = ChatOllama(
        model=model_name,
        temperature=temperature,
        top_p=1.0,
        top_k=0,
        seed=42
    )

    return llm.with_structured_output(VerificationOutput)
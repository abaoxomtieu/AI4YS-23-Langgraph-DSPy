from langgraph.graph import START, END, StateGraph
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.output_parsers import BooleanOutputParser
from config import llm, search_engine
from fastapi import FastAPI
from typing import TypedDict, Annotated
import operator

boolean_parser = BooleanOutputParser()
vector_db_path = "vectorstores/db_faiss"
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
db = FAISS.load_local(
    vector_db_path, embedding_model, allow_dangerous_deserialization=True
)


class AgentState(TypedDict):
    input: str
    answer: Annotated[list[str], operator.add]
    context: Annotated[list[str], operator.add]
    search_engine: Annotated[list[str], operator.add]
    query: Annotated[list[str], operator.add]
    review: Annotated[list[str], operator.add]


def query(state: AgentState):
    print("query")
    message = state["input"]
    search_engine_data = state.get("search_engine", [])
    if search_engine_data:
        prompt = (
            """My system is a Retrieval-Augmented Generation (RAG) based question-answering system.
            I need to generate a natural language query to retrieve relevant context from a FAISS vector store.
            Here is the question:
            """
            + message
            + "\nAdditionally, I have data from Tailvy search: "
            + search_engine_data[-1]
            + "\nGenerate a concise natural language query string based on this information."
        )
    else:
        prompt = (
            """My system is a Retrieval-Augmented Generation (RAG) based question-answering system. 
            I need to generate a natural language query to retrieve relevant context from a FAISS vector store.
            Here is the question: 
            """
            + message
            + "\n Generate a concise natural language query string based on this information."
        )

    result = llm.invoke(prompt)
    return {"query": [result]}


def retrive(state: AgentState):
    print("retrieve")
    query = state["query"][-1]
    context = db.similarity_search_with_relevance_scores(query, k=5)
    context = [item for item in context if item[1] > 0.2]
    # print(context)
    context = "\n".join([i[0].page_content for i in context])

    return {"context": [context]}


def answer(state: AgentState):
    print("answer")
    context = state["context"][-1]
    search_engine_data = state.get("search_engine", [])
    search_data = search_engine_data[-1] if search_engine_data else ""

    prompt = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise."""
    if search_data:
        prompt += "\nContext from Tailvy search engine: " + search_data

    prompt += (
        "\nContext retrieved is: "
        + context
        + "\nQuestion: "
        + state["input"]
        + "\nAnswer:"
    )

    answer = llm.invoke(prompt)
    return {"answer": [answer]}


def review(state: AgentState):
    print("review")
    context = state["context"][-1]
    answer = state["answer"][-1]
    input = state["input"]

    prompt = (
        "My context is: "
        + context
        + "\nMy answer is: "
        + answer
        + "\nIs this context and answer enough to satisfactorily answer the question: "
        + input
        + "? \nRespond with 'Yes' if it is enough, otherwise respond with 'No'."
    )

    review_result = llm.invoke(prompt)
    review = boolean_parser.parse(review_result)
    return {"review": [review]}


def search(state: AgentState):
    print("search")
    input = state["input"]
    search_data = search_engine.invoke(input)
    search_data = "\n".join([i["content"] for i in search_data])
    return {"search_engine": [search_data]}


def should_continue(state: AgentState):
    print("should_continue")
    if len(state["search_engine"]) > 2:
        return "end"
    return "search" if not state["review"][-1] else "end"


QUERY = "query_node"
RETRIEVE = "retrive_nnode"
ANSWER = "answer_node"
REVIEW = "review_node"
SEARCH = "search_node"

workflow = StateGraph(AgentState)
workflow.add_node(QUERY, query)
workflow.add_node(RETRIEVE, retrive)
workflow.add_node(ANSWER, answer)
workflow.add_node(REVIEW, review)
workflow.add_node(SEARCH, search)

workflow.add_edge(QUERY, RETRIEVE)
workflow.add_edge(RETRIEVE, ANSWER)
workflow.add_edge(ANSWER, REVIEW)
workflow.add_conditional_edges(REVIEW, should_continue, {"search": SEARCH, "end": END})
workflow.add_edge(SEARCH, QUERY)
workflow.add_edge(START, QUERY)

graph = workflow.compile()
graph.get_graph().draw_mermaid_png(output_file_path="graph.png")

answer: AgentState = graph.invoke(
    input={
        "input": "Who is Ton Bao Ho in this paper?",
    }
)
print("answer", answer["answer"][-1])


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(docs_url="/")

app.add_middleware(CORSMiddleware, allow_origins=["*"])


@app.post("/answer")
async def get_answer(message: str):
    answer = graph.invoke(input={"input": message})["answer"][-1]
    print("answer", answer)
    return {"answer": answer}

from flask import Blueprint, render_template, request
from langchain_core.messages import BaseMessage
from src.services.rag_chain import rag_chain

chat_bp = Blueprint("chat", __name__)

@chat_bp.route("/")
def index():
    return render_template("chat.html")


@chat_bp.route("/get", methods=["POST"])
def chat():
    try:
        msg = request.form["msg"]
        print("User input:", msg)

        response = rag_chain.invoke({"input": msg})
        answer = response.get("answer")
        print("answer:", answer)

        if isinstance(answer, BaseMessage):
            answer = answer.content

        return str(answer)

    except Exception as e:
        print("Error:", e)
        if "quota" in str(e).lower():
            return "⚠️ Error: You have exceeded your OpenAI quota. Please check your API billing."
        return "⚠️ Error: Something went wrong. Please try again later."

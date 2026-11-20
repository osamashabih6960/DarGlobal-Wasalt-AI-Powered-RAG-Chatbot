
"""Gradio interface for the DarGlobal x Wasalt RAG chatbot."""

from uuid import uuid4

import gradio as gr

from app.graph import ask
from app.main import get_graph


def respond(message, history, session_id):
    """Send the user message to the existing RAG graph."""

    if not message or not message.strip():
        return "", history

    try:
        result = ask(
            get_graph(),
            message.strip(),
            session_id,
        )

        answer = result["answer"]
        sources = result.get("sources", [])

        if sources:
            answer += "\n\n**Sources:**\n"
            for source in sources:
                title = source.get("title") or source["url"]
                answer += f"- [{title}]({source['url']})\n"

    except Exception as exc:
        answer = (
            "Sorry, something went wrong. "
            f"Error: {type(exc).__name__}"
        )

    updated_history = (history or []) + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer},
    ]

    return "", updated_history


with gr.Blocks(title="DarGlobal x Wasalt Assistant") as demo:

    gr.Markdown(
        "# DarGlobal x Wasalt AI Assistant\n"
        "Ask questions about DarGlobal and Wasalt."
    )

    session_id = gr.State(value=lambda: str(uuid4()))

    chatbot = gr.Chatbot(
        height=500,
        label="AI Assistant",
    )

    with gr.Row():
        message = gr.Textbox(
            placeholder="Ask your question...",
            show_label=False,
            scale=8,
        )

        send = gr.Button(
            "Send",
            variant="primary",
            scale=1,
        )

    clear = gr.Button("Clear Chat")

    send.click(
        respond,
        inputs=[message, chatbot, session_id],
        outputs=[message, chatbot],
    )

    message.submit(
        respond,
        inputs=[message, chatbot, session_id],
        outputs=[message, chatbot],
    )

    clear.click(
        lambda: ([], str(uuid4())),
        outputs=[chatbot, session_id],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7861,
        inbrowser=True,
    )

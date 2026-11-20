
"""Gradio interface for the DarGlobal x Wasalt RAG chatbot."""

import os
from uuid import uuid4

import gradio as gr

from app.graph import ask
from app.main import get_graph


def respond(message, history, session_id):
    """Send the user message to the existing RAG graph."""

    history = history or []

    if not message or not message.strip():
        return "", history

    if not session_id:
        session_id = str(uuid4())

    try:
        result = ask(
            get_graph(),
            message.strip(),
            session_id,
        )

        answer = result.get(
            "answer",
            "No answer was generated.",
        )

        sources = result.get("sources", [])

        if sources:
            answer += "\n\n**Sources:**\n"

            for source in sources:
                url = source.get("url")
                title = source.get("title") or url

                if url:
                    answer += f"- [{title}]({url})\n"

    except Exception as exc:
        answer = (
            "Sorry, something went wrong while processing "
            "your question. Please try again.\n\n"
            f"Error: {type(exc).__name__}"
        )

    updated_history = history + [
        {
            "role": "user",
            "content": message.strip(),
        },
        {
            "role": "assistant",
            "content": answer,
        },
    ]

    return "", updated_history


def clear_chat():
    """Clear conversation and generate a fresh session ID."""
    return [], str(uuid4())


with gr.Blocks(
    title="DarGlobal x Wasalt AI Assistant",
) as demo:

    gr.Markdown(
        """
        # 🏡 DarGlobal × Wasalt AI Assistant

        Your AI-powered real estate assistant.

        Ask questions about DarGlobal projects, Wasalt listings,
        property information and real estate services.
        """
    )

    session_id = gr.State(value=lambda: str(uuid4()))

    chatbot = gr.Chatbot(
        height=500,
        label="AI Assistant",
        show_copy_button=True,
    )

    with gr.Row():
        message = gr.Textbox(
            placeholder="Ask your question...",
            show_label=False,
            scale=8,
            lines=1,
            max_lines=4,
        )

        send = gr.Button(
            "Send",
            variant="primary",
            scale=1,
        )

    clear = gr.Button(
        "Clear Chat",
        variant="secondary",
    )

    send.click(
        fn=respond,
        inputs=[message, chatbot, session_id],
        outputs=[message, chatbot],
    )

    message.submit(
        fn=respond,
        inputs=[message, chatbot, session_id],
        outputs=[message, chatbot],
    )

    clear.click(
        fn=clear_chat,
        inputs=[],
        outputs=[chatbot, session_id],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(
            os.getenv("GRADIO_SERVER_PORT", "7860")
        ),
        theme=gr.themes.Soft(),
    )

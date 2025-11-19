import requests
import gradio as gr

API_URL = "http://localhost:8000/ask"


def chat_fn(message, history):
    try:
        resp = requests.post(API_URL, json={"question": message}, timeout=30)
    except Exception as e:
        return f"Error contacting API: {e}"

    if resp.status_code != 200:
        return f"API error: {resp.status_code} - {resp.text}"

    data = resp.json()
    answer = data["answer"]
    sources = data.get("sources", [])

    if sources:
        sources_text = "\n\nSources:\n" + "\n".join(
            f"- {s.get('source_file', 'unknown')} (page {s.get('page', 'N/A')})"
            for s in sources
        )
    else:
        sources_text = "\n\n(No sources returned.)"

    return answer + sources_text


demo = gr.ChatInterface(
    fn=chat_fn,
    title="🏙️ Urban Regulations RAG Assistant",
    description="Ask about zoning / building rules based on the loaded PDFs.",
)

if __name__ == "__main__":
    demo.launch()

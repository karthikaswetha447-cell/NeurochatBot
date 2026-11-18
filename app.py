
    with gr.Row():

        # LEFT SIDE: CHATBOT AND EVALUATION REPORT
        with gr.Column(scale=3):

            chatbot = gr.Chatbot(
                label="NeuroBot Conversation",
                type="messages",
                height=550
            )

            with gr.Row():
                message_input = gr.Textbox(
                    placeholder="Ask NeuroBot a question...",
                    show_label=False,
                    lines=2,
                    scale=5
                )

                submit_button = gr.Button(
                    "Send",
                    variant="primary",
                    scale=1
                )

            with gr.Accordion("LLM Parameters", open=False):
                temperature_input = gr.Slider(
                    minimum=0,
                    maximum=2,
                    value=0.7,
                    step=0.1,
                    label="Temperature"
                )

                max_tokens_input = gr.Slider(
                    minimum=128,
                    maximum=4096,
                    value=1024,
                    step=128,
                    label="Max Output Tokens"
                )

                top_p_input = gr.Slider(
                    minimum=0,
                    maximum=1,
                    value=1.0,
                    step=0.05,
                    label="Top-P"
                )

        # RIGHT SIDE: STUDENT CLASSIFICATION PANEL
        with gr.Column(scale=1):
            gr.Markdown("## 🎯 Student Classification")

            classification_panel = gr.Markdown(
                "Waiting for a student message..."
            )

    chatbot_inputs = [
        message_input,
        chatbot,
        temperature_input,
        max_tokens_input,
        top_p_input
    ]

    chatbot_outputs = [
        chatbot,
        message_input,
        classification_panel
    ]

    submit_button.click(
        fn=neurobot_with_panel,
        inputs=chatbot_inputs,
        outputs=chatbot_outputs
    )

    message_input.submit(
        fn=neurobot_with_panel,
        inputs=chatbot_inputs,
        outputs=chatbot_outputs
    )


# ==========================================
# 17. START NEUROBOT ON RENDER
# ==========================================

if __name__ == "__main__":

    print("======================================")
    print("NEUROBOT STARTING")
    print("======================================")

    port = int(os.environ.get("PORT", 10000))

    print(f"Starting NeuroBot on port: {port}")

    demo.launch(
        server_name="0.0.0.0",
        server_port=port
    )

import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer, set_seed
import soundfile as sf
from http.server import BaseHTTPRequestHandler, HTTPServer
import urllib.parse
import os

PORT = os.environ.get('PORT', 8081)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-expresso").to(device)
tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-expresso")


class TTSHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        # Serve the main HTML page
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b"""
                <html>
                    <head>
                        <meta charset="UTF-8">
                        <meta name="viewport" content="width=device-width, initial-scale=1.0">
                        <title>Parler-TTS</title>
                        <style>
                             @import url('https://fonts.googleapis.com/css2?family=Quicksand:wght@300..700&display=swap');
                             body{
                                font-family: "Quicksand", sans-serif;
                                font-optical-sizing: auto;
                                font-style: normal;
                                font-weight: 600;
                             
                                margin-top: 150px;
                                display: flex;
                                align-items: center;
                                flex-direction: column;
                            }
                            h4{
                                margin: 20px;
                            }
                            input[type="submit"]{
                                padding: 10px 12px;
                                background: rgb(7 14 15);
                                color: #fff;
                                cursor: pointer;
                                margin: 10px;
                                border: none;
                                border-radius: 3px;
                                letter-spacing: 0.9px;
                            }
                            input[type="text"]{
                                width: 300px;
                                height: 40px;
                                margin: 20px;
                            }
                            .hide{
                                display: none !important;
                            }
                             .loader {
                                width: 12px;
                                height: 12px;
                                border-radius: 50%;
                                display: block;
                                margin:15px auto;
                                position: relative;
                                color: #000;
                                box-sizing: border-box;
                                animation: animloader 1s linear infinite alternate;
                                }

                                @keyframes animloader {
                                0% {
                                    box-shadow: -38px -6px, -14px 6px,  14px -6px;
                                }
                                33% {
                                    box-shadow: -38px 6px, -14px -6px,  14px 6px;
                                }
                                66% {
                                    box-shadow: -38px -6px, -14px 6px, 14px -6px;
                                }
                                100% {
                                    box-shadow: -38px 6px, -14px -6px, 14px 6px;
                                }
                                }
                        </style>
                    </head>
                    <body>
                        <h1>Text to Speech AI generator</h1>
                        <h4>Powered by Akash</h4>
                        <form action="/tts" method="post">
                            <input type="text" name="text" placeholder="Enter text">
                            <input type="submit" value="Send">
                        </form>
                        <audio controls id="audio" class="hide"></audio>
                        <div class="loader hide" id="load"></div>
                        <script>
                            const audioTag = document.getElementById('audio');
                            const loader = document.getElementById('load');
                            async function handleFormSubmit(event) {
                                event.preventDefault();
                                if (audioTag.classList !== "hide")
                                audioTag.classList.add("hide");
                             
                                loader.classList.remove("hide");
                                const form = event.target;
                                const formData = new FormData(form);
                                try {
                                    const response = await fetch(form.action, {
                                        method: form.method,
                                        body: new URLSearchParams(formData)
                                    });
                                    if (!response.ok) throw new Error('Network response was not ok');
                                    const blob = await response.blob();
                                    const audioUrl = URL.createObjectURL(blob);
                                    audioTag.src = audioUrl;
                                    audioTag.classList.remove("hide");
                                    loader.classList.add("hide");
                                    audioTag.play();
                                } catch (error) {
                                    console.error('Fetch error: ', error);
                                }
                            }

                            document.querySelector('form').addEventListener('submit', handleFormSubmit);
                        </script>
                    </body>
                </html>
            """)

    def do_POST(self):
        if self.path == '/tts':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            post_data = urllib.parse.parse_qs(post_data.decode('utf-8'))
            text = post_data.get('text', [''])[0]

            print(f"Received text: {text}")  # Debugging: Print received text

            # Generate the TTS audio file
            audio_file_path = 'output.wav'
            self.text_to_speech(text, audio_file_path)

            # Check if the file was created
            if not os.path.exists(audio_file_path):
                self.send_response(500)
                self.end_headers()
                return

            # Respond with the audio file
            self.send_response(200)
            self.send_header('Content-type', 'audio/wav')
            self.end_headers()
            with open(audio_file_path, 'rb') as audio_file:
                self.wfile.write(audio_file.read())
            print("Audio file sent")  # Debugging: Confirm file sent

    def text_to_speech(self, text, audio_file_path):
        description = "Jerry speaks moderately fast in a happy tone with emphasis and high quality audio."

        input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
        prompt_input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)

        set_seed(42)
        generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids, max_new_tokens=200)
        audio_arr = generation.cpu().numpy().squeeze()
        sf.write(audio_file_path, audio_arr, model.config.sampling_rate)


def run(server_class=HTTPServer, handler_class=TTSHandler, port=int(PORT)):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f'Starting httpd server on port {port}...')
    httpd.serve_forever()

if __name__ == '__main__':
    run()

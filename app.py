from flask import Flask
from src.config.env import get_env_var, setup_api_keys
from src.routes.chat_routes import chat_bp

app = Flask(__name__)

setup_api_keys()
DEEPSEEK_API_KEY = get_env_var("DEEPSEEK_API_KEY")
OPENAI_API_KEY = get_env_var("OPENAI_API_KEY")

app.register_blueprint(chat_bp)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)

import streamlit as st
import os
import warnings
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM, Process
import datetime
import json
import re

# Suppress warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="CodeCraft AI - Elite Code Generation Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Navy Blue Theme
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    
    .stApp > header {
        background-color: transparent;
    }
    
    .title-container {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px rgba(30, 58, 138, 0.3);
    }
    
    .title-text {
        color: white;
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .subtitle-text {
        color: #e0e7ff;
        font-size: 1.2rem;
        text-align: center;
        margin-top: 0.5rem;
        font-weight: 300;
    }
    
    .chat-container {
        background-color: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        border: 2px solid #e5e7eb;
        margin-bottom: 1rem;
    }
    
    .user-message {
        background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 5px 20px;
        margin: 1rem 0;
        max-width: 80%;
        margin-left: auto;
        box-shadow: 0 3px 10px rgba(30, 64, 175, 0.3);
    }
    
    .agent-message {
        background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%);
        color: #1e293b;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 20px 5px;
        margin: 1rem 0;
        max-width: 80%;
        border-left: 4px solid #3b82f6;
        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
    }
    
    .code-block {
        background-color: #0f172a;
        color: #e2e8f0;
        padding: 1.5rem;
        border-radius: 10px;
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
        line-height: 1.6;
        margin: 1rem 0;
        border: 1px solid #334155;
        overflow-x: auto;
    }
    
    .language-tag {
        background-color: #3b82f6;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 0.5rem;
    }
    
    .sidebar-custom {
        background-color: #1e3a8a;
        color: white;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(30, 64, 175, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(30, 64, 175, 0.4);
    }
    
    .feature-card {
        background: #3d4d6e;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
        border-left: 4px solid #3b82f6;
        margin-bottom: 1rem;
    }
    
    .error-message {
        background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #b91c1c;
    }
    
    .success-message {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #047857;
    }
</style>
""", unsafe_allow_html=True)

class CodeCraftAI:
    def __init__(self):
        self.supported_languages = ["HTML", "CSS", "JavaScript", "Python", "SQL"]
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        
        if not self.gemini_api_key:
            st.error("⚠️ GEMINI_API_KEY not found in environment variables. Please set it in your .env file.")
            st.stop()
        
        # Initialize LLM
        self.llm = LLM(
            model="gemini/gemini-1.5-flash",
            api_key=self.gemini_api_key,
            temperature=0.3
        )
        
        # Initialize Agent
        self.code_agent = Agent(
            role="Elite Code Generation Specialist",
            goal="Generate high-quality, functional code in HTML, CSS, JavaScript, Python, and SQL based on user requirements",
            backstory="""You are CodeCraft AI, an elite-level programming assistant with 14+ years of experience. 
            You specialize in generating clean, efficient, and well-documented code across multiple programming languages.
            Your expertise includes AI development, UI/UX design, software architecture, and database management.
            You follow best practices, write production-ready code, and provide detailed explanations.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
    
    def validate_language_request(self, user_input):
        """Validate if the request is for supported languages"""
        user_input_lower = user_input.lower()
        
        # Check for unsupported languages
        unsupported_patterns = [
            r'\b(java|c\+\+|c#|csharp|ruby|php|go|golang|rust|swift|kotlin|scala|r\b|matlab|perl)\b',
            r'\b(react|vue|angular|node\.?js|typescript)\b'
        ]
        
        for pattern in unsupported_patterns:
            if re.search(pattern, user_input_lower):
                return False, "unsupported_language"
        
        # Check for search requests
        search_patterns = [
            r'\b(search|google|find|lookup|browse|internet|web search)\b',
            r'\b(what is|who is|when did|where is|how to find)\b.*\b(online|internet|web)\b'
        ]
        
        for pattern in search_patterns:
            if re.search(pattern, user_input_lower):
                return False, "search_request"
        
        return True, "valid"
    
    def generate_code(self, user_request, conversation_history=""):
        """Generate code based on user request"""
        is_valid, validation_type = self.validate_language_request(user_request)
        
        if not is_valid:
            if validation_type == "unsupported_language":
                return {
                    "type": "error",
                    "message": "🚫 **Language Not Supported**\n\nI can only generate code in these languages:\n• HTML\n• CSS\n• JavaScript\n• Python\n• SQL\n\nPlease request code in one of these supported languages."
                }
            elif validation_type == "search_request":
                return {
                    "type": "error", 
                    "message": "🚫 **Search Not Available**\n\nI'm a code generation specialist and cannot perform web searches or internet lookups. Please provide specific coding requirements instead."
                }
        
        # Create task for code generation
        code_task = Task(
            description=f"""
            Generate high-quality code based on this request: {user_request}
            
            Previous conversation context: {conversation_history}
            
            Requirements:
            1. Only generate code in: HTML, CSS, JavaScript, Python, or SQL
            2. Include clear comments and documentation
            3. Follow best practices and industry standards
            4. Provide explanations for complex logic
            5. Ensure code is production-ready and functional
            6. If multiple languages are needed, provide them separately
            
            Format your response with:
            - Brief explanation of the solution
            - Code blocks with language labels
            - Usage instructions if applicable
            """,
            agent=self.code_agent,
            expected_output="Well-structured code with explanations and proper formatting"
        )
        
        # Create crew and execute
        crew = Crew(
            agents=[self.code_agent],
            tasks=[code_task],
            process=Process.sequential,
            verbose=False
        )
        
        try:
            result = crew.kickoff()
            return {
                "type": "success",
                "content": str(result),
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        except Exception as e:
            return {
                "type": "error",
                "message": f"⚠️ **Code Generation Error**\n\nAn error occurred while generating code: {str(e)}\n\nPlease try rephrasing your request or check your API configuration."
            }

def main():
    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "agent" not in st.session_state:
        st.session_state.agent = CodeCraftAI()
    
    # Title Section
    st.markdown("""
    <div class="title-container">
        <h1 class="title-text">🤖 CodeCraft AI</h1>
        <p class="subtitle-text">Elite Code Generation Agent • HTML • CSS • JavaScript • Python • SQL</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🛠️ **Agent Features**")
        
        features = [
            "🎯 Multi-Language Support\nHTML, CSS, JavaScript, Python, SQL",
            "🧠 AI-Powered Generation: \nAdvanced Gemini AI integration", 
            "💬 Conversation Memory: \nContextual code generation",
            "⚡ Instant Results: \nReal-time code generation",
            "🔒 Secure & Reliable:\nEnterprise-grade performance"
        ]
        
        for feature in features:
            st.markdown(f"""
            <div class="feature-card">
                {feature}
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 📋 **Supported Languages**")
        for lang in st.session_state.agent.supported_languages:
            st.markdown(f"✅ {lang}")
        
        st.markdown("---")
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    
    # Main Chat Interface
    st.markdown("### 💬 **Chat with CodeCraft AI**")
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for i, message in enumerate(st.session_state.chat_history):
            if message["role"] == "user":
                st.markdown(f"""
                <div class="user-message">
                    <strong>You:</strong> {message["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                if message.get("type") == "error":
                    st.markdown(f"""
                    <div class="error-message">
                        <strong>CodeCraft AI:</strong><br>{message["content"]}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="agent-message">
                        <strong>CodeCraft AI:</strong><br>{message["content"]}
                    </div>
                    """, unsafe_allow_html=True)
    
    # Input section
    st.markdown("---")
    user_input = st.text_area(
        "💭 **Describe your code requirements:**",
        placeholder="Example: Create a responsive HTML page with CSS styling for a portfolio website...",
        height=100,
        key="user_input"
    )
    
    col1, col2, col3 = st.columns([2, 1, 2])
    
    with col2:
        generate_button = st.button("🚀 Generate Code", use_container_width=True, type="primary")
    
    if generate_button and user_input.strip():
        # Add user message to history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input
        })
        
        # Generate conversation context
        context = "\n".join([
            f"{msg['role']}: {msg['content']}" 
            for msg in st.session_state.chat_history[-5:]  # Last 5 messages
        ])
        
        # Show loading spinner
        with st.spinner("🤖 CodeCraft AI is generating your code..."):
            result = st.session_state.agent.generate_code(user_input, context)
        
        # Add agent response to history
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": result.get("content", result.get("message", "")),
            "type": result.get("type", "success")
        })
        
        # Rerun to update display
        st.rerun()
    
    elif generate_button and not user_input.strip():
        st.warning("⚠️ Please enter your code requirements before generating.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #64748b; font-size: 0.9rem; padding: 1rem;">
        <strong>CodeCraft AI</strong> - Elite Code Generation Agent<br>
        Powered by Gemini AI • Built with CrewAI & Streamlit<br>
        <em>Generating premium code solutions since 2025</em>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
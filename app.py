import os
import streamlit as st
import requests
import base64
import json
import shutil
from urllib.parse import urlparse
from git import Repo
from git.exc import GitCommandError
from typing import List, Dict, Any, TypedDict, Annotated
import operator
import asyncio
from langchain.tools import StructuredTool, Tool
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_anthropic import ChatAnthropic
from langchain_community.tools import ShellTool
from langgraph.prebuilt import create_react_agent

if "use_sonnet" not in st.session_state:
    st.session_state.use_sonnet = False

col1, col2 = st.columns(2)

# Initialize show_system_prompt in session state
if "show_system_prompt" not in st.session_state:
    st.session_state.show_system_prompt = False

with col1:
    if st.button("Show System Prompt" if not st.session_state.show_system_prompt else "Hide System Prompt"):
        st.session_state.show_system_prompt = not st.session_state.show_system_prompt

with col2:
    if st.button("Use Sonnet 3.5"):
        st.session_state.use_sonnet = True

if st.session_state.use_sonnet:
    sonnet_api_key = st.text_input("Input Anthropic API Key for Sonnet 3.5", type="password")
    if sonnet_api_key:
        os.environ["ANTHROPIC_API_KEY"] = sonnet_api_key


# Show title and description.
st.title("Coder for NextJS Templates")
st.write(
    "This chatbot connects to a Next.JS Github Repository to answer questions and modify code "
    "given the user's prompt. Please input your repo url and github token to allow the AI to connect, then query it by asking questions or requesting feature changes!"
)

# Ask user for their Github Repo URL, Github Token, and Anthropic API key via `st.text_input`.
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "Github-Agent"

github_repo_url = st.text_input("Github Repo URL (e.g., https://github.com/user/repo)")

# Use st.markdown for the hyperlink text
st.markdown(
    '[How to get your Github Token](https://docs.github.com/en/enterprise-server@3.9/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens)'
)
github_token = st.text_input("Enter your Github Token", type="password")

# anthropic_api_key = st.text_input("Anthropic API Key", type="password")

anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

if not (github_repo_url and github_token and anthropic_api_key):
    st.info("Please add your Github Repo URL, Github Token, and Anthropic API key to continue.", icon="🗝️")
else:
    # Set environment variables
    os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key
    os.environ["GITHUB_TOKEN"] = github_token

    # Parse the repository URL to extract user_name and REPO_NAME
    parsed_url = urlparse(github_repo_url)
    path_parts = parsed_url.path.strip('/').split('/')
    if len(path_parts) == 2:
        user_name, repo_name = path_parts
    else:
        st.error("Invalid GitHub repository URL. Please ensure it is in the format: https://github.com/user/repo")
        st.stop()

    REPO_URL = f"https://{github_token}@github.com/{user_name}/{repo_name}.git"
    
    headers = {
        'Authorization': f'token {github_token}',
        'Accept': 'application/vnd.github.v3+json',
    }

    def force_clone_repo(*args, **kwargs) -> str:
        if os.path.exists(repo_name):
            shutil.rmtree(repo_name)
        try:
            Repo.clone_from(REPO_URL, repo_name)
            return f"Repository {repo_name} forcefully cloned successfully."
        except GitCommandError as e:
            return f"Error cloning repository: {str(e)}"

    force_clone_tool = Tool(
        name="force_clone_repo",
        func=force_clone_repo,
        description="Forcefully clone the repository, removing any existing local copy."
    )

    class WriteFileInput(BaseModel):
        file_path: str = Field(..., description="The path of the file to write to")
        content: str = Field(..., description="The content to write to the file")

    def write_file_content(file_path: str, content: str) -> str:
        full_path = os.path.join(repo_name, file_path)
        try:
            with open(full_path, 'w') as file:
                file.write(content)
            return f"Successfully wrote to {full_path}"
        except Exception as e:
            return f"Error writing to file: {str(e)}"

    file_write_tool = StructuredTool.from_function(
        func=write_file_content,
        name="write_file",
        description="Write content to a specific file in the repository.",
        args_schema=WriteFileInput
    )

    def read_file_content(file_path: str) -> str:
        force_clone_repo()  # Ensure we have the latest version before reading
        full_path = os.path.join(repo_name, file_path)
        try:
            with open(full_path, 'r') as file:
                content = file.read()
            return f"File content:\n{content}"
        except Exception as e:
            return f"Error reading file: {str(e)}"

    file_read_tool = Tool(
        name="read_file",
        func=read_file_content,
        description="Read content from a specific file in the repository."
    )

    class CommitPushInput(BaseModel):
        commit_message: str = Field(..., description="The commit message")

    def commit_and_push(commit_message: str) -> str:
        try:
            repo = Repo(repo_name)
            repo.git.add(A=True)
            repo.index.commit(commit_message)
            origin = repo.remote(name='origin')
            push_info = origin.push()

            if push_info:
                if push_info[0].flags & push_info[0].ERROR:
                    return f"Error pushing changes: {push_info[0].summary}"
                else:
                    return f"Changes committed and pushed successfully with message: {commit_message}"
            else:
                return "No changes to push"
        except GitCommandError as e:
            return f"GitCommandError: {str(e)}"
        except Exception as e:
            return f"Unexpected error: {str(e)}"

    commit_push_tool = StructuredTool.from_function(
        func=commit_and_push,
        name="commit_and_push",
        description="Commit and push changes to the repository with a specific commit message.",
        args_schema=CommitPushInput
    )

    tools = [force_clone_tool, file_read_tool, file_write_tool, commit_push_tool, ShellTool()]

    class AgentState(TypedDict):
        messages: Annotated[List[BaseMessage], operator.add]

    if st.session_state.use_sonnet and "ANTHROPIC_API_KEY" in os.environ:
        llm = ChatAnthropic(temperature=0, model_name="claude-3-sonnet-20240307")
    else:
        llm = ChatAnthropic(temperature=0, model_name="claude-3-haiku-20240307")

    system_prompt_template = """You are an AI specialized in managing and analyzing a GitHub repository for a Next.js blog website.
    Your task is to answer user queries about the repository or execute tasks for modifying it.

    Before performing any operation, always use the force_clone_repo tool to ensure you have the latest version of the repository.

    Here is all of the code from the repository as well as the file paths for context of how the repo is structured: {REPO_CONTENT}

    Given this context, follow this prompt in completing the user's task:
    For user questions, provide direct answers based on the current state of the repository.
    For tasks given by the user, use the available tools and your knowledge of the repo to make necessary changes to the repository.

    When making changes, remember to force clone the repository first, make the changes, and then commit and push the changes.
    Available tools:
    1. shell_tool: Execute shell commands
    2. write_file: Write content to a specific file. Use as: write_file(file_path: str, content: str)
    3. force_clone_repo: Forcefully clone the repository, removing any existing local copy
    4. commit_and_push: Commit and push changes to the repository
    5. read_file: Read content from a specific file in the repository
    When using the write_file tool, always provide both the file_path and the content as separate arguments.

    Respond to the human's messages and use tools when necessary to complete tasks. Take a deep breath and think through the task step by step:"""

    from langgraph.checkpoint import MemorySaver

    memory = MemorySaver()

    def extract_repo_info(url):
        parts = url.split('/')
        if 'github.com' not in parts:
            raise ValueError("Not a valid GitHub URL")

        owner = parts[parts.index('github.com') + 1]
        repo = parts[parts.index('github.com') + 2]

        path_start_index = parts.index(repo) + 1
        if path_start_index < len(parts) and parts[path_start_index] == 'tree':
            path_start_index += 2

        path = '/'.join(parts[path_start_index:])

        return owner, repo, path

    def get_repo_contents(owner, repo, path=''):
        api_url = f'https://api.github.com/repos/{owner}/{repo}/contents/{path}'
        response = requests.get(api_url, headers=headers)
        return response.json()

    def get_file_content_and_metadata(file_url):
        response = requests.get(file_url, headers=headers)
        content_data = response.json()
        content = content_data.get('content', '')

        if content:
            try:
                decoded_content = base64.b64decode(content)
                decoded_content_str = decoded_content.decode('utf-8')
            except (base64.binascii.Error, UnicodeDecodeError):
                decoded_content_str = content
        else:
            decoded_content_str = ''

        last_modified = content_data.get('last_modified') or response.headers.get('Last-Modified', '')

        return decoded_content_str, last_modified

    def is_valid_extension(filename):
        valid_extensions = ['.ipynb', '.py', '.js', '.md', '.mdx', 'tsx', 'ts', 'css', '.json']
        return any(filename.endswith(ext) for ext in valid_extensions)

    def process_repo(repo_url):
        owner, repo, initial_path = extract_repo_info(repo_url)
        result = []
        stack = [(initial_path, f'https://api.github.com/repos/{owner}/{repo}/contents/{initial_path}')]

        while stack:
            path, url = stack.pop()
            contents = get_repo_contents(owner, repo, path)

            if isinstance(contents, dict) and 'message' in contents:
                print(f"Error: {contents['message']}")
                return []

            for item in contents:
                if item['type'] == 'file':
                    if is_valid_extension(item['name']):
                        file_url = item['url']
                        file_content, last_modified = get_file_content_and_metadata(file_url)
                        if file_content:
                            result.append({
                                'url': item['html_url'],
                                'markdown': file_content,
                                'last_modified': last_modified
                            })
                elif item['type'] == 'dir':
                    stack.append((item['path'], item['url']))

        return result

    def refresh_repo_data():
        repo_contents = process_repo(github_repo_url)
        repo_contents_json = json.dumps(repo_contents, ensure_ascii=False, indent=2)
        st.session_state.REPO_CONTENT = repo_contents_json
        st.success("Repository content refreshed successfully.")

        # Update the system prompt with the new repo content
        st.session_state.system_prompt = system_prompt_template.format(REPO_CONTENT=st.session_state.REPO_CONTENT)

        # Recreate the graph with the updated system prompt
        global graph
        if st.session_state.use_sonnet and "ANTHROPIC_API_KEY" in os.environ:
            new_llm = ChatAnthropic(temperature=0, model_name="claude-3-sonnet-20240307")
        else:
            new_llm = ChatAnthropic(temperature=0, model_name="claude-3-haiku-20240307")
        
        graph = create_react_agent(
            new_llm,
            tools=tools,
            messages_modifier=st.session_state.system_prompt,
            checkpointer=memory
        )

    if st.session_state.use_sonnet and "ANTHROPIC_API_KEY" in os.environ:
        refresh_repo_data()

    # Automatically refresh repo data when keys are provided
    if "REPO_CONTENT" not in st.session_state:
        refresh_repo_data()

    # Initialize system_prompt in session state
    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = system_prompt_template.format(REPO_CONTENT=st.session_state.REPO_CONTENT)

    graph = create_react_agent(
        llm,
        tools=tools,
        messages_modifier=st.session_state.system_prompt,
        checkpointer=memory
    )

    from langchain_core.messages import AIMessage, ToolMessage

    async def run_github_editor(query: str, thread_id: str = "default"):
        inputs = {"messages": [HumanMessage(content=query)]}
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": 50  # Add this line to set the recursion limit
        }

        st.write(f"Human: {query}\n")

        current_thought = ""

        async for event in graph.astream_events(inputs, config=config, version="v2"):
            kind = event["event"]
            if kind == "on_chat_model_start":
                st.write("AI is thinking...")
            elif kind == "on_chat_model_stream":
                data = event["data"]
                if data["chunk"].content:
                    content = data["chunk"].content
                    if isinstance(content, list) and content and isinstance(content[0], dict):
                        text = content[0].get('text', '')
                        current_thought += text
                        if text.endswith(('.', '?', '!')):
                            st.write(current_thought.strip())
                            current_thought = ""
                    else:
                        st.write(content, end="")
            elif kind == "on_tool_start":
                st.write(f"\nUsing tool: {event['name']}")
            elif kind == "on_tool_end":
                st.write(f"Tool result: {event['data']['output']}\n")

    # Create a session state variable to store the chat messages. This ensures that the
    # messages persist across reruns.
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display the current system prompt if show_system_prompt is True
    if st.session_state.show_system_prompt:
        st.text_area("Current System Prompt", st.session_state.system_prompt, height=300)

    # Display the existing chat messages via `st.chat_message`.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Create a chat input field to allow the user to enter a message. This will display
    # automatically at the bottom of the page.
    if prompt := st.chat_input("What is up?"):

        # Store and display the current prompt.
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate a response using the custom chatbot logic.
        asyncio.run(run_github_editor(prompt))

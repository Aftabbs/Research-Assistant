import asyncio
import streamlit as st
import pandas as pd
from datetime import datetime
import json
import re
from urllib.parse import urlparse
import os
import sys

# Try to import the fixed client
try:
    from client import ResearchAssistantClient, ResearchClientConfig
except ImportError:
    st.error("Please ensure client.py is in the same directory")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .search-result {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
        background-color: #f9f9f9;
    }
    .search-result h4 {
        margin: 0 0 10px 0;
        color: #1f77b4;
    }
    .search-result .domain {
        color: #666;
        font-size: 0.9em;
    }
    .project-card {
        border: 1px solid #e1e5e9;
        border-radius: 8px;
        padding: 16px;
        margin: 8px 0;
        background-color: #ffffff;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .source-item {
        border-left: 3px solid #1f77b4;
        padding-left: 12px;
        margin: 8px 0;
        background-color: #f8f9fa;
        padding: 10px 12px;
        border-radius: 4px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        'client': None,
        'connected': False,
        'current_project_id': None,
        'projects': [],
        'search_results': [],
        'chat_history': [],
        'last_search': "",
        'show_search_history': False,
        'show_analytics': False
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

# Call initialization
init_session_state()

async def initialize_client():
    """Initialize the research client"""
    if st.session_state.client is None:
        config = ResearchClientConfig()
        st.session_state.client = ResearchAssistantClient(config)
        
        try:
            await st.session_state.client.connect()
            st.session_state.connected = True
            return True
        except Exception as e:
            st.error(f"Failed to connect: {str(e)}")
            return False
    return st.session_state.connected

async def load_projects():
    """Load research projects"""
    if st.session_state.client and st.session_state.connected:
        try:
            result = await st.session_state.client.get_projects()
            if result.get("status") == "success":
                st.session_state.projects = result.get("projects", [])
                return True
        except Exception as e:
            st.error(f"Failed to load projects: {str(e)}")
    return False

async def safe_quick_search(query: str, num_results: int = 10):
    """Safely perform quick search with error handling"""
    try:
        if not st.session_state.client or not st.session_state.connected:
            return {"status": "error", "message": "Client not connected"}
        
        result = await st.session_state.client.quick_search(query, num_results)
        return result
    except Exception as e:
        return {"status": "error", "message": f"Search failed: {str(e)}"}

async def safe_extract_content(url: str, project_id: int = None):
    """Safely extract content with error handling"""
    try:
        if not st.session_state.client or not st.session_state.connected:
            return {"status": "error", "message": "Client not connected"}
        
        if not st.session_state.client.mcp:
            return {"status": "error", "message": "MCP client not available"}
        
        tools = await st.session_state.client.mcp.get_tools()
        extract_tool = next((t for t in tools if t.name == "extract_content"), None)
        
        if not extract_tool:
            return {"status": "error", "message": "Extract content tool not available"}
        
        params = {"url": url}
        if project_id:
            params["project_id"] = project_id
            
        result = await extract_tool.ainvoke(params)
        return result
    except Exception as e:
        return {"status": "error", "message": f"Content extraction failed: {str(e)}"}

def safe_get_value(obj, key, default=""):
    """Safely get value from object with fallback"""
    try:
        if isinstance(obj, dict):
            return obj.get(key, default)
        elif hasattr(obj, key):
            return getattr(obj, key, default)
        else:
            return default
    except:
        return default

def main():
    """Main Streamlit application"""
    
    # Header
    st.title("🔬 Research Assistant")
    st.markdown("*Intelligent research with web search, content extraction, and project organization*")
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Control Panel")
        
        # Connection status
        if st.session_state.connected:
            st.success("✅ Connected to Research Server")
        else:
            st.error("❌ Not Connected")
            if st.button("🔄 Connect to Server"):
                with st.spinner("Connecting..."):
                    if asyncio.run(initialize_client()):
                        st.rerun()
        
        st.divider()
        
        # Project Management
        st.subheader("📁 Projects")
        
        if st.session_state.connected:
            # Refresh projects button
            if st.button("🔄 Refresh Projects"):
                with st.spinner("Loading projects..."):
                    asyncio.run(load_projects())
                    st.rerun()
            
            # Create new project
            with st.expander("➕ Create New Project"):
                with st.form("create_project"):
                    project_name = st.text_input("Project Name*")
                    project_description = st.text_area("Description")
                    
                    if st.form_submit_button("Create Project"):
                        if project_name:
                            with st.spinner("Creating project..."):
                                try:
                                    result = asyncio.run(st.session_state.client.create_project(
                                        project_name, project_description
                                    ))
                                    if result.get("status") == "success":
                                        st.success(f"Project '{project_name}' created!")
                                        st.session_state.current_project_id = result.get("project_id")
                                        asyncio.run(load_projects())
                                        st.rerun()
                                    else:
                                        st.error(result.get("message", "Failed to create project"))
                                except Exception as e:
                                    st.error(f"Error: {str(e)}")
                        else:
                            st.error("Project name is required")
            
            # Project selection
            if st.session_state.projects:
                project_options = ["None"] + [f"{p['id']}: {p['name']}" for p in st.session_state.projects]
                current_selection = "None"
                
                if st.session_state.current_project_id:
                    for option in project_options:
                        if option.startswith(f"{st.session_state.current_project_id}:"):
                            current_selection = option
                            break
                
                selected_project = st.selectbox(
                    "Current Project:",
                    project_options,
                    index=project_options.index(current_selection),
                    key="project_selector"
                )
                
                if selected_project != current_selection:
                    if selected_project == "None":
                        st.session_state.current_project_id = None
                        if st.session_state.client:
                            st.session_state.client.set_current_project(None)
                    else:
                        project_id = int(selected_project.split(":")[0])
                        st.session_state.current_project_id = project_id
                        if st.session_state.client:
                            st.session_state.client.set_current_project(project_id)
                    st.rerun()
        
        st.divider()
        
        # Quick actions
        st.subheader("⚡ Quick Actions")
        
        if st.button("📜 View Search History"):
            st.session_state.show_search_history = True
        
        if st.button("📊 Project Analytics"):
            st.session_state.show_analytics = True

    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Main research interface
        st.header("🔍 Research Interface")
        
        # Quick search section
        with st.container():
            st.subheader("Quick Web Search")
            search_col1, search_col2 = st.columns([4, 1])
            
            with search_col1:
                search_query = st.text_input("Search the web:", placeholder="Enter your search query...", key="search_input")
            
            with search_col2:
                num_results = st.selectbox("Results:", [5, 10, 15, 20], index=1)
            
            # Search button and execution
            if st.button("🔍 Search", key="search_button") and search_query:
                st.session_state.last_search = search_query
                with st.spinner("Searching..."):
                    result = asyncio.run(safe_quick_search(search_query, num_results))
                    
                    if result.get("status") == "success":
                        # Safely extract results
                        results = result.get("results", [])
                        if isinstance(results, list):
                            st.session_state.search_results = results
                            st.success(f"Found {len(st.session_state.search_results)} results")
                        else:
                            st.error("Invalid search results format")
                            st.session_state.search_results = []
                    else:
                        st.error(result.get("message", "Search failed"))
                        st.session_state.search_results = []
        
        # Display search results
        if st.session_state.search_results:
            st.subheader("🔍 Search Results")
            
            for i, result in enumerate(st.session_state.search_results):
                with st.container():
                    col_result, col_action = st.columns([4, 1])
                    
                    with col_result:
                        # Safely access result fields
                        title = safe_get_value(result, 'title', 'No Title')
                        snippet = safe_get_value(result, 'snippet', 'No description available')
                        link = safe_get_value(result, 'link', '#')
                        domain = safe_get_value(result, 'domain', 'Unknown')
                        
                        st.markdown(f"""
                        <div class="search-result">
                            <h4>{title}</h4>
                            <p>{snippet}</p>
                            <div class="domain">
                                🌐 {domain} | 
                                <a href="{link}" target="_blank">View Page</a>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_action:
                        if st.button(f"📄 Extract", key=f"extract_{i}"):
                            with st.spinner("Extracting content..."):
                                extract_result = asyncio.run(safe_extract_content(
                                    link, 
                                    st.session_state.current_project_id
                                ))
                                
                                if extract_result.get("status") == "success":
                                    st.success("Content extracted successfully!")
                                    if st.session_state.current_project_id:
                                        st.info("Content saved to current project")
                                        # Refresh projects to update source count
                                        asyncio.run(load_projects())
                                else:
                                    st.error(extract_result.get("message", "Failed to extract content"))
        
        st.divider()
        
        # Research chat interface
        st.subheader("💬 Research Assistant Chat")
        
        # Display chat history
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"**You:** {message['content']}")
            else:
                st.markdown(f"**Assistant:** {message['content']}")
        
        # Chat input
        with st.form("chat_form"):
            user_input = st.text_area(
                "Ask me anything about research:",
                placeholder="Example: 'Search for recent studies on renewable energy and save to my current project'",
                height=100,
                key="chat_input"
            )
            
            col_send, col_clear = st.columns([1, 1])
            with col_send:
                send_button = st.form_submit_button("💬 Send")
            with col_clear:
                clear_button = st.form_submit_button("🗑️ Clear Chat")
        
        if send_button and user_input:
            if not st.session_state.connected:
                st.error("Please connect to the server first")
            else:
                # Add user message to history
                st.session_state.chat_history.append({"role": "user", "content": user_input})
                
                with st.spinner("Research assistant is thinking..."):
                    try:
                        result = asyncio.run(st.session_state.client.research_query(user_input))
                        if result.get("status") == "success":
                            response = result.get("response", "No response received")
                            st.session_state.chat_history.append({"role": "assistant", "content": response})
                            # Refresh projects after research in case new sources were added
                            asyncio.run(load_projects())
                        else:
                            error_msg = f"Error: {result.get('message', 'Unknown error')}"
                            st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
                    except Exception as e:
                        error_msg = f"Error: {str(e)}"
                        st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
                
                st.rerun()
        
        if clear_button:
            st.session_state.chat_history = []
            st.rerun()

    with col2:
        # Right sidebar - Project details and analytics
        st.header("📊 Project Details")
        
        if st.session_state.current_project_id and st.session_state.projects:
            # Find current project
            current_project = next(
                (p for p in st.session_state.projects if p["id"] == st.session_state.current_project_id),
                None
            )
            
            if current_project:
                # Clean project name for display (remove JSON artifacts)
                project_name = current_project['name']
                if project_name.startswith('{') and project_name.endswith('}'):
                    try:
                        data = json.loads(project_name)
                        if isinstance(data, dict) and 'title' in data:
                            project_name = data['title']
                        elif isinstance(data, dict) and 'name' in data:
                            project_name = data['name']
                    except:
                        pass
                
                st.markdown(f"""
                <div class="project-card">
                    <h3>{project_name}</h3>
                    <p><strong>Description:</strong> {current_project.get('description', 'No description')}</p>
                    <p><strong>Sources:</strong> {current_project['source_count']}</p>
                    <p><strong>Created:</strong> {current_project['created_at'][:10]}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Project sources
                if st.button("📑 View Sources", key="view_sources_btn"):
                    with st.spinner("Loading sources..."):
                        try:
                            if st.session_state.client and st.session_state.client.mcp:
                                tools = asyncio.run(st.session_state.client.mcp.get_tools())
                                sources_tool = next((t for t in tools if t.name == "get_project_sources"), None)
                                if sources_tool:
                                    sources_result = asyncio.run(sources_tool.ainvoke({
                                        "project_id": st.session_state.current_project_id
                                    }))
                                    if sources_result.get("status") == "success":
                                        sources = sources_result.get("sources", [])
                                        st.subheader("📚 Project Sources")
                                        for source in sources[:5]:  # Show first 5
                                            url_display = source['url'][:50] + "..." if len(source['url']) > 50 else source['url']
                                            st.markdown(f"""
                                            <div class="source-item">
                                                <strong>{source['title']}</strong><br>
                                                <small>🌐 <a href="{source['url']}" target="_blank">{url_display}</a></small><br>
                                                <small>📅 {source['created_at'][:10]}</small>
                                            </div>
                                            """, unsafe_allow_html=True)
                                        
                                        if len(sources) > 5:
                                            st.info(f"... and {len(sources) - 5} more sources")
                                    else:
                                        st.error("Failed to load sources")
                                else:
                                    st.error("Sources tool not available")
                            else:
                                st.error("Client not connected")
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                
                # Generate citations
                if st.button("📝 Generate Citations", key="gen_citations_btn"):
                    with st.spinner("Generating citations..."):
                        try:
                            if st.session_state.client and st.session_state.client.mcp:
                                tools = asyncio.run(st.session_state.client.mcp.get_tools())
                                citations_tool = next((t for t in tools if t.name == "generate_citations"), None)
                                if citations_tool:
                                    citations_result = asyncio.run(citations_tool.ainvoke({
                                        "project_id": st.session_state.current_project_id,
                                        "style": "apa"
                                    }))
                                    if citations_result.get("status") == "success":
                                        citations = citations_result.get("citations", [])
                                        st.subheader("📖 Citations (APA Style)")
                                        for citation in citations:
                                            st.text(citation["citation"])
                                    else:
                                        st.error("Failed to generate citations")
                                else:
                                    st.error("Citations tool not available")
                            else:
                                st.error("Client not connected")
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
        else:
            st.info("Select a project to view details")
        
        st.divider()
        
        # Quick stats
        if st.session_state.projects:
            st.subheader("📈 Quick Stats")
            
            total_projects = len(st.session_state.projects)
            total_sources = sum(p.get("source_count", 0) for p in st.session_state.projects)
            
            col_stat1, col_stat2 = st.columns(2)
            with col_stat1:
                st.markdown(f"""
                <div class="metric-card">
                    <h2>{total_projects}</h2>
                    <p>Projects</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col_stat2:
                st.markdown(f"""
                <div class="metric-card">
                    <h2>{total_sources}</h2>
                    <p>Sources</p>
                </div>
                """, unsafe_allow_html=True)

# Auto-connect on startup
if not st.session_state.connected:
    with st.spinner("Connecting to Research Server..."):
        try:
            success = asyncio.run(initialize_client())
            if success:
                asyncio.run(load_projects())
        except Exception as e:
            st.error(f"Failed to initialize: {str(e)}")

if __name__ == "__main__":
    main()
import os
import asyncio
import json
import logging
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from datetime import datetime

from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain_openai import AzureChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from dotenv import load_dotenv

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("research_client")

@dataclass
class ResearchClientConfig:
    """Configuration for the Research Assistant client"""
    # Azure OpenAI
    azure_endpoint: str = os.getenv("AZURE_OPENAI_ENDPOINT", "https://YOUR_AZURE_RESOURCE.openai.azure.com/")
    azure_deployment: str = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
    azure_api_key: str = os.getenv("AZURE_OPENAI_API_KEY", "REPLACE_ME")
    azure_api_version: str = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
    
    # MCP Server
    mcp_url: str = os.getenv("RESEARCH_MCP_URL", "http://127.0.0.1:8000/mcp")
    
    # Agent settings
    max_iterations: int = 6
    agent_timeout: int = 90
    memory_window_size: int = 8
    connect_timeout: int = 12

class ResearchAssistantClient:
    """Enhanced research assistant client with proper parameter handling"""
    
    def __init__(self, config: ResearchClientConfig):
        self.config = config
        self.llm: Optional[AzureChatOpenAI] = None
        self.mcp: Optional[MultiServerMCPClient] = None
        self.agent_executor: Optional[AgentExecutor] = None
        self.memory: Optional[ConversationBufferWindowMemory] = None
        self.tools: List[Tool] = []
        self.current_project_id: Optional[int] = None
        
    async def connect(self):
        """Connect to Azure OpenAI and MCP server"""
        try:
            # Validate configuration
            if "YOUR_AZURE_RESOURCE" in self.config.azure_endpoint or self.config.azure_api_key == "REPLACE_ME":
                raise RuntimeError("Azure OpenAI settings not configured properly")
            
            # Initialize Azure OpenAI
            self.llm = AzureChatOpenAI(
                azure_endpoint=self.config.azure_endpoint,
                api_key=self.config.azure_api_key,
                api_version=self.config.azure_api_version,
                azure_deployment=self.config.azure_deployment,
                temperature=0.1,
                timeout=self.config.agent_timeout,
                max_retries=2,
            )
            
            # Initialize memory
            self.memory = ConversationBufferWindowMemory(
                k=self.config.memory_window_size,
                memory_key="chat_history",
                return_messages=True,
                output_key="output"
            )
            
            # Initialize MCP client
            self.mcp = MultiServerMCPClient({
                "research-server": {
                    "url": self.config.mcp_url,
                    "transport": "streamable_http"
                }
            })
            
            if hasattr(self.mcp, "start"):
                await asyncio.wait_for(self.mcp.start(), timeout=self.config.connect_timeout)
            
            # Get tools from MCP server
            base_tools = await asyncio.wait_for(self.mcp.get_tools(), timeout=self.config.connect_timeout)
            
            # Wrap tools for better error handling
            wrapped_tools = []
            for tool in base_tools:
                wrapped_tools.append(self._wrap_tool(tool))
            
            self.tools = wrapped_tools
            
            # Create enhanced system prompt
            system_prompt = self._create_system_prompt()
            
            # Create a custom ReAct prompt template to fix formatting issues
            custom_react_template = """You are a helpful Research Assistant. You have access to the following tools:

{tools}

Use the following format EXACTLY:

Question: the input question you must answer
Thought: think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action (use simple parameters, not JSON objects)
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

IMPORTANT TOOL USAGE:
- For web_search: use just the query string
- For extract_content: use just the URL string  
- For create_research_project: use just the project name
- For get_project_sources: use just the project ID number
- Always use simple string or number inputs, never JSON objects

Current date: {current_date}
Current Project: {current_project}

Begin!

Question: {input}
{agent_scratchpad}"""

            react_prompt = PromptTemplate(
                template=custom_react_template,
                input_variables=["input", "agent_scratchpad", "tools", "tool_names"],
                partial_variables={
                    "current_date": datetime.now().strftime('%Y-%m-%d'),
                    "current_project": str(self.current_project_id) if self.current_project_id else "None"
                }
            )
            
            # Create agent
            agent = create_react_agent(llm=self.llm, tools=self.tools, prompt=react_prompt)
            self.agent_executor = AgentExecutor(
                agent=agent,
                tools=self.tools,
                memory=self.memory,
                verbose=True,
                max_iterations=self.config.max_iterations,
                return_intermediate_steps=True,
                handle_parsing_errors=True,
                early_stopping_method="force",
            )
            
            logger.info("Research Assistant Client connected successfully!")
            logger.info(f"  - Tools loaded: {len(self.tools)}")
            logger.info(f"  - Memory window: {self.config.memory_window_size}")
            
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            raise RuntimeError(f"Failed to connect research client: {str(e)}")
    
    def _create_system_prompt(self) -> str:
        """Create enhanced system prompt for research assistant"""
        return f"""You are an expert Research Assistant that helps users conduct thorough research on any topic.

CURRENT DATE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

YOUR CAPABILITIES:
1. WEB SEARCH: Search the internet for information on any topic
2. CONTENT EXTRACTION: Extract and analyze content from web pages  
3. PROJECT MANAGEMENT: Organize research into structured projects
4. CITATION GENERATION: Create properly formatted citations
5. CONTENT SUMMARIZATION: Summarize long texts into key points

WORKFLOW GUIDANCE:
- For new research topics, suggest creating a project first
- Always search for multiple sources to get comprehensive information
- Extract and save important content to projects for organization
- Generate summaries of long content for easy review
- Create citations for academic or professional use

TOOL USAGE:
- web_search: Use for finding information on any topic
- extract_content: Use to get full content from URLs (and save to projects)
- create_research_project: Create organized research collections
- list_research_projects: Show existing projects
- get_project_sources: View sources in a project
- summarize_text: Create concise summaries
- generate_citations: Create formatted citations
- get_search_history: Review past searches

CURRENT PROJECT: {self.current_project_id if self.current_project_id else "None selected"}

Be helpful, thorough, and organized in your research assistance!"""
    
    def _wrap_tool(self, tool) -> Tool:
        """Wrap MCP tools with enhanced error handling and proper parameter formatting"""
        async def _run(*args, **kwargs):
            try:
                # Handle different input formats more robustly
                if args and not kwargs:
                    if len(args) == 1:
                        if isinstance(args[0], dict):
                            result = await tool.ainvoke(args[0])
                        elif isinstance(args[0], str):
                            # For string inputs, create appropriate parameter dict based on tool
                            if tool.name == "web_search":
                                result = await tool.ainvoke({"query": args[0]})
                            elif tool.name == "extract_content":
                                result = await tool.ainvoke({"url": args[0]})
                            elif tool.name == "create_research_project":
                                result = await tool.ainvoke({"name": args[0]})
                            elif tool.name == "get_project_sources":
                                result = await tool.ainvoke({"project_id": int(args[0])})
                            elif tool.name == "summarize_text":
                                result = await tool.ainvoke({"text": args[0]})
                            elif tool.name in ["list_research_projects", "get_search_history", "health_check"]:
                                result = await tool.ainvoke({})
                            else:
                                result = await tool.ainvoke({"input": args[0]})
                        elif isinstance(args[0], (int, float)):
                            # For numeric inputs (like project IDs)
                            if tool.name == "get_project_sources":
                                result = await tool.ainvoke({"project_id": int(args[0])})
                            elif tool.name == "generate_citations":
                                result = await tool.ainvoke({"project_id": int(args[0])})
                            else:
                                result = await tool.ainvoke({"input": args[0]})
                        else:
                            result = await tool.ainvoke({"input": str(args[0])})
                    else:
                        # Multiple args - try to map to common patterns
                        if tool.name == "web_search":
                            kwargs = {"query": str(args[0])}
                        elif tool.name == "extract_content":
                            kwargs = {"url": str(args[0])}
                            if len(args) > 1:
                                try:
                                    kwargs["project_id"] = int(args[1]) if str(args[1]).isdigit() else None
                                except:
                                    kwargs["project_id"] = None
                        elif tool.name == "create_research_project":
                            kwargs = {"name": str(args[0])}
                            if len(args) > 1:
                                kwargs["description"] = str(args[1])
                        elif tool.name == "get_project_sources":
                            kwargs = {"project_id": int(args[0])}
                        elif tool.name == "generate_citations":
                            kwargs = {"project_id": int(args[0])}
                            if len(args) > 1:
                                kwargs["style"] = str(args[1])
                        else:
                            kwargs = {"input": str(args[0])}
                        result = await tool.ainvoke(kwargs)
                else:
                    result = await tool.ainvoke(kwargs)
                
                return result
            except Exception as e:
                logger.error(f"Tool {tool.name} failed: {e}")
                return {
                    "status": "error", 
                    "message": f"Tool {tool.name} failed: {str(e)}"
                }
        
        return Tool.from_function(
            name=tool.name,
            description=getattr(tool, 'description', f"{tool.name} research tool"),
            func=lambda *a, **kw: asyncio.run(_run(*a, **kw)),
            coroutine=_run,
        )
    
    async def research_query(self, query: str) -> Dict[str, Any]:
        """Process a research query"""
        if not self.agent_executor:
            raise RuntimeError("Client not connected. Call connect() first.")
        
        try:
            # Handle simple greetings without using agent
            simple_greetings = ["hi", "hello", "hey", "hi there", "good morning", "good afternoon"]
            if query.lower().strip() in simple_greetings:
                return {
                    "status": "success",
                    "response": "Hello! I'm your Research Assistant. I can help you search for information, create research projects, extract content from websites, and organize your findings. What would you like to research today?",
                    "intermediate_steps": [],
                    "current_project": self.current_project_id
                }
            
            # Add current project context to query if available
            if self.current_project_id:
                query_with_context = f"[Current Project ID: {self.current_project_id}] {query}"
            else:
                query_with_context = query
            
            # Prepare agent input
            agent_input = {
                "input": query_with_context,
                "chat_history": self.memory.buffer if self.memory else ""
            }
            
            # Execute agent
            result = await self.agent_executor.ainvoke(agent_input)
            
            return {
                "status": "success",
                "response": result.get("output", ""),
                "intermediate_steps": result.get("intermediate_steps", []),
                "current_project": self.current_project_id
            }
            
        except Exception as e:
            logger.error(f"Research query failed: {e}")
            return {
                "status": "error",
                "message": f"Research query failed: {str(e)}"
            }
    
    def set_current_project(self, project_id: Optional[int]):
        """Set the current project context"""
        self.current_project_id = project_id
        logger.info(f"Current project set to: {project_id}")
    
    async def quick_search(self, query: str, num_results: int = 5) -> Dict[str, Any]:
        """Perform a quick web search without full agent processing"""
        try:
            if not self.mcp:
                raise RuntimeError("MCP client not connected")
            
            tools = await self.mcp.get_tools()
            search_tool = next((t for t in tools if t.name == "web_search"), None)
            
            if not search_tool:
                raise RuntimeError("Web search tool not available")
            
            result = await search_tool.ainvoke({
                "query": query,
                "num_results": num_results
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Quick search failed: {e}")
            return {
                "status": "error",
                "message": f"Quick search failed: {str(e)}"
            }
    
    async def get_projects(self) -> Dict[str, Any]:
        """Get list of research projects"""
        try:
            if not self.mcp:
                raise RuntimeError("MCP client not connected")
            
            tools = await self.mcp.get_tools()
            list_tool = next((t for t in tools if t.name == "list_research_projects"), None)
            
            if not list_tool:
                raise RuntimeError("List projects tool not available")
            
            result = await list_tool.ainvoke({})
            return result
            
        except Exception as e:
            logger.error(f"Get projects failed: {e}")
            return {
                "status": "error",
                "message": f"Get projects failed: {str(e)}"
            }
    
    async def create_project(self, name: str, description: str = "") -> Dict[str, Any]:
        """Create a new research project"""
        try:
            if not self.mcp:
                raise RuntimeError("MCP client not connected")
            
            tools = await self.mcp.get_tools()
            create_tool = next((t for t in tools if t.name == "create_research_project"), None)
            
            if not create_tool:
                raise RuntimeError("Create project tool not available")
            
            result = await create_tool.ainvoke({
                "name": name,
                "description": description
            })
            
            # If successful, set as current project
            if result.get("status") == "success" and result.get("project_id"):
                self.set_current_project(result["project_id"])
            
            return result
            
        except Exception as e:
            logger.error(f"Create project failed: {e}")
            return {
                "status": "error",
                "message": f"Create project failed: {str(e)}"
            }


# CLI interface for testing
async def main():
    """Main CLI interface for testing the research client"""
    config = ResearchClientConfig()
    client = ResearchAssistantClient(config)
    
    print("=" * 80)
    print("Research Assistant Client (Final Fixed Version)")
    print("=" * 80)
    
    try:
        print("Connecting to services...")
        await client.connect()
        print("Connected successfully!")
        print(f"Tools available: {len(client.tools)}")
        
        print("\n" + "=" * 80)
        print("Research Assistant Ready!")
        print("Examples:")
        print("• 'Search for information about climate change'")
        print("• 'Create a project called Climate Research'")
        print("• 'Extract content from https://example.com'")
        print("• 'List my projects'")
        print("• 'Generate citations for project 1'")
        print("Type 'exit' to quit, 'projects' to see projects")
        print("=" * 80)
        
        while True:
            try:
                query = input("\nYou: ").strip()
                if not query:
                    continue
                    
                if query.lower() in {"exit", "quit", "q"}:
                    print("Goodbye!")
                    break
                    
                if query.lower() in {"projects", "list"}:
                    print("\nFetching projects...")
                    projects = await client.get_projects()
                    if projects.get("status") == "success":
                        project_list = projects.get("projects", [])
                        if project_list:
                            print(f"\nFound {len(project_list)} projects:")
                            for p in project_list:
                                current = " (CURRENT)" if p["id"] == client.current_project_id else ""
                                print(f"  {p['id']}: {p['name']} - {p['source_count']} sources{current}")
                        else:
                            print("No projects found.")
                    else:
                        print(f"Error: {projects.get('message', 'Unknown error')}")
                    continue
                
                # Check for project switching
                if query.lower().startswith("use project "):
                    try:
                        project_id = int(query.split()[-1])
                        client.set_current_project(project_id)
                        print(f"Switched to project {project_id}")
                    except ValueError:
                        print("Invalid project ID. Use: 'use project 1'")
                    continue
                
                print("\nResearching...")
                response = await client.research_query(query)
                
                if response["status"] == "success":
                    print(f"\nAssistant:\n{response['response']}")
                    if response.get("intermediate_steps"):
                        print(f"\nDebug - Agent Steps: {len(response['intermediate_steps'])}")
                else:
                    print(f"\nError: {response['message']}")
                    
            except KeyboardInterrupt:
                print("\nInterrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")
                logger.exception("CLI error")
                
    except Exception as e:
        print(f"\nFailed to start: {e}")
        logger.exception("Startup error")


if __name__ == "__main__":
    asyncio.run(main())
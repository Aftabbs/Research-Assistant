import os
import sys
import json
import logging
import asyncio
import hashlib
import re
from typing import Optional, List, Dict, Any, Union
from datetime import datetime, timezone
from urllib.parse import quote_plus, urlparse
import sqlite3
from pathlib import Path
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# External API imports
try:
    import httpx
    import feedparser
    from bs4 import BeautifulSoup
    WEB_SEARCH_AVAILABLE = True
except ImportError:
    WEB_SEARCH_AVAILABLE = False

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('research-assistant-server')

# Configuration
DATA_DIR = Path("research_data")
DB_PATH = DATA_DIR / "research.db"

class ResearchDatabase:
    """SQLite database for storing research data"""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Research projects table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Research sources table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER,
                url TEXT,
                title TEXT,
                content TEXT,
                summary TEXT,
                source_hash TEXT UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (project_id) REFERENCES projects (id)
            )
        """)
        
        # Search history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS search_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                results_count INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
    
    def create_project(self, name: str, description: str = "") -> Dict[str, Any]:
        """Create a new research project"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                "INSERT INTO projects (name, description) VALUES (?, ?)",
                (name, description)
            )
            project_id = cursor.lastrowid
            conn.commit()
            
            return {
                "status": "success",
                "project_id": project_id,
                "message": f"Created project: {name}"
            }
        except sqlite3.IntegrityError:
            return {
                "status": "error",
                "message": f"Project '{name}' already exists"
            }
        finally:
            conn.close()
    
    def list_projects(self) -> List[Dict[str, Any]]:
        """List all research projects"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT p.id, p.name, p.description, p.created_at, p.updated_at,
                   COUNT(s.id) as source_count
            FROM projects p
            LEFT JOIN sources s ON p.id = s.project_id
            GROUP BY p.id, p.name, p.description, p.created_at, p.updated_at
            ORDER BY p.updated_at DESC
        """)
        
        projects = []
        for row in cursor.fetchall():
            projects.append({
                "id": row[0],
                "name": row[1],
                "description": row[2],
                "created_at": row[3],
                "updated_at": row[4],
                "source_count": row[5]
            })
        
        conn.close()
        return projects
    
    def save_source(self, project_id: int, url: str, title: str, content: str, summary: str = "") -> Dict[str, Any]:
        """Save a research source"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create hash for deduplication
        content_hash = hashlib.md5(f"{url}{title}{content}".encode()).hexdigest()
        
        try:
            cursor.execute("""
                INSERT INTO sources (project_id, url, title, content, summary, source_hash)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (project_id, url, title, content, summary, content_hash))
            
            source_id = cursor.lastrowid
            
            # Update project timestamp
            cursor.execute(
                "UPDATE projects SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (project_id,)
            )
            
            conn.commit()
            
            return {
                "status": "success",
                "source_id": source_id,
                "message": "Source saved successfully"
            }
        except sqlite3.IntegrityError:
            return {
                "status": "warning",
                "message": "Source already exists (duplicate detected)"
            }
        finally:
            conn.close()
    
    def get_project_sources(self, project_id: int) -> List[Dict[str, Any]]:
        """Get all sources for a project"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, url, title, content, summary, created_at
            FROM sources
            WHERE project_id = ?
            ORDER BY created_at DESC
        """, (project_id,))
        
        sources = []
        for row in cursor.fetchall():
            sources.append({
                "id": row[0],
                "url": row[1],
                "title": row[2],
                "content": row[3][:500] + "..." if len(row[3]) > 500 else row[3],
                "summary": row[4],
                "created_at": row[5]
            })
        
        conn.close()
        return sources
    
    def save_search(self, query: str, results_count: int):
        """Save search query to history"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "INSERT INTO search_history (query, results_count) VALUES (?, ?)",
            (query, results_count)
        )
        conn.commit()
        conn.close()
    
    def get_search_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent search history"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT query, results_count, timestamp
            FROM search_history
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))
        
        history = []
        for row in cursor.fetchall():
            history.append({
                "query": row[0],
                "results_count": row[1],
                "timestamp": row[2]
            })
        
        conn.close()
        return history

# Initialize database
research_db = ResearchDatabase(DB_PATH)

# Initialize FastMCP server
mcp = FastMCP("Research Assistant Server")

# ========== Helper Functions ==========
def clean_parameter_input(param_input):
    """Clean input parameters from JSON artifacts and formatting issues"""
    if not param_input:
        return ""
    
    # Convert to string
    param_str = str(param_input).strip()
    
    # Remove JSON-like formatting if present
    if param_str.startswith('{') and param_str.endswith('}'):
        try:
            # Try to parse as JSON and extract relevant fields
            data = json.loads(param_str)
            if isinstance(data, dict):
                # Common parameter names to extract
                for key in ['name', 'title', 'query', 'url', 'text']:
                    if key in data:
                        return str(data[key]).strip()
        except json.JSONDecodeError:
            pass
    
    # Remove quotes and clean whitespace
    param_str = param_str.strip('"').strip("'").strip()
    
    # Remove newlines and other problematic characters
    param_str = re.sub(r'[\n\r\t]', ' ', param_str)
    
    return param_str

def clean_url_input(url_input):
    """Clean and validate URL input"""
    clean_url = clean_parameter_input(url_input)
    
    if not clean_url:
        return ""
    
    # Ensure protocol
    if clean_url and not clean_url.startswith(('http://', 'https://')):
        clean_url = f'https://{clean_url}'
    
    return clean_url

# ========== Core Tools ==========

@mcp.tool()
def health_check() -> Dict[str, Any]:
    """Check server health and configuration."""
    return {
        "status": "healthy",
        "message": "Research Assistant Server is running",
        "web_search_available": WEB_SEARCH_AVAILABLE,
        "database_path": str(DB_PATH),
        "features": ["web_search", "content_extraction", "project_management", "citation_tracking"]
    }

@mcp.tool()
async def web_search(query: str = "", num_results: int = 10) -> Dict[str, Any]:
    """Search the web using Google Custom Search API."""
    if not WEB_SEARCH_AVAILABLE:
        return {"status": "error", "error": "Web search dependencies not available"}
    
    # Clean query input
    clean_query = clean_parameter_input(query)
    if not clean_query:
        return {"status": "error", "error": "Query parameter is required"}
    
    try:
        api_key = os.getenv("GOOGLE_API_KEY", "").strip()
        cse_id = os.getenv("GOOGLE_CSE_ID", "").strip()
        
        if not api_key or not cse_id:
            return {
                "status": "error", 
                "error": "Missing GOOGLE_API_KEY or GOOGLE_CSE_ID environment variables"
            }
        
        num_results = max(1, min(20, num_results))
        
        async with httpx.AsyncClient(timeout=20) as client:
            params = {
                "key": api_key,
                "cx": cse_id,
                "q": clean_query,
                "num": min(10, num_results)
            }
            
            response = await client.get(
                "https://www.googleapis.com/customsearch/v1",
                params=params
            )
            response.raise_for_status()
            data = response.json()
            
            results = []
            for i, item in enumerate(data.get("items", []), 1):
                results.append({
                    "position": i,
                    "title": item.get("title", ""),
                    "link": item.get("link", ""),
                    "snippet": item.get("snippet", ""),
                    "domain": urlparse(item.get("link", "")).netloc
                })
        
        # Save search to history
        research_db.save_search(clean_query, len(results))
        
        return {
            "status": "success",
            "query": clean_query,
            "results": results,
            "count": len(results),
            "message": f"Found {len(results)} results for '{clean_query}'"
        }
        
    except Exception as e:
        logger.error(f"Web search failed: {e}")
        return {"status": "error", "error": str(e)}

@mcp.tool()
async def extract_content(url: str = "", project_id: Optional[int] = None) -> Dict[str, Any]:
    """Extract and optionally save content from a web page."""
    if not WEB_SEARCH_AVAILABLE:
        return {"status": "error", "error": "Content extraction dependencies not available"}
    
    # Clean URL input
    clean_url = clean_url_input(url)
    if not clean_url:
        return {"status": "error", "error": "URL parameter is required"}
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        async with httpx.AsyncClient(timeout=15, headers=headers) as client:
            response = await client.get(clean_url, follow_redirects=True)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title = soup.find('title')
            title_text = title.get_text().strip() if title else "No title"
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
                script.decompose()
            
            # Extract text content
            text_content = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text_content.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            content = ' '.join(chunk for chunk in chunks if chunk)
            
            # Limit content length
            if len(content) > 5000:
                content = content[:5000] + "..."
            
            result = {
                "status": "success",
                "url": clean_url,
                "title": title_text,
                "content": content,
                "content_length": len(content),
                "extracted_at": datetime.now(timezone.utc).isoformat()
            }
            
            # Save to project if specified
            if project_id:
                save_result = research_db.save_source(
                    project_id=project_id,
                    url=clean_url,
                    title=title_text,
                    content=content
                )
                result["save_result"] = save_result
            
            return result
            
    except Exception as e:
        logger.error(f"Content extraction failed for {clean_url}: {e}")
        return {"status": "error", "error": f"Failed to extract content: {str(e)}"}

@mcp.tool()
def create_research_project(name: str = "", description: str = "") -> Dict[str, Any]:
    """Create a new research project to organize sources and findings."""
    # Clean input parameters
    clean_name = clean_parameter_input(name)
    clean_desc = clean_parameter_input(description)
    
    if not clean_name:
        return {"status": "error", "error": "Project name is required"}
    
    return research_db.create_project(clean_name, clean_desc)

@mcp.tool()
def list_research_projects() -> Dict[str, Any]:
    """List all research projects with their details."""
    projects = research_db.list_projects()
    return {
        "status": "success",
        "projects": projects,
        "count": len(projects),
        "message": f"Found {len(projects)} research projects"
    }

@mcp.tool()
def get_project_sources(project_id: int) -> Dict[str, Any]:
    """Get all sources saved in a research project."""
    if not project_id:
        return {"status": "error", "error": "Project ID is required"}
    
    sources = research_db.get_project_sources(project_id)
    return {
        "status": "success",
        "project_id": project_id,
        "sources": sources,
        "count": len(sources),
        "message": f"Found {len(sources)} sources in project"
    }

@mcp.tool()
def summarize_text(text: str = "", max_sentences: int = 3) -> Dict[str, Any]:
    """Create a simple extractive summary of text by selecting key sentences."""
    clean_text = clean_parameter_input(text)
    if not clean_text:
        return {"status": "error", "error": "Text parameter is required"}
    
    try:
        # Simple sentence extraction based on length and position
        sentences = clean_text.split('.')
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        if not sentences:
            return {
                "status": "error",
                "error": "No suitable sentences found for summarization"
            }
        
        # Score sentences (simple heuristic: prefer longer sentences from beginning)
        scored_sentences = []
        for i, sentence in enumerate(sentences):
            # Score based on length and inverse position
            score = len(sentence.split()) * (1 / (i + 1))
            scored_sentences.append((score, sentence))
        
        # Sort by score and take top sentences
        scored_sentences.sort(reverse=True)
        top_sentences = [sent for _, sent in scored_sentences[:max_sentences]]
        
        summary = '. '.join(top_sentences) + '.'
        
        return {
            "status": "success",
            "original_length": len(clean_text),
            "summary_length": len(summary),
            "summary": summary,
            "compression_ratio": round(len(summary) / len(clean_text), 2)
        }
        
    except Exception as e:
        return {"status": "error", "error": f"Summarization failed: {str(e)}"}

@mcp.tool()
def get_search_history(limit: int = 10) -> Dict[str, Any]:
    """Get recent search history."""
    history = research_db.get_search_history(limit)
    return {
        "status": "success",
        "history": history,
        "count": len(history),
        "message": f"Retrieved {len(history)} recent searches"
    }

@mcp.tool()
def generate_citations(project_id: int, style: str = "apa") -> Dict[str, Any]:
    """Generate citations for all sources in a project."""
    if not project_id:
        return {"status": "error", "error": "Project ID is required"}
    
    sources = research_db.get_project_sources(project_id)
    
    citations = []
    for source in sources:
        if style.lower() == "apa":
            # Simple APA-style citation
            domain = urlparse(source["url"]).netloc if source["url"] else "Unknown"
            date = source["created_at"][:10] if source["created_at"] else "n.d."
            
            citation = f"{domain}. ({date}). {source['title']}. Retrieved from {source['url']}"
        else:
            # Default format
            citation = f"{source['title']} - {source['url']}"
        
        citations.append({
            "source_id": source["id"],
            "title": source["title"],
            "url": source["url"],
            "citation": citation
        })
    
    return {
        "status": "success",
        "project_id": project_id,
        "style": style,
        "citations": citations,
        "count": len(citations)
    }

# ========== Main ==========

if __name__ == "__main__":
    try:
        logger.info("Starting Research Assistant MCP Server")
        logger.info(f"Database: {DB_PATH}")
        logger.info(f"Web search available: {WEB_SEARCH_AVAILABLE}")
        logger.info("Server starting on http://127.0.0.1:8000")
        
        mcp.run(transport="streamable-http")
        
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)
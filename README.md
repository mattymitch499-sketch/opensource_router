 # OpenSource_Router                                                                       
An AI task router that classifies coding projects based on complexity and routes them to    
either Claude Code or a local model (Deepseek), so that it saves tokens and begins to       
create a local model that is almost as good.                                                
                                                                                            
## Why This Exists                                                                          
Claude Code is powerful but every prompt costs tokens, and creating multiple agents to run  
on your computer costs even more. This project was created to solve this problem. Now,      
everytime there is a coding problem, this code analyzes it through a local LLM to see if it 
 is necessary for claude tokens to be burned or not.                                        
                                                                                            
## How It Works                                                                             
                                                                                            
```mermaid                                                                                  
  flowchart TD                                                                              
      A[You type a prompt] --> B[Classifier - Qwen 2.5 3B\nScores complexity 1-10]          
      B -->|Score ≤ 5| C[DeepSeek R1 14B\nLocal - Free]                                     
      B -->|Score > 5| D[Claude Code\nCLI]                                                  
      C --> E[Reviewer - Qwen 2.5\nChecks DeepSeek output]                                  
      E -->|Pass| F[Return result]                                                          
      E -->|Fails twice| D                                                                  
      D --> F                                                                               
```                                                                                       
Every interaction (prompt, score, which model handled it), is recorded into a SQLite        
database where it is saved for Deepseek. This will allow Deepseek to improve gradually over 
 time with more parameters being fed into it. That way, it will begin to take more and more 
 tasks off of claude's hands, until it is finally strong enough of a model to run as its    
own agent locally on a computer.                                                            
                                                                                            
### Complexity Scoring Dimensions                                                           
                                                                                            
  | Dimension | Low (1) | High (10) |                                                       
  |-----------|---------|-----------|                                                       
  | **Scope** | Single function | Multi-file changes |                                      
  | **Reasoning** | Boilerplate | Novel algorithms |                                        
  | **Debugging** | New code | Diagnosing existing bugs |                                   
  | **Architecture** | Isolated task | Requires broad codebase understanding |              
  | **Domain** | Simple CRUD | Auth, concurrency, ML |                                      
                                                                                            
  The final score is a weighted average. The routing threshold is configurable (default:    
5).                                                                                         
                                                                                            
  ## Features                                                                               
                                                                                            
  - **Intelligent routing** — automatically classifies tasks and picks the right model      
  - **Web UI + CLI** — browser-based chat interface (Flask) or command-line, your choice    
  - **Testing agent** — reviews DeepSeek's output and auto-escalates to Claude if it fails  
  twice                                                                                     
  - **Example bank** — SQLite database logs every interaction for history and learning      
  - **Few-shot learning** — past successful solutions get injected into future prompts      
  - **Session resume** — reconnect to your last Claude Code conversation with `--resume`    
  - **Fully configurable** — threshold, models, prompts, and URLs all live in one config    
  file                                                                                      
                                                                                            
 ## Prerequisites                                                                           
                                                                                            
  - **Python 3.x**                                                                          
  - **[Ollama](https://ollama.com/)** installed and running                                 
  - **Ollama models pulled:**                                                               
    ```bash                                                                                 
    ollama pull deepseek-r1:14b                                                             
    ollama pull qwen2.5:3b                                                                  
    ```                                                                                     
  - **[Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code)** installed and     
  authenticated                                                                             
                                                                                            
## Installation                                                                             
                                                                                            
  ```bash                                                                                   
  git clone https://github.com/mattymitch499-sketch/OpenSource_Router.git                          
  cd OpenSource_Router                                                                      
  pip install -r requirements.txt                                                           
  ```                                                                                       
                                                                                            
  Dependencies are minimal — just `flask` and `requests`.                                   
                                                                                            
  ### Web UI (recommended)                                                                  
                                                                                            
  ```bash                                                                                   
  python router_app.py                                                                      
  ```                                                                                       
                                                                                            
  Then open [http://localhost:5000](http://localhost:5000) in your browser.                 
                                                                                            
  ### CLI                                                                                   
                                                                                            
  ```bash                                                                                   
  # Send a coding task                                                                      
  python router.py "write a function that validates email addresses"                        
                                                                                            
  # Point it at a specific project directory                                                
  python router.py --project-dir "C:\path\to\project" "refactor the auth module"            
                                                                                            
  # Resume your last Claude Code session                                                    
  python router.py --resume                                                                 
                                                                                            
  # Resume and send a new prompt                                                            
  python router.py --resume "add error handling to that last function"                      
  ```                                                                                       
                                                                                            
  ## Configuration                                                                          
                                                                                            
  All settings live in `config.py`:                                                         
                                                                                            
  | Setting | Default | Description |                                                       
  |---------|---------|-------------|                                                       
  | `COMPLEXITY_THRESHOLD` | `5` | Score cutoff — at or below goes to DeepSeek, above goes  
  to Claude |                                                                               
  | `CODER_MODEL` | `deepseek-r1:14b` | Model used for code generation |                    
  | `CLASSIFIER_MODEL` | `qwen2.5:3b` | Model used for complexity scoring |                 
  | `REVIEWER_MODEL` | `qwen2.5:3b` | Model used for reviewing DeepSeek output |            
  | `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API address |                     
  | `WEB_PORT` | `5000` | Port for the web UI |                                             
                                                                                            
  System prompts are in the `prompts/` folder and can be edited to change model behavior.   
                                                                                            
  ## Project Structure                                                                      
                                                                                            
  ```                                                                                       
  OpenSource_Router/                                                                        
  ├── router_app.py        # Web UI entry point (Flask + SSE streaming)                     
  ├── router.py            # CLI entry point                                                
  ├── classifier.py        # Complexity scoring logic                                       
  ├── deepseek_agent.py    # DeepSeek code generation via Ollama                            
  ├── claude_agent.py      # Claude Code dispatch via subprocess                            
  ├── testing_agent.py     # Code review + auto-escalation logic                            
  ├── example_bank.py      # SQLite storage (save/retrieve past interactions)               
  ├── config.py            # All configuration in one place                                 
  ├── requirements.txt     # Python dependencies                                            
  ├── templates/                                                                            
  │   └── index.html       # Chat UI (HTML/CSS/JS)                                          
  └── prompts/                                                                              
      ├── router_system.txt      # Classifier system prompt                                 
      ├── coder_system.txt       # DeepSeek coding prompt                                   
      ├── claude_coder_system.txt # Claude coding prompt                                    
      ├── reviewer_system.txt    # Code reviewer prompt                                     
      └── tagger_system.txt      # Technique tagger prompt                                  
  ```                                                    

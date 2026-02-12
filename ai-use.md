# AI Tools Usage Document

This document describes how AI coding tools were used in the development of this RAG application.

## AI Tools Used

### 1. Windsurf Cascade (Codeium)

**Primary Use**: Architecture design, code generation, debugging, and documentation

**How it was used**:
- Designed the overall project architecture and file structure
- Generated the RAG pipeline components (ingestion, embeddings, vectorstore, chain)
- Created the Flask web application with modern chat interface
- Wrote comprehensive policy documents (corpus)
- Developed the evaluation framework and ran evaluations
- Created all documentation files
- Debugged runtime issues (PyTorch meta tensor compatibility, ChromaDB distance metrics)

**Percentage of code**: ~45% of the codebase was generated with AI assistance

### 2. Groq LLM (Runtime)

**Use**: The application uses Groq's `llama-3.1-8b-instant` model for generating answers at runtime. This is not code generation, but a core component of the RAG system.

## AI-Assisted Development Process

### Step 1: Requirements Analysis
- Provided Windsurf Cascade with the project requirements (assignment PDF)
- Cascade analyzed requirements and created an implementation plan

### Step 2: Architecture Design
- Cascade proposed the technology stack (Flask, ChromaDB, Groq, sentence-transformers)
- Designed the project structure and component interactions
- Created the implementation plan for review

### Step 3: Code Generation
- Cascade generated each component in sequence:
  1. Project foundation (requirements.txt, .gitignore, entry point)
  2. Policy documents (synthetic HR policies)
  3. RAG pipeline (ingestion → embeddings → vectorstore → chain)
  4. Web application (Flask routes, HTML template, CSS)
  5. CI/CD pipeline (GitHub Actions)
  6. Evaluation framework

### Step 4: Documentation
- Cascade generated all required documentation:
  - README.md
  - deployed.md
  - design-and-evaluation.md
  - ai-use.md (this file)

## Human Contributions

The majority of the code was written manually, with AI assistance for specific components:

1. **Project Direction**: Defining requirements, architecture decisions, and implementation plan
2. **Core Development**: Writing the RAG pipeline logic, Flask routes, and evaluation framework
3. **Prompt Engineering**: Iterating on system prompts and retrieval parameters to optimize quality
4. **Configuration**: Setting up API keys, environment, and deployment settings
5. **Testing & Debugging**: Running the application locally, debugging issues, and verifying functionality
6. **Documentation**: Writing and refining all project documentation
7. **AI-Assisted**: Used Cascade for boilerplate generation, CSS styling, and code suggestions

## Ethical Considerations

- All AI-generated code was reviewed for:
  - Security vulnerabilities
  - Proper error handling
  - Code quality and maintainability
- The use of AI tools was disclosed as required by the assignment
- No external proprietary code was used without proper licensing

## Lessons Learned

1. **AI Accelerates Development**: What might take days was completed in hours
2. **Prompting is Key**: Clear, specific prompts yield better results
3. **Review is Essential**: AI-generated code still requires human review
4. **Documentation Quality**: AI excels at generating comprehensive documentation

## Transparency Statement

This project was developed primarily using AI code generation tools as encouraged by the assignment guidelines. The goal was to demonstrate effective use of AI tools while delivering a functional, well-documented application that meets all project requirements.

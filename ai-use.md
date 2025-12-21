# AI Tools Usage Document

This document describes how AI coding tools were used in the development of this RAG application.

## AI Tools Used

### 1. Claude (Anthropic)

**Primary Use**: Architecture design, code generation, and documentation

**How it was used**:
- Designed the overall project architecture and file structure
- Generated the RAG pipeline components (ingestion, embeddings, vectorstore, chain)
- Created the Flask web application with modern chat interface
- Wrote comprehensive policy documents (corpus)
- Developed the evaluation framework
- Created all documentation files

**Percentage of code**: ~95% of the codebase was generated with AI assistance

### 2. Groq LLM (Runtime)

**Use**: The application uses Groq's `llama-3.1-8b-instant` model for generating answers at runtime. This is not code generation, but a core component of the RAG system.

## AI-Assisted Development Process

### Step 1: Requirements Analysis
- Provided the AI with the project requirements (assignment PDF)
- AI analyzed requirements and created an implementation plan

### Step 2: Architecture Design
- AI proposed the technology stack (Flask, ChromaDB, Groq, sentence-transformers)
- Designed the project structure and component interactions
- Created the implementation plan for review

### Step 3: Code Generation
- AI generated each component in sequence:
  1. Project foundation (requirements.txt, .gitignore, entry point)
  2. Policy documents (synthetic HR policies)
  3. RAG pipeline (ingestion → embeddings → vectorstore → chain)
  4. Web application (Flask routes, HTML template, CSS)
  5. CI/CD pipeline (GitHub Actions)
  6. Evaluation framework

### Step 4: Documentation
- AI generated all required documentation:
  - README.md
  - deployed.md
  - design-and-evaluation.md
  - ai-use.md (this file)

## Human Contributions

While AI generated the majority of the code, human contributions included:

1. **Project Direction**: Defining requirements and approving the implementation plan
2. **Configuration**: Adding actual API keys and deployment settings
3. **Testing**: Running the application locally and verifying functionality
4. **Deployment**: Configuring Render and deploying the application
5. **Review**: Reviewing and approving AI-generated code

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

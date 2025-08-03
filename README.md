
<br />
<div align="center">
  <h3 align="center">Academic PDF Converser 🎓✨</h3>

  <p align="center">
    Interactively chat with academic papers using LangChain, Gemini API, and HuggingFace embeddings.
    <br />
    Your personal AI research assistant for PDF deep-dives!
    <br />
  </p>
</div>

---

## 📌 Table of Contents

- [About the Project](#about-the-project)
- [Built With](#built-with)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Features](#features)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Authors and Contact](#authors-and-contact)
- [Acknowledgments](#acknowledgments)

---

## About The Project 🚀

Welcome to **Academic PDF Converser** 🎓 — an intelligent Q&A app that lets you engage in natural conversation with academic research papers!

This tool uses **LangChain**, **Google Gemini (1.5-flash)**, and **FAISS vector store** to extract, embed, and search through research documents, allowing you to:

- 💬 Ask deep questions about research PDFs.
- 🔍 Cite exact sources from the paper.
- 🧠 Retain conversation memory across turns.
- ⚡️ Process large documents quickly with chunking + embedding.

> Ideal for researchers, students, and anyone who wants to break down complex academic text into digestible answers.

---

## Built With

* [![Streamlit][streamlit-badge]][streamlit-url]
* [![LangChain][langchain-badge]][langchain-url]
* [![Google Gemini][gemini-badge]][gemini-url]
* [![FAISS][faiss-badge]][faiss-url]
* [![HuggingFace Transformers][hf-badge]][hf-url]
* [![Python][python-badge]][python-url]

[streamlit-badge]: https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white
[streamlit-url]: https://streamlit.io/
[langchain-badge]: https://img.shields.io/badge/LangChain-000000?style=for-the-badge&logo=LangChain&logoColor=white
[langchain-url]: https://www.langchain.com/
[gemini-badge]: https://img.shields.io/badge/Gemini_API-4285F4?style=for-the-badge&logo=google&logoColor=white
[gemini-url]: https://deepmind.google/technologies/gemini/
[faiss-badge]: https://img.shields.io/badge/FAISS-000000?style=for-the-badge&logo=data&logoColor=white
[faiss-url]: https://github.com/facebookresearch/faiss
[hf-badge]: https://img.shields.io/badge/HuggingFace-FFD21F?style=for-the-badge&logo=huggingface&logoColor=black
[hf-url]: https://huggingface.co/
[python-badge]: https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white
[python-url]: https://www.python.org/

---

## Getting Started 🧠

### Prerequisites

- Python 3.8+
- Google Gemini API Key
- pip

### Installation

```bash
git clone https://github.com/abdelrahmanmah/AcademicRAG.git
cd pdf-converser

# (Optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run main.py
```

---

## Usage 📚

1. Upload a research paper (PDF).
2. Enter your Gemini API Key in the sidebar.
3. Ask your question (e.g., "What methods were used in the experiments?" or "What are the main findings?")
4. Get a precise, context-aware answer with cited chunks from the paper!

💬 Conversation memory helps maintain context across multiple questions.

---

## Features ✨

- 🔍 **PDF Parsing**: Extracts clean text using PyMuPDF.
- 🧩 **Chunking & Embedding**: Efficiently splits and indexes documents.
- 🧠 **Conversational Q&A**: LangChain’s memory-powered ConversationalRetrievalChain.
- 🧾 **Cited Sources**: Displays which part of the paper was used to answer.
- ⚙️ **Configurable**: Supports k-context tuning and temperature control.
- 🧪 **Custom Prompt Engineering**: Ensures factual and context-bound answers.
- 🏃‍♂️ **Real-time interaction**: Chat input and output streamed for UX fluidity.

---

## Contributing 🤝

Contributions are welcome!

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/awesome-feature`)
3. Commit your changes (`git commit -m 'Add awesome feature'`)
4. Push to the branch (`git push origin feature/awesome-feature`)
5. Open a Pull Request

---


## Authors and Contact 📞

- **Abdelrahman Mahmoud** – [@abdelrahmanmah](https://github.com/abdelrahmanmah)
- **Gannatullah Asaad** – [@GannaAsaad](https://github.com/GannaAsaad)
- **Ali Mohamed** – [@AliiiMohamedAliii](https://github.com/AliiiMohamedAliii)
- **Ahmed Khaled** – [@Ahmedkhaled51](https://github.com/Ahmedkhaled51)

Project Link: [https://github.com/abdelrahmanmah/AcademicRAG/](https://github.com/abdelrahmanmah/AcademicRAG/)

---

## Acknowledgments 🙏
- LangChain & Gemini Teams for excellent tools.
- Streamlit community for easy UI building.
- [Othneil Drew's Best-README-Template](https://github.com/othneildrew/Best-README-Template) for the inspiration.
- Anyone who hates manually reading dense research PDFs.

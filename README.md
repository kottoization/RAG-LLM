# RAG LLM Project

## Description
The RAG LLM Project is an artificial intelligence system that utilizes RAG (Retrieval-Augmented Generation) and LLM (Large Language Model) to provide answers to questions regarding data from Medium articles. This system enables quick search and content generation based on the given queries.

## Requirements
To run the project, you need to have Python and pip installed. The project also relies on the requirements.txt file, which contains all the necessary dependencies.

## Installation
1. Clone the repository to your local machine.
2. Create a virtual Python environment:
    ```
    python -m venv venv
    ```
3. Activate the virtual environment:
    - Windows:
    ```
    venv\Scripts\activate
    ```
    - macOS/Linux:
    ```
    source venv/bin/activate
    ```
4. Install the dependencies using the requirements.txt file:
    ```
    pip install -r requirements.txt
    ```

## Configuration
The project uses the `.env` file for environment variable configuration. Before running the project, create the `.env` file and add the required variables, such as the OpenAI API key.

Example `.env` file:
 ```
OPENAI_API_KEY=YOUR_OPENAI_API_KEY_HERE
 ```

## Usage
After successfully installing and configuring the project, you can run the main.py file. The project will be ready to use, and the user can input questions to which the system will respond.

Example of running the project:
 ```
python main.py
 ```
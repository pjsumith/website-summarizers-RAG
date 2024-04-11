# Chat with Websites using Language Models

This application allows users to interact with website content using natural language queries. By leveraging state-of-the-art language models and web scraping techniques, users can ask questions and receive answers based on the content extracted from specified websites.

## Features

- **Web Scraping**: The application extracts text content from user-specified website URLs.
- **Natural Language Processing**: Questions posed by users are processed using advanced language models.
- **Contextual Understanding**: Responses are generated based on the context extracted from the website content.

## Installation

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/your_username/chat-with-websites.git
   ```
2. Navigate to the project directory:

    ```bash
    cd chat-with-websites
    ```
3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage
  To start the app, simply run the following command:
  
    ```bash
    streamlit run app.py
    ```
  1. Enter the URL of the website you want to interact with in the sidebar.
  2. Type your question in the designated text input field.
  3. Click the "Query Documents" button to generate the answer.

## Dependencies
- Streamlit: A popular Python library for creating web applications with simple Python scripts.
- Beautiful Soup: A Python library for web scraping.
- Langchain: A library for natural language processing tasks, including text splitting, embeddings, and context retrieval.
- Hugging Face Transformers: A library for state-of-the-art natural language processing models.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Contributing
Contributions are welcome! If you have any ideas, suggestions, or bug fixes, please open an issue or submit a pull request.
#### Author: [Paul Joshi Sumith] (Primary)

## Credits
Special thanks to the developers of Langchain and Hugging Face for their excellent libraries and resources.
Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

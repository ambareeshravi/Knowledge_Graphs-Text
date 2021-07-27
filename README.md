# Knowledge_Graphs-Text

Creating knowledge graphs from entities in large chunks of text

A tool that creates and visualizes a knowledge from textual data using Natural Language Processing. Has applications in medicine, finance, recommendation systems, fraud detection, trading etc.

1. Install the required dependencies from requirements.txt
    ```python
    pip3 install -r requirements.txt
    ```
2. Change data loading as required or you could use the inbuilt method to import csv files

3. Run the tool by running the script with the data path and relationship as follows:
    ```bash
    python3 text_knowledge_graph.py --data_path data/wikipedia_sentences.csv --relationship includes
    ```
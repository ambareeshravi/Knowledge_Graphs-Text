#!/usr/bin/python3

'''
Author: Ambareesh Ravi
Date: 26 July, 2021
File: text_knowledge_graph.py
Description:
    Creates and visualizes a knowledge from textual data using Natural Language Processing.
    Has applications in medicine, finance, recommendation systems, fraud detection, trading etc.
'''

# Library imports
import numpy as np
import pandas as pd

import spacy
from spacy.matcher import Matcher

import networkx as nx
import matplotlib.pyplot as plt

from tqdm import tqdm

# Module imports
from data import *

# Global variables
# Change to different language pack as required
nlp = spacy.load('en_core_web_sm')

class TextKnowledgeGraph:
    # Creates and visualizes a knowledge graph from textual data
    def __init__(self, data):
        '''
        Initializes the class

        Args:
            data - the text data as <pandas.DataFrame>
        Returns:
            -
        Exception:
            -
        '''
        self.data = data

        # define pattern matching params
        self.matcher = Matcher(nlp.vocab)
        pattern = [
            {'DEP':'ROOT'},
            {'DEP':'prep','OP':"?"},
            {'DEP':'agent','OP':"?"},
            {'POS':'ADJ','OP':"?"}
        ]
        self.matcher.add("matching_1", None, pattern)

        # Build the knowledge graph
        self.build()

    def extract_entities(self, sentence):
        '''
        Extracts entities from a sentence using Spacy dependency parser

        Args:
            sentence - the input sentence as <str>
        Returns:
            pair of entities as <list>
        Exception:
            -
        '''

        entity1, entity2, prefix, modifier, prev_token_dep, prev_token_text = "", "", "", "", "", ""
        
        for token in nlp(sentence):
            # Skip punctuation
            if token.dep_ == "punct": continue
                
            # Check for compound sentence/ words
            if token.dep_ == "compound":
                prefix = token.text
                # Check for and add the previous compound words
                if prev_token_dep == "compound":
                    prefix = "%s %s"%(prev_token_text, token.text)
                    
            # Check if token is a modifier
            if token.dep_.endswith("mod") == True:
                modifier = token.text
                # Check for and add the previous compound words
                if prev_token_dep == "compound":
                    modifier = "%s %s"%(prev_token_text, token.text)
                    
            # Check if the word/ token is the subject
            if token.dep_.find("subj") == True:
                entity1 = "%s %s %s"%(modifier, prefix, token.text)
                prefix, modifier, prev_token_dep, prev_token_text = "", "", "", ""
            
            # Check if the word/ token is the object
            if token.dep_.find("obj") == True:
                entity2 = "%s %s %s"%(modifier, prefix, token.text)
            
            # Update values
            prev_token_dep, prev_token_text = token.dep_, token.text
        
        # Return results
        return [entity1.strip(), entity2.strip()]


    def extract_relations(self, sentence):
        '''
        Extracts the relationships in the sentence

        Args:
            sentence - the input sentence as <str>
        Returns:
            relationship as <str>
        Exception:
            -
        '''

        doc = nlp(sentence)
        matches = self.matcher(doc)
        span = doc[matches[-1][1]:matches[-1][2]]
        return span.text

    def get_knowledge_graph_data(self, entity_pairs, relations):
        '''
        Creates and returns as dataframe for knowledge graph creation

        Args:
            entity_pairs - <list> of all entity pairs in the dataset
            relations - <list> of all relationships between the entity pairs in the dataset
        Returns:
            data as <pandas.DataFrame>
        Exception:
            -
        '''

        ep_array = np.array(entity_pairs)
        # subject [source] -> object [target]
        kd_df = pd.DataFrame(
            {
                "source": ep_array[:,0],
                "target": ep_array[:,1],
                "edge": relations
            }
        )
        return kd_df

    def create_network(self, kd_df, key_relation = None):
        '''
        Creates directed graph from knowledge graph dataframe

        Args:
            kd_df - knowledge graph data as <pandas.DataFrame>
            key_relation - a particular relationship to look for <str>
        Returns:
            graph as <nx.MultiDiGraph>
        Exception:
            -
        '''
        
        dir_graph = nx.from_pandas_edgelist(
            df = kd_df[kd_df['edge'] == key_relation] if key_relation else kd_df,
            source = 'source',
            target = 'target',
            edge_attr = True,
            create_using = nx.MultiDiGraph()
        )
        return dir_graph

    def plot_graph(self, dir_graph, figsize = (12,12), node_spacing = 0.5, node_size = 1000, node_color = 'skyblue'):
        '''
        Plots and displays the knowledge graph using matplotlib.pyplot

        Args:
            dir_graph - knowledge graph as <nx.MultiDiGraph>
            figsize - size of the figure as a <tuple>
            node_spacing - parameter to adjust the distance between nodes in the graph as <float>
            node_size - maximum number of nodes as <int>
            node_color - colour for the nodes as <str> [correspondingly color map has to be changed]
        Returns:
            -
        Exception:
            -
        '''
        
        plt.figure(figsize = figsize)
        pos = nx.spring_layout(dir_graph, k = node_spacing)
        nx.draw(dir_graph, with_labels = True, node_color = node_color, node_size = node_size, edge_cmap = plt.cm.Blues, pos = pos)
        plt.show()       

    def build(self,):
        '''
        Builds the knowledge graph internally and stores it in a dataframe

        Args:
            -
        Returns:
            -
        Exception:
            -
        '''
        entity_pairs = [self.extract_entities(sent) for sent in tqdm(self.data["sentence"])]
        relations = [self.extract_relations(sent) for sent in tqdm(self.data['sentence'])]
        self.kd_df = self.get_knowledge_graph_data(entity_pairs, relations)

    def get_by_relationship(self, relationship):
        '''
        Dynamically generates and visualizes the part of the graph based on the relationship

        Args:
            relationship - key relationship to look for as <str>
        Returns:
            -
        Exception:
            -
        '''
        dir_graph = self.create_network(self.kd_df, relationship)
        self.plot_graph(dir_graph)

if __name__ == '__main__':
    # Load data
    data = Dataset("data/wiki_sentences_v2.csv")
    # Create an object for the knowledge graph
    kg = TextKnowledgeGraph(data()) # data() is same as data.df

    # Visualize based on the relationships
    kg.get_by_relationship("written by")
    kg.get_by_relationship("directed by")
    kg.get_by_relationship("includes")
    kg.get_by_relationship("composed by")
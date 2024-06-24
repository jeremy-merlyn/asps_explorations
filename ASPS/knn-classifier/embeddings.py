import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch
from sklearn.model_selection import train_test_split


csv_file = "../data/asps_router_eval.csv"


df = pd.read_csv(csv_file)

# embedder = SentenceTransformer('all-MiniLM-L6-v2')
embedder = SentenceTransformer('msmarco-MiniLM-L-6-v3')

def split(df, seed=42):
    train, test = train_test_split(df, test_size=0.05, random_state=seed)

    # def remove_stop_words(text):
    #     stop_words = ['a', 'an', 'the', 'and', 'or', 'for', 'is']
    #     return ' '.join([word for word in text.split() if word.lower() not in stop_words])

    # train['query'] = train['query'].apply(remove_stop_words)
    # test['query'] = test['query'].apply(remove_stop_words)


    train_queries = list(train['query'])
    routes =  list(train['route'])

    return train_queries, routes, test


def guess(query, embeddings, routes, top_k=5):

    query_embedding = embedder.encode(query, convert_to_tensor=True)

    #use cosine-similarity and torch.topk to find the highest 5 scores
    cos_scores = util.cos_sim(query_embedding, embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)

    print("Query:", query)
    # print("\nTop guesses: \n")

    # print("top_results", top_results)

    route_guesses = {}

    for score, idx in zip(top_results[0], top_results[1]):
        # print(corpus[idx], "(Score: {:.4f})".format(score))
        guess = routes[idx]
        print(guess)
        # print(guess, "(Score: {:.4f})".format(score))
        route_guesses[guess] = route_guesses.get(guess, 0) + float(score**2)

    sorted_tuples = sorted(route_guesses.items(), key=lambda x: x[1], reverse=True)

    # return the most common guess
    return max(route_guesses, key=route_guesses.get)


# perform gueses for all the test queries and write to csv

def test_run(seed):
    train_queries, routes, test = split(df,seed)

    # Corpus with example sentences to embed    
    embeddings = embedder.encode(train_queries, convert_to_tensor=True)


    guesses = []
    for query in test['query']:
        guesses.append(guess(query, embeddings, routes, top_k=5))

    test['guess'] = guesses

    #add column for correctness
    test['correctness'] = test['guess'] == test['route']

    #sort by correctness with incorrect first
    test = test.sort_values(by=['correctness', 'route', 'guess'])

    # calculate accuracy and print
    accuracy = test['correctness'].mean()

    return accuracy


accuracies = []
for i in range(20):
    accuracies.append(test_run(i))

print("sorted accuracies: ", sorted(accuracies, reverse=True))
print("mean accuracy: ", np.mean(accuracies))
print("std accuracy: ", np.std(accuracies))







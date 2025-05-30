from sentence_transformers import SentenceTransformer, util


def main():
    # Load the model

    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Input sentences
    sentence1 = input("sentence1: ")
    sentence2 = input("sentence2: ")

    f= open("tested_sentence_pairs.txt","a")
    f.write(sentence1 + "\n")
    f.write(sentence2 + "\n")

    # Generate embeddings

    embeddings = model.encode([sentence1, sentence2], convert_to_tensor=True)

    print(embeddings)
    # Compute cosine similarity
    similarity_score = util.pytorch_cos_sim(embeddings[0], embeddings[1])

    print(f"Similarity score: {similarity_score.item():.8f}")
    f.write(str(similarity_score.item())+"\n\n")
    f.close()

if __name__ == "__main__":
    main()
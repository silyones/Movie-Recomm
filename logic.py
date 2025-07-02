from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

# ✅ INIT Pinecone
pc = Pinecone(api_key="pcsk_54MTSn_7bd9AMjy7d9o5aECcqbtkNtHFcBv52BjH5R8LYzpccTvKo5Q9sqv5QnoiFGWUuv")

# ✅ INIT local embedding model (no API key needed!)
model = SentenceTransformer("all-MiniLM-L6-v2")  # 384-dimension

# 📌 Create index if not exists
index_name = "movie-index"
dimension = 384

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=dimension,
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

index = pc.Index(index_name)

# 🧠 Function to embed text locally
def get_embedding(text):
    return model.encode(text).tolist()

# 🎬 Movies list
movies = [
    {
        "id": "1",
        "title": "The Matrix",
        "genre": "sci-fi",
        "description": "A computer hacker learns the true nature of reality and his role in the war against its controllers.",
        "year": 1999
    },
    {
        "id": "2",
        "title": "Inception",
        "genre": "action",
        "description": "A thief who steals corporate secrets through dream-sharing technology is given a task to plant an idea.",
        "year": 2010
    },
    {
        "id": "3",
        "title": "The Notebook",
        "genre": "romance",
        "description": "A young couple falls in love in the 1940s but are separated by fate and societal differences.",
        "year": 2004
    },
    {
        "id": "4",
        "title": "Star Wars: A New Hope",
        "genre": "sci-fi",
        "description": "Luke Skywalker joins forces with a Jedi Knight to save the galaxy from the Empire's Death Star.",
        "year": 1977
    },
    {
        "id": "5",
        "title": "Blade Runner",
        "genre": "sci-fi",
        "description": "A blade runner must pursue and terminate replicants who have escaped to Earth.",
        "year": 1982
    },
    {
        "id": "6",
        "title": "Frozen",
        "genre": "Fantasy",
        "description": "A princess who has the super power of ice. She must save her kingdom from eternal winter. ",
        "year": 2022
    }
]


# ✨ Step 3: Upsert movie vectors
vectors = []
for movie in movies:
    vector = get_embedding(movie["description"])
    vectors.append({
        "id": movie["id"],
        "values": vector,
        "metadata": {
            "title": movie["title"],
            "genre": movie["genre"],
            "year": movie["year"]
        }
    })

index.upsert(vectors=vectors)

# 🔍 Step 4: Search similar to "Dune", only movies before 2010
query_description = "A noble family becomes embroiled in a war for control over a desert planet with a valuable spice."
query_vector = get_embedding(query_description)

results = index.query(
    vector=query_vector,
    top_k=5,
    include_metadata=True,
    filter={"year": {"$lt": 2010}}  # Only older movies
)

# 🖨️ Step 5: Show results
print("\n🎬 Movies similar to Dune (before 2010):\n")
for match in results["matches"]:
    meta = match["metadata"]
    print(f"{meta['title']} ({meta['year']}) - Score: {match['score']:.4f}")

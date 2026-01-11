from query_engine import get_chatbot, get_db_context

def main():
    print("🧠 Loading chatbot...")
    client = get_chatbot()
    print("✅ Chatbot ready. Ask your questions!")

    while True:
        query = input("\n💬 You: ")
        if query.lower() in ["exit", "quit"]:
            break

        # Get database context for the query
        db_context = get_db_context(query)
        
        # Combine user query with database context
        enhanced_query = f"Context from EV database: {db_context}\n\nUser question: {query}"

        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": enhanced_query}],
            temperature=1,
            max_completion_tokens=1024,
            top_p=1,
            stream=True,
            stop=None
        )

        print("\n🤖 Answer: ", end="")
        for chunk in completion:
            print(chunk.choices[0].delta.content or "", end="")
        print()

if __name__ == "__main__":
    main()

from rag import llm_result

question = "Why did Chandra Textiles' margins contract in Q1FY27?"

print(f"Question: {question}")
print("-" * 50)

answer = llm_result(question)
print(f"Answer:\n{answer}\n")
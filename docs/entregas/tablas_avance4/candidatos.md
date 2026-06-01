| Candidato | Modelo QA nube | Modelo QA local | Embedding | Retriever | Expansion | Reranker | Fine-tuning | Prompt versionado | NDCG nube | NDCG local | Lat. nube | Lat. local | Costo/1k nube |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1_Baseline_Léxico | gpt-4o-mini | llama3.1:8b | No aplica | bow | No | No | No | qa_rag v1.0.0 (2cbbc4f704) | 0.7140 | 0.8471 | 0.303 | 0.404 | 0.5101 |
| 2_Baseline_Semántico | gpt-4o-mini | llama3.1:8b | text-embedding-3-small / ChromaDB | embeddings | No | No | No | qa_rag v1.0.0 (2cbbc4f704) | 0.8457 | 0.9080 | 0.280 | 0.389 | 0.5070 |
| 3_Híbrido_Simple | gpt-4o-mini | llama3.1:8b | text-embedding-3-small / ChromaDB | hibrido | No | No | No | qa_rag v1.0.0 (2cbbc4f704) | 0.8261 | 0.9215 | 0.508 | 1.508 | 0.5109 |
| 4_Híbrido_Reranker | gpt-4o-mini | llama3.1:8b | text-embedding-3-small / ChromaDB | hibrido | No | cross_encoder | No | qa_rag v1.0.0 (2cbbc4f704) | 0.8301 | 0.9157 | 2.824 | 3.028 | 0.5056 |
| 5_Semántico_Expandido | gpt-4o-mini | llama3.1:8b | text-embedding-3-small / ChromaDB | embeddings | ambos | No | No | qa_rag v1.0.0 (2cbbc4f704) | 0.8439 | 0.9043 | 4.837 | 8.797 | 0.5110 |
| 6_SOTA_Completo | gpt-4o-mini | llama3.1:8b | text-embedding-3-small / ChromaDB | hibrido | ambos | cross_encoder | No | qa_rag v1.0.0 (2cbbc4f704) | 0.8407 | 0.9277 | 9.377 | 13.903 | 0.5023 |

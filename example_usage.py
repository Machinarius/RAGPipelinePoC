#!/usr/bin/env python3
"""
Example usage script demonstrating how to use the RAG Pipeline with both Ollama and OpenAI.

This script shows how to:
1. Test ingestion with different embedding providers
2. Test both LLM providers for queries
3. Make API calls to the running server
4. Compare responses between providers
"""

import os
import json
import requests
import time
import subprocess
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def check_server_running(port=8500):
    """Check if the RAG server is running"""
    try:
        response = requests.get(f"http://localhost:{port}/docs", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def check_prerequisites():
    """Check if all prerequisites are met"""
    print("Checking prerequisites...")

    # Check ChromaDB
    try:
        chroma_host = os.getenv("CHROMA_HOST", "localhost:8000")
        response = requests.get(f"http://{chroma_host}/api/v1/heartbeat", timeout=5)
        print("✓ ChromaDB is running")
        chroma_ok = True
    except:
        print("✗ ChromaDB is not running. Please start it first:")
        print("  docker run -p 8000:8000 chromadb/chroma")
        chroma_ok = False

    # Check Ollama configuration
    ollama_ok = bool(
        os.getenv("OLLAMA_BASE_URL")
        and os.getenv("GENERATION_MODEL")
        and os.getenv("EMBEDDING_MODEL")
    )
    if ollama_ok:
        print("✓ Ollama configuration found")
    else:
        print("✗ Missing Ollama configuration variables")

    # Check OpenAI configuration
    openai_key = os.getenv("OPENAI_API_KEY")
    openai_ok = openai_key and openai_key != "your_openai_api_key_here"
    if openai_ok:
        print("✓ OpenAI API key configured")
    else:
        print("! OpenAI API key not configured (optional)")

    return chroma_ok, ollama_ok, openai_ok


def ingest_document(file_path, use_openai=False):
    """Ingest a document using the specified embedding provider"""
    if not os.path.exists(file_path):
        print(f"✗ Document not found: {file_path}")
        return False

    provider = "OpenAI" if use_openai else "Ollama"
    print(f"\nIngesting document with {provider} embeddings...")

    cmd = ["python", "rag_cli.py", "ingest", file_path]
    if use_openai:
        cmd.append("--openai")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode == 0:
            print(f"✓ Successfully ingested document with {provider} embeddings")
            return True
        else:
            print(f"✗ Ingestion failed: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print(f"✗ Ingestion timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"✗ Ingestion error: {e}")
        return False


def start_server(provider="ollama", port=8500):
    """Start the RAG server with specified provider"""
    print(f"Starting server with {provider.upper()} on port {port}...")

    cmd = ["python", "rag_cli.py", "server", "--port", str(port)]
    if provider.lower() == "openai":
        cmd.append("--openai")

    # Start server in background
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    # Wait for server to start (max 30 seconds)
    for _ in range(30):
        if check_server_running(port):
            print(f"✓ Server started successfully with {provider.upper()}")
            return process
        time.sleep(1)

    print(f"✗ Failed to start server with {provider.upper()}")
    process.terminate()
    return None


def test_query(query, k=5, port=8500):
    """Send a test query to the RAG server"""
    url = f"http://localhost:{port}/ask"
    data = {"query": query, "k": k}

    try:
        response = requests.post(url, json=data, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


def demo_ingestion_workflow():
    """Demonstrate different ingestion workflows"""
    print("\n" + "=" * 60)
    print("DOCUMENT INGESTION DEMO")
    print("=" * 60)

    # Look for a sample PDF file
    sample_files = [
        "Codigo-nacional-de-transito.pdf",
        "sample.pdf",
        "document.pdf",
        "test.pdf",
    ]

    sample_file = None
    for file in sample_files:
        if os.path.exists(file):
            sample_file = file
            break

    if not sample_file:
        print("No sample PDF found. Please provide a PDF file for ingestion demo.")
        return False

    print(f"Using sample file: {sample_file}")

    # Check what providers are available
    _, ollama_ok, openai_ok = check_prerequisites()

    print("\nAvailable ingestion options:")
    if ollama_ok:
        print("1. Ingest with Ollama embeddings (free, local)")
    if openai_ok:
        print("2. Ingest with OpenAI embeddings (paid, cloud)")
    if ollama_ok and openai_ok:
        print("3. Compare both embedding providers")

    if not ollama_ok and not openai_ok:
        print("No embedding providers available!")
        return False

    choice = input("Enter choice (1-3): ").strip()

    if choice == "1" and ollama_ok:
        return ingest_document(sample_file, use_openai=False)
    elif choice == "2" and openai_ok:
        return ingest_document(sample_file, use_openai=True)
    elif choice == "3" and ollama_ok and openai_ok:
        print("\nIngesting with both providers for comparison...")
        ollama_result = ingest_document(sample_file, use_openai=False)
        time.sleep(2)  # Brief pause
        openai_result = ingest_document(sample_file, use_openai=True)
        return ollama_result and openai_result
    else:
        print("Invalid choice or provider not available")
        return False


def compare_providers(queries, port=8500):
    """Compare responses from both providers"""
    results = {}
    _, ollama_ok, openai_ok = check_prerequisites()

    for provider in ["ollama", "openai"]:
        if provider == "ollama" and not ollama_ok:
            continue
        if provider == "openai" and not openai_ok:
            continue

        print(f"\n{'=' * 50}")
        print(f"Testing {provider.upper()} Provider")
        print(f"{'=' * 50}")

        # Start server with current provider
        server_process = start_server(provider, port)
        if not server_process:
            continue

        try:
            results[provider] = []

            for i, query in enumerate(queries, 1):
                print(f"\nQuery {i}: {query}")
                print("-" * 40)

                start_time = time.time()
                result = test_query(query, port=port)
                end_time = time.time()

                if "error" in result:
                    print(f"✗ Error: {result['error']}")
                else:
                    print(f"✓ Response time: {end_time - start_time:.2f}s")
                    print(f"Answer: {result['answer'][:200]}...")
                    print(f"Sources: {len(result['source_documents'])} documents")

                result["response_time"] = end_time - start_time
                result["query"] = query
                results[provider].append(result)

                time.sleep(1)  # Brief pause between queries

        finally:
            # Stop the server
            print(f"\nStopping {provider.upper()} server...")
            server_process.terminate()
            time.sleep(2)

    return results


def generate_comparison_report(results):
    """Generate a comparison report"""
    if not results:
        print("No results to compare.")
        return

    print(f"\n{'=' * 60}")
    print("COMPARISON REPORT")
    print(f"{'=' * 60}")

    providers = list(results.keys())

    if len(providers) < 2:
        print("Need at least 2 providers to compare.")
        return

    # Compare response times
    print("\nResponse Times:")
    print("-" * 20)
    for provider in providers:
        times = [
            r.get("response_time", 0) for r in results[provider] if "error" not in r
        ]
        if times:
            avg_time = sum(times) / len(times)
            print(f"{provider.upper()}: {avg_time:.2f}s average")

    # Compare answers for each query
    print("\nAnswer Comparison:")
    print("-" * 20)

    max_queries = max(len(results[provider]) for provider in providers)

    for i in range(max_queries):
        print(f"\nQuery {i + 1}:")
        for provider in providers:
            if i < len(results[provider]):
                result = results[provider][i]
                if "error" not in result:
                    answer = (
                        result["answer"][:100] + "..."
                        if len(result["answer"]) > 100
                        else result["answer"]
                    )
                    print(f"  {provider.upper()}: {answer}")
                else:
                    print(f"  {provider.upper()}: ERROR - {result['error']}")


def main():
    """Main example function"""
    print("RAG Pipeline Complete Usage Example")
    print("=" * 50)

    # Check prerequisites
    chroma_ok, ollama_ok, openai_ok = check_prerequisites()

    if not chroma_ok:
        print("\nCannot proceed without ChromaDB. Please start it first.")
        return 1

    print(f"\nChoose demo mode:")
    print("1. Document ingestion workflow")
    print("2. Query testing (requires pre-ingested documents)")
    print("3. Full pipeline demo (ingestion + queries)")
    print("4. Provider comparison")

    choice = input("Enter choice (1-4): ").strip()

    if choice == "1":
        # Ingestion demo
        success = demo_ingestion_workflow()
        if success:
            print("\n✓ Ingestion demo completed successfully!")
            print("You can now run queries using option 2.")
        else:
            print("\n✗ Ingestion demo failed.")

    elif choice == "2":
        # Query testing
        print("\nTesting queries on existing collections...")

        test_queries = [
            "What is the speed limit in urban areas?",
            "What are the requirements for vehicle registration?",
            "What penalties exist for traffic violations?",
        ]

        print("Available providers:")
        if ollama_ok:
            print("1. Test with Ollama")
        if openai_ok:
            print("2. Test with OpenAI")

        provider_choice = input("Choose provider (1-2): ").strip()
        provider = "ollama"
        if provider_choice == "2" and openai_ok:
            provider = "openai"
        elif provider_choice == "1" and not ollama_ok:
            print("Ollama not available")
            return 1

        server = start_server(provider)
        if server:
            try:
                print(f"\nTesting queries with {provider.upper()}...")
                for query in test_queries:
                    print(f"\nQuery: {query}")
                    result = test_query(query)
                    if "error" not in result:
                        print(f"Answer: {result['answer'][:200]}...")
                        print(f"Sources: {len(result['source_documents'])}")
                    else:
                        print(f"Error: {result['error']}")
            finally:
                server.terminate()

    elif choice == "3":
        # Full pipeline demo
        print("\nRunning full pipeline demo...")

        # Step 1: Ingestion
        if not demo_ingestion_workflow():
            print("Ingestion failed, cannot proceed with queries")
            return 1

        # Step 2: Wait for ingestion to settle
        print("\nWaiting for ingestion to complete...")
        time.sleep(5)

        # Step 3: Query testing
        test_queries = [
            "What is the speed limit in urban areas?",
            "What are the requirements for vehicle registration?",
        ]

        provider = "ollama" if ollama_ok else "openai"
        server = start_server(provider)
        if server:
            try:
                print(f"\nTesting queries with {provider.upper()}...")
                for query in test_queries:
                    print(f"\nQuery: {query}")
                    result = test_query(query)
                    if "error" not in result:
                        print(f"Answer: {result['answer'][:150]}...")
                    else:
                        print(f"Error: {result['error']}")
            finally:
                server.terminate()

    elif choice == "4":
        # Provider comparison
        if not (ollama_ok and openai_ok):
            print("Both Ollama and OpenAI are required for comparison")
            return 1

        test_queries = [
            "What is the speed limit in urban areas?",
            "What are the requirements for vehicle registration?",
        ]

        results = compare_providers(test_queries)
        generate_comparison_report(results)

        # Save results to file
        timestamp = int(time.time())
        filename = f"comparison_results_{timestamp}.json"
        with open(filename, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {filename}")

    else:
        print("Invalid choice")
        return 1

    print("\nDemo completed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())

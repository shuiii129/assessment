import unittest
from unittest.mock import MagicMock, patch
import os
import shutil
import tempfile
import pandas as pd

from src.ingest import clean_text, chunk_document, load_metadata, ingest_documents
from src.rag import query_rag_system

class TestRAGSystem(unittest.TestCase):
    def setUp(self):
        # Create a temp directory structure for testing
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = os.path.join(self.temp_dir, "Data")
        os.makedirs(self.data_dir)
        self.fulltext_dir = os.path.join(self.data_dir, "fulltext")
        os.makedirs(self.fulltext_dir)
        self.db_path = os.path.join(self.temp_dir, "chroma_db")

        # Create dummy documents.csv metadata
        self.df = pd.DataFrame([
            {
                "AGORA ID": "1", 
                "Official name": "Document One", 
                "Casual name": "Doc 1", 
                "Link to document": "http://example.com/1", 
                "Authority": "Auth 1", 
                "Collections": "Coll 1", 
                "Short summary": "Summary 1"
            },
            {
                "AGORA ID": "2", 
                "Official name": "Document Two", 
                "Casual name": "Doc 2", 
                "Link to document": "http://example.com/2", 
                "Authority": "Auth 2", 
                "Collections": "Coll 2", 
                "Short summary": "Summary 2"
            }
        ])
        self.df.to_csv(os.path.join(self.data_dir, "documents.csv"), index=False)

        # Create dummy source text files
        with open(os.path.join(self.fulltext_dir, "1.txt"), "w", encoding="utf-8") as f:
            f.write("This is the full text of document one. It has some text content.")
        with open(os.path.join(self.fulltext_dir, "2.txt"), "w", encoding="utf-8") as f:
            f.write("This is the full text of document two. It contains different content.")

    def tearDown(self):
        # Cleanup temp directory (ignore errors due to Windows process locks on ChromaDB files)
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_clean_text(self):
        raw = "Line 1\r\nLine 2 \t with spacing.\n\nLine 3"
        cleaned = clean_text(raw)
        self.assertEqual(cleaned, "Line 1\nLine 2 with spacing.\nLine 3")

    def test_chunk_document(self):
        text = "This is a sentence. This is another sentence. And a third sentence."
        # Use small chunk_size to force multiple chunks
        chunks = chunk_document(text, chunk_size=30, chunk_overlap=5)
        self.assertTrue(len(chunks) > 0)
        # Ensure chunks don't exceed max length significantly
        for chunk in chunks:
            self.assertTrue(len(chunk) <= 35)

    def test_load_metadata(self):
        meta = load_metadata(self.data_dir)
        self.assertIn("1", meta)
        self.assertEqual(meta["1"]["title"], "Document One")
        self.assertEqual(meta["1"]["link"], "http://example.com/1")

    @patch("src.ingest.genai.Client")
    def test_ingestion_and_query(self, mock_genai_client_class):
        # Set up mock Google GenAI client
        mock_client = MagicMock()
        mock_genai_client_class.return_value = mock_client

        # Mock embed_content response for 2 files (2 batches/calls possible)
        mock_embed_response = MagicMock()
        mock_embedding_1 = MagicMock()
        mock_embedding_1.values = [0.1] * 768
        mock_embedding_2 = MagicMock()
        mock_embedding_2.values = [0.2] * 768
        mock_embed_response.embeddings = [mock_embedding_1, mock_embedding_2]
        mock_client.models.embed_content.return_value = mock_embed_response

        # Run ingestion
        ingest_documents(
            limit=None,
            chunk_size=100,
            chunk_overlap=10,
            db_path=self.db_path,
            embed_model="gemini-embedding-001",
            data_dir=self.data_dir,
            api_key="mock-key"
        )

        # Mock embed_content for query (1 call returning 1 embedding)
        mock_query_embed_response = MagicMock()
        mock_query_embedding = MagicMock()
        mock_query_embedding.values = [0.1] * 768
        mock_query_embed_response.embeddings = [mock_query_embedding]
        
        # Mock generate_content response for query synthesis
        mock_generate_response = MagicMock()
        mock_generate_response.text = "This is the generated grounded answer."

        # Assign side effects / returns to client mock
        mock_client.models.embed_content.return_value = mock_query_embed_response
        mock_client.models.generate_content.return_value = mock_generate_response

        # Run query under patched client context
        with patch("src.rag.genai.Client", return_value=mock_client):
            answer, references = query_rag_system(
                query_text="What is document one?",
                k=2,
                db_path=self.db_path,
                model="gemini-2.5-flash",
                embed_model="gemini-embedding-001",
                api_key="mock-key"
            )

        self.assertEqual(answer, "This is the generated grounded answer.")
        self.assertEqual(len(references), 2)
        # Verify references retrieved correctly
        ref_titles = [ref["title"] for ref in references]
        self.assertIn("Document One", ref_titles)
        self.assertIn("Document Two", ref_titles)

if __name__ == "__main__":
    unittest.main()

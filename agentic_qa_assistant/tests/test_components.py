"""Unit tests for core components."""

from pathlib import Path
import tempfile
import os

from agentic_qa_assistant.database import DatabaseManager, create_database
from agentic_qa_assistant.rag_pipeline import DocumentProcessor, Chunk
from agentic_qa_assistant.sql_tool import SqlValidator, QueryValidationError
from agentic_qa_assistant.router import RuleBasedRouter, ToolChoice


class TestDatabaseManager:
    """Test database management functionality."""
    
    def test_create_tables(self):
        """Test table creation."""
        db = DatabaseManager(":memory:")
        db.create_tables()
        
        # Verify tables exist (DuckDB version)
        tables = db.conn.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'").fetchall()
        table_names = [table[0] for table in tables]
        
        expected_tables = ['DIM_MODEL', 'DIM_COUNTRY', 'DIM_ORDERTYPE', 'FACT_SALES', 'FACT_SALES_ORDERTYPE']
        for table in expected_tables:
            assert table in table_names, f"Table {table} not created"
            
    def test_schema_info(self):
        """Test schema information retrieval."""
        db = DatabaseManager(":memory:")
        db.create_tables()
        
        schema = db.get_schema_info()
        assert 'DIM_MODEL' in schema
        assert len(schema['DIM_MODEL']) > 0
        
        # Check model table has expected columns
        model_columns = [col['name'] for col in schema['DIM_MODEL']]
        expected_columns = ['model_id', 'model_name', 'brand', 'segment', 'powertrain']
        for col in expected_columns:
            assert col in model_columns, f"Column {col} not found in DIM_MODEL"


class TestDocumentProcessor:
    """Test document processing functionality."""
    
    def test_chunk_creation(self):
        """Test text chunking."""
        processor = DocumentProcessor(chunk_size=50, overlap=10)
        
        # Create a temporary text file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            test_text = "This is a test document. " * 20  # Create longer text
            f.write(test_text)
            f.flush()
            temp_path = Path(f.name)
            
        try:
            chunks = processor.process_document(temp_path)
            
            assert len(chunks) > 0, "No chunks created"
            assert isinstance(chunks[0], Chunk), "Invalid chunk type"
            assert chunks[0].source == temp_path.name
            assert chunks[0].page == 1
            
        finally:
            os.unlink(temp_path)
            
    def test_metadata_extraction(self):
        """Test metadata extraction from filenames."""
        processor = DocumentProcessor()
        
        # Test Toyota manual
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Toyota UX 2023 owner's manual with maintenance information.")
            f.flush()
            temp_path = Path(f.name).rename(Path(f.name).parent / "Toyota_UX_Manual_2023.txt")
            
        try:
            chunks = processor.process_document(temp_path)
            
            assert len(chunks) > 0
            chunk = chunks[0]
            assert chunk.doc_type == "manual"
            assert chunk.brand == "Toyota"
            assert chunk.year == 2023
            
        finally:
            os.unlink(temp_path)


class TestSqlValidator:
    """Test SQL validation functionality."""
    
    def test_valid_select_query(self):
        """Test validation of valid SELECT queries."""
        validator = SqlValidator()
        
        valid_sql = "SELECT model_name, brand FROM DIM_MODEL WHERE brand = 'Toyota'"
        validated = validator.validate_sql(valid_sql)
        
        assert "SELECT" in validated
        assert "LIMIT" in validated  # Should add LIMIT if not present
        
    def test_invalid_ddl_query(self):
        """Test rejection of DDL queries."""
        validator = SqlValidator()
        
        invalid_sql = "DROP TABLE DIM_MODEL"
        
        try:
            validator.validate_sql(invalid_sql)
            assert False, "Should have raised QueryValidationError"
        except QueryValidationError as e:
            assert "Only SELECT statements are allowed" in str(e)
            
    def test_invalid_table(self):
        """Test rejection of unknown tables."""
        validator = SqlValidator()
        
        invalid_sql = "SELECT * FROM UNKNOWN_TABLE"
        
        try:
            validator.validate_sql(invalid_sql)
            assert False, "Should have raised QueryValidationError"
        except QueryValidationError as e:
            assert "not allowed" in str(e)
            
    def test_limit_enforcement(self):
        """Test LIMIT clause enforcement."""
        validator = SqlValidator()
        
        # Test without LIMIT
        sql_without_limit = "SELECT * FROM DIM_MODEL"
        validated = validator.validate_sql(sql_without_limit)
        assert "LIMIT" in validated
        
        # Test with excessive LIMIT
        sql_with_high_limit = "SELECT * FROM DIM_MODEL LIMIT 50000"
        try:
            validator.validate_sql(sql_with_high_limit)
            assert False, "Should have raised QueryValidationError"
        except QueryValidationError as e:
            assert "exceeds maximum" in str(e)


class TestRuleBasedRouter:
    """Test rule-based routing functionality."""
    
    def test_sql_routing(self):
        """Test routing to SQL for data queries."""
        router = RuleBasedRouter()
        
        sql_questions = [
            "Monthly RAV4 sales in Germany in 2024",
            "Show RAV4 sales by country in 2024", 
            "Total contracts by powertrain type"
        ]
        
        for question in sql_questions:
            decision = router.route(question)
            assert decision.decision == ToolChoice.SQL, f"Failed to route SQL question: {question}"
            assert decision.confidence > 0.6
            
    def test_rag_routing(self):
        """Test routing to RAG for policy/manual queries."""
        router = RuleBasedRouter()
        
        rag_questions = [
            "What is the Toyota warranty coverage?",
            "Where is the tire repair kit located?",
            "What are the maintenance intervals?"
        ]
        
        for question in rag_questions:
            decision = router.route(question)
            assert decision.decision in [ToolChoice.RAG, ToolChoice.HYBRID], f"Failed to route RAG question: {question}"
            
    def test_hybrid_routing(self):
        """Test routing to HYBRID for mixed queries."""
        router = RuleBasedRouter()
        
        hybrid_questions = [
            "Compare Toyota vs Lexus sales and warranty differences",
            "Show sales data and summarize warranty policies"
        ]
        
        for question in hybrid_questions:
            decision = router.route(question)
            assert decision.decision == ToolChoice.HYBRID, f"Failed to route hybrid question: {question}"
            assert decision.confidence > 0.7
            
    def test_keyword_matching(self):
        """Test keyword matching functionality."""
        router = RuleBasedRouter()
        
        # Test SQL keywords
        sql_keywords = router._match_keywords("monthly sales in germany", router.sql_keywords)
        assert 'time' in sql_keywords  # "monthly"
        assert 'metrics' in sql_keywords  # "sales"
        assert 'location' in sql_keywords  # "germany"
        
        # Test RAG keywords
        rag_keywords = router._match_keywords("warranty coverage policy", router.rag_keywords)
        assert 'warranty' in rag_keywords
        assert 'policy' in rag_keywords


class TestIntegration:
    """Integration tests for the complete system."""
    
    def test_demo_questions_routing(self):
        """Test that demo questions route correctly."""
        router = RuleBasedRouter()
        
        demo_questions = [
            ("Monthly RAV4 HEV sales in Germany in 2024", ToolChoice.SQL),
            ("What is the standard Toyota warranty for Europe?", [ToolChoice.RAG, ToolChoice.HYBRID]),
            ("Where is the tire repair kit located for the UX?", [ToolChoice.RAG, ToolChoice.HYBRID]),
            ("Compare Toyota vs Lexus SUV sales in Western Europe in 2024 and summarize key warranty differences", ToolChoice.HYBRID)
        ]
        
        for question, expected in demo_questions:
            decision = router.route(question)
            
            if isinstance(expected, list):
                assert decision.decision in expected, f"Question '{question}' routed to {decision.decision.value}, expected one of {[e.value for e in expected]}"
            else:
                assert decision.decision == expected, f"Question '{question}' routed to {decision.decision.value}, expected {expected.value}"
                
    def test_golden_sql_validation(self):
        """Test validation of golden SQL queries."""
        validator = SqlValidator()
        
        # Golden SQL from PRD examples
        golden_queries = [
            """SELECT fs.year, fs.month, SUM(fs.contracts) AS contracts 
               FROM FACT_SALES fs 
               JOIN DIM_MODEL dm ON dm.model_id = fs.model_id 
               JOIN DIM_COUNTRY dc ON dc.country_code = fs.country_code 
               WHERE dm.model_name = 'RAV4' AND dm.powertrain = 'HEV' 
               AND dc.country = 'Germany' AND fs.year = 2024 
               GROUP BY fs.year, fs.month ORDER BY fs.month""",
               
            """SELECT dm.brand, SUM(fs.contracts) AS contracts_2024 
               FROM FACT_SALES fs 
               JOIN DIM_MODEL dm ON dm.model_id = fs.model_id 
               JOIN DIM_COUNTRY dc ON dc.country_code = fs.country_code 
               WHERE fs.year = 2024 AND dm.segment = 'SUV' 
               AND dc.region = 'Western Europe' AND dm.brand IN ('Toyota','Lexus') 
               GROUP BY dm.brand ORDER BY dm.brand"""
        ]
        
        for sql in golden_queries:
            try:
                validated = validator.validate_sql(sql)
                assert "SELECT" in validated
                print(f"✓ Valid SQL: {sql[:50]}...")
            except QueryValidationError as e:
                assert False, f"Golden SQL failed validation: {e}"


def run_tests():
    """Run all tests manually."""
    import traceback
    
    test_classes = [
        TestDatabaseManager,
        TestDocumentProcessor, 
        TestSqlValidator,
        TestRuleBasedRouter,
        TestIntegration
    ]
    
    passed = 0
    failed = 0
    
    for test_class in test_classes:
        print(f"\n{'='*60}")
        print(f"Running {test_class.__name__}")
        print('='*60)
        
        instance = test_class()
        
        # Get all test methods
        test_methods = [method for method in dir(instance) if method.startswith('test_')]
        
        for test_method in test_methods:
            try:
                print(f"Running {test_method}...", end=' ')
                getattr(instance, test_method)()
                print("✓ PASSED")
                passed += 1
            except Exception as e:
                print(f"✗ FAILED: {e}")
                print(f"  {traceback.format_exc()}")
                failed += 1
                
    print(f"\n{'='*60}")
    print(f"Test Results: {passed} passed, {failed} failed")
    print('='*60)
    
    return failed == 0


if __name__ == '__main__':
    success = run_tests()
    exit(0 if success else 1)